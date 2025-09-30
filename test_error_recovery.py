# this_file: test_error_recovery.py
"""Test error recovery mechanisms under simulated failure conditions.

This script validates that retry logic, circuit breakers, and degradation
strategies work correctly under various failure scenarios.
"""

from __future__ import annotations

import asyncio
import time

from uutel.core import (
    CircuitBreaker,
    CircuitState,
    RetryConfig,
    get_circuit_breaker,
    retry_with_backoff,
    retry_with_backoff_async,
)
from uutel.core.exceptions import UUTELError
from uutel.core.logging_config import get_logger

logger = get_logger(__name__)


def simulate_network_failure() -> str:
    """Simulate a network failure that should trigger retries."""
    import httpx

    raise httpx.ConnectTimeout("Simulated network timeout")


def simulate_intermittent_failure(attempt_count: list[int]) -> str:
    """Simulate a failure that succeeds after a few attempts."""
    attempt_count[0] += 1
    if attempt_count[0] < 3:
        import httpx

        raise httpx.HTTPStatusError(
            f"Simulated failure (attempt {attempt_count[0]})",
            request=None,
            response=None,
        )
    return f"Success after {attempt_count[0]} attempts"


def simulate_permanent_failure() -> str:
    """Simulate a permanent failure that should exhaust all retries."""
    raise UUTELError("Simulated permanent failure", error_code="PERMANENT_ERROR")


async def simulate_async_recovery(attempt_count: list[int]) -> str:
    """Simulate async operation that recovers after retries."""
    attempt_count[0] += 1
    await asyncio.sleep(0.01)  # Simulate async work
    if attempt_count[0] < 2:
        import httpx

        raise httpx.ConnectTimeout(f"Async failure (attempt {attempt_count[0]})")
    return f"Async success after {attempt_count[0]} attempts"


def test_retry_with_recovery() -> None:
    """Test that retry logic works when operation eventually succeeds."""
    print("🔄 Testing retry with recovery...")

    retry_config = RetryConfig(
        max_retries=5,
        backoff_factor=1.1,  # Fast backoff for testing
        use_enhanced_categorization=True,  # Use enhanced error categorization
    )

    attempt_count = [0]

    try:
        result = retry_with_backoff(
            simulate_intermittent_failure, retry_config, attempt_count
        )
        print(f"✅ Retry with recovery: {result}")
        return True
    except Exception as e:
        print(f"❌ Retry with recovery failed: {e}")
        return False


def test_retry_with_permanent_failure() -> None:
    """Test that retry logic eventually gives up on permanent failures."""
    print("🔄 Testing retry with permanent failure...")

    retry_config = RetryConfig(
        max_retries=2,  # Small number for faster testing
        backoff_factor=1.1,
        use_enhanced_categorization=True,  # Use enhanced error categorization
    )

    try:
        result = retry_with_backoff(simulate_permanent_failure, retry_config)
        print(f"❌ Should not succeed: {result}")
        return False
    except UUTELError as e:
        print(f"✅ Correctly failed after retries: {e}")
        return True


async def test_async_retry_with_recovery() -> bool:
    """Test async retry logic with recovery."""
    print("🔄 Testing async retry with recovery...")

    retry_config = RetryConfig(
        max_retries=3,
        backoff_factor=1.2,
        use_enhanced_categorization=True,  # Use enhanced error categorization
    )

    attempt_count = [0]

    try:
        result = await retry_with_backoff_async(
            simulate_async_recovery, retry_config, attempt_count
        )
        print(f"✅ Async retry with recovery: {result}")
        return True
    except Exception as e:
        print(f"❌ Async retry failed: {e}")
        return False


def test_circuit_breaker() -> None:
    """Test circuit breaker functionality."""
    print("⚡ Testing circuit breaker...")

    from uutel.core.utils import CircuitBreakerConfig

    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
    breaker = CircuitBreaker(provider_name="test_breaker", config=config)

    # Test initial closed state
    if breaker.state != CircuitState.CLOSED:
        print(f"❌ Circuit breaker should start closed, got {breaker.state}")
        return False

    # Record failures to trip the breaker
    import httpx

    breaker.record_failure(httpx.ConnectTimeout("Test failure 1"))
    if breaker.state != CircuitState.CLOSED:
        print(
            f"❌ Circuit breaker should still be closed after 1 failure, got {breaker.state}"
        )
        return False

    breaker.record_failure(httpx.ConnectTimeout("Test failure 2"))
    if breaker.state != CircuitState.OPEN:
        print(
            f"❌ Circuit breaker should be open after 2 failures, got {breaker.state}"
        )
        return False

    print("✅ Circuit breaker opened after threshold failures")

    # Wait for recovery timeout
    time.sleep(1.1)

    # Next call should put it in half-open state
    breaker.record_success()
    if breaker.state != CircuitState.CLOSED:
        print(f"❌ Circuit breaker should be closed after success, got {breaker.state}")
        return False

    print("✅ Circuit breaker recovered and closed after success")
    return True


def test_circuit_breaker_integration() -> None:
    """Test circuit breaker with actual provider name."""
    print("⚡ Testing circuit breaker integration...")

    # Get circuit breaker for test provider
    breaker = get_circuit_breaker("test_provider")

    # Record some failures
    import httpx

    for i in range(3):
        breaker.record_failure(httpx.ConnectTimeout(f"Test failure {i + 1}"))

    # Should be open now
    if breaker.state != CircuitState.OPEN:
        print(f"❌ Circuit breaker should be open, got {breaker.state}")
        return False

    print(f"✅ Circuit breaker for 'test_provider' is {breaker.state}")

    # Reset for next test
    breaker.reset()
    print("✅ Circuit breaker reset")
    return True


def run_error_recovery_tests() -> bool:
    """Run all error recovery tests."""
    print("🧪 Starting error recovery tests...")
    print("=" * 50)

    results = []

    # Test retry scenarios
    results.append(test_retry_with_recovery())
    results.append(test_retry_with_permanent_failure())

    # Test async retry
    try:
        async_result = asyncio.run(test_async_retry_with_recovery())
        results.append(async_result)
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        results.append(False)

    # Test circuit breaker
    results.append(test_circuit_breaker())
    results.append(test_circuit_breaker_integration())

    # Summary
    print("=" * 50)
    passed = sum(results)
    total = len(results)

    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("✅ All error recovery tests passed!")
        return True
    else:
        print("❌ Some error recovery tests failed!")
        return False


if __name__ == "__main__":
    success = run_error_recovery_tests()
    exit(0 if success else 1)
