# UUTEL Development Guide

> Quick-start guide for developers contributing to UUTEL

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for package management
- [hatch](https://hatch.pypa.io/) for development workflow

## Quick Setup

```bash
# 1. Clone and enter directory
git clone https://github.com/twardoch/uutel.git
cd uutel

# 2. Create development environment
hatch shell

# 3. Verify installation
python -c "import uutel; print(uutel.__version__)"

# 4. Run tests to ensure everything works
hatch run test
```

## Common Development Commands

### Testing Commands

```bash
# Run all tests (recommended - parallel execution)
hatch run test

# Run tests without parallel execution (for debugging)
hatch run test-single

# Run only CI-safe tests (excludes performance-sensitive tests)
hatch run test-ci

# Run performance tests separately
hatch run test-performance

# Run with coverage report
hatch run test-cov

# Run specific test file
hatch run test tests/test_utils.py

# Run specific test function
hatch run test tests/test_utils.py::test_validate_model_name
```

### Code Quality Commands

```bash
# Run all linting checks
hatch run lint

# Auto-format code
hatch run format

# Run type checking
hatch run typecheck

# Check + format (combined)
hatch run check
```

### Make Commands (Alternative)

```bash
# Quick test (uses make commands from Makefile)
make test

# Full quality check
make check

# Show all available commands
make help
```

## Development Workflow

### 1. Before Starting Work

```bash
# Update to latest main
git checkout main
git pull

# Create feature branch
git checkout -b feature/your-feature-name

# Ensure tests pass
hatch run test
```

### 2. During Development

```bash
# Test frequently
hatch run test tests/test_your_module.py

# Check code quality
hatch run lint
hatch run format

# Verify types
hatch run typecheck
```

### 3. Before Committing

```bash
# Run full test suite
hatch run test

# Run all quality checks
hatch run check

# Check security
uvx bandit -r src/

# Manual final verification
make check
```

## Project Structure

```
uutel/
├── src/uutel/           # Main package code
│   ├── core/            # Core infrastructure
│   │   ├── base.py      # BaseUU class
│   │   ├── auth.py      # Authentication framework
│   │   ├── utils.py     # Utility functions
│   │   └── exceptions.py # Custom exceptions
│   └── providers/       # Provider implementations (future)
├── tests/               # Test suite
├── examples/            # Usage examples
├── scripts/             # Development scripts
└── docs/               # Documentation
```

## Writing Tests

### Test File Organization

```bash
# Test files mirror source structure
src/uutel/core/utils.py  →  tests/test_utils.py
src/uutel/core/base.py   →  tests/test_base.py
```

### Test Naming Convention

```python
def test_function_name_when_condition_then_result():
    """Test description."""
    # Test implementation
```

### Running Specific Test Categories

```bash
# Performance tests only
hatch run test -m performance

# Integration tests only
hatch run test -m integration

# Skip slow tests
hatch run test -m "not slow"
```

## Debugging

### Common Issues and Solutions

#### Tests Failing in Parallel Mode

```bash
# Run without parallel execution
hatch run test-single

# Check if it's a performance test issue
hatch run test-performance

# Check specific failing test
hatch run test tests/test_file.py::test_name -v
```

#### Import Errors

```bash
# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Verify package installation
python -c "import uutel; print(uutel.__file__)"
```

#### Type Check Failures

```bash
# Check specific file
uvx mypy src/uutel/core/utils.py

# Show all type issues
hatch run typecheck --show-traceback
```

#### Linting Issues

```bash
# Auto-fix most issues
hatch run format

# Check what can't be auto-fixed
hatch run lint --diff

# Line-length violations
hatch run lint | grep E501
```

### Performance Debugging

```bash
# Run with timing information
hatch run test --durations=10

# Profile slow tests
hatch run test tests/test_performance.py -v --tb=short

# Memory profiling
python -m pytest tests/test_memory.py -v
```

## Release Process

### Version Management

```bash
# Current version
python -c "import uutel; print(uutel.__version__)"

# Version is auto-managed by hatch-vcs from git tags
git tag v1.0.5
git push origin v1.0.5
```

### Pre-Release Checklist

- [ ] All tests pass: `hatch run test`
- [ ] Code quality checks pass: `hatch run check`
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped (git tag)

## IDE Configuration

### VS Code Settings

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff"
}
```

### PyCharm Settings

- Interpreter: Use hatch environment
- Test Runner: pytest
- Code Style: Follow .ruff.toml configuration

## Contributing Guidelines

### Code Standards

- Follow PEP 8 (enforced by ruff)
- Write docstrings for all public functions
- Include type hints
- Add tests for new functionality
- Keep functions under 20 lines when possible

### Commit Convention

```bash
# Use conventional commit format
feat: add new utility function
fix: resolve memory leak in client creation
docs: update development guide
test: add edge case tests for validation
```

### Pull Request Process

1. Create feature branch from main
2. Implement changes with tests
3. Run full test suite: `hatch run test`
4. Run quality checks: `hatch run check`
5. Update documentation as needed
6. Submit PR with clear description

## Environment Variables

### Development

```bash
# Enable debug logging
export UUTEL_LOG_LEVEL=DEBUG

# Skip slow tests
export PYTEST_SKIP_SLOW=1

# Use test database
export UUTEL_TEST_MODE=1
```

### CI/CD

```bash
# CI indicator (auto-detected)
export CI=true

# Parallel execution worker (auto-set by pytest-xdist)
export PYTEST_XDIST_WORKER=gw0
```

## Troubleshooting

### "Tests hang" or "Very slow execution"

```bash
# Check if running with correct command
hatch run test  # ✅ Correct (parallel)
uvx hatch test  # ❌ May have issues with async

# Check for resource conflicts
top | grep python
ps aux | grep pytest
```

### "Module not found" errors

```bash
# Reinstall package in development mode
pip install -e .

# Check installation
pip list | grep uutel

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -exec rm -rf {} +
```

### "Coverage too low"

```bash
# Run with coverage details
hatch run test-cov --cov-report=html

# Check coverage report
open htmlcov/index.html

# Find uncovered lines
hatch run test-cov --cov-report=term-missing
```

## Advanced Development

### Memory Profiling

```bash
# Run memory tests
hatch run test tests/test_memory.py -v

# Profile specific function
python -m memory_profiler script.py
```

### Performance Optimization

```bash
# Profile performance
python -m cProfile -o profile.stats script.py

# Analyze results
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('time').print_stats(10)"
```

### Security Scanning

```bash
# Security scan
uvx bandit -r src/

# Dependency vulnerability check
uvx safety check

# Check for secrets
uvx detect-secrets scan
```

---

## Quick Reference Card

```bash
# Essential commands
hatch shell                    # Enter dev environment
hatch run test                 # Run all tests
hatch run test-single          # Debug tests (no parallel)
hatch run lint                 # Check code style
hatch run format               # Auto-format code
hatch run typecheck            # Type checking
make test                      # Quick test via make
make check                     # Full quality check
```

For more help: `make help` or check [CONTRIBUTING.md](CONTRIBUTING.md)