# this_file: tests/test_runners.py
"""Tests for subprocess runner utilities."""

from __future__ import annotations

import asyncio

import pytest

from uutel.core.exceptions import UUTELError
from uutel.core.runners import (
    SubprocessResult,
    astream_subprocess_lines,
    run_subprocess,
    stream_subprocess_lines,
)


def _python_command(code: str) -> list[str]:
    """Return a Python one-liner command array."""

    return ["python", "-c", code]


def test_run_subprocess_returns_stdout() -> None:
    """run_subprocess should capture stdout and metadata."""

    result = run_subprocess(_python_command("print('hello')"))
    assert isinstance(result, SubprocessResult)
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"
    assert result.stderr == ""


def test_run_subprocess_raises_on_nonzero_exit() -> None:
    """Non-zero exit codes should raise a UUTELError when check=True."""

    with pytest.raises(UUTELError) as excinfo:
        run_subprocess(_python_command("import sys; sys.exit(3)"), check=True)
    assert "exit code 3" in str(excinfo.value)


def test_stream_subprocess_lines_yields_incremental_output() -> None:
    """Streaming helper should yield each line as it arrives."""

    command = _python_command(
        "import sys,time;\nfor value in range(2):\n    print(f'chunk {value}')\n    sys.stdout.flush()\n    time.sleep(0.01)"
    )

    chunks = list(stream_subprocess_lines(command))
    assert chunks == ["chunk 0", "chunk 1"], f"Unexpected chunks: {chunks}"


def test_astream_subprocess_lines_produces_async_chunks() -> None:
    """Async streaming should produce identical output to sync version."""

    command = _python_command(
        "import sys,asyncio;\nasync def main():\n    for value in range(2):\n        print(f'async {value}')\n        sys.stdout.flush()\n        await asyncio.sleep(0.01)\nasyncio.run(main())"
    )

    async def _collect() -> list[str]:
        output: list[str] = []
        async for chunk in astream_subprocess_lines(command):
            output.append(chunk)
        return output

    output = asyncio.run(_collect())
    assert output == ["async 0", "async 1"], f"Unexpected async chunks: {output}"
