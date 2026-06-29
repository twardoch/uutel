# this_file: src/uutel/core/runners.py
"""Subprocess execution utilities.

This module handles running CLI tools (like `claude` or `gemini`) in the background.
It provides standard synchronous execution, streaming output (for real-time typing effects), 
and async support.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

from uutel.core.exceptions import UUTELError
from uutel.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SubprocessResult:
    """What happened when we ran the command.

    Attributes:
        command: The exact command array we tried to execute.
        returncode: The exit code (0 usually means success).
        stdout: The raw standard output text.
        stderr: The raw standard error text (where CLIs usually complain).
        duration_seconds: How long the execution took.
    """

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


def _build_env(extra_env: Mapping[str, str] | None) -> dict[str, str]:
    """Merge extra environment variables with the system environment.

    Useful when passing temporary auth tokens or config flags to a CLI tool 
    without polluting the global environment.
    """

    merged = os.environ.copy()
    if extra_env:
        merged.update(extra_env)
    return merged


def run_subprocess(
    command: Sequence[str],
    *,
    check: bool = True,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    encoding: str = "utf-8",
) -> SubprocessResult:
    """Fire off a command and wait for the final text.

    Args:
        command: The command array (e.g., ["claude", "login"]).
        check: If True, raise an error if the exit code isn't 0.
        cwd: Where to run the command.
        env: Extra environment variables.
        timeout: Stop waiting after this many seconds.
        encoding: How to decode the raw bytes (usually utf-8).

    Raises:
        UUTELError: If the command isn't found or times out.
        subprocess.CalledProcessError: If check=True and the command fails.
        
    Returns:
        A SubprocessResult with the exit code and outputs.
    """

    start = perf_counter()
    logger.debug(
        "Running subprocess", command=list(command), cwd=str(cwd) if cwd else None
    )
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=_build_env(env),
        check=False,
        capture_output=True,
        text=True,
        encoding=encoding,
        errors="replace",
        timeout=timeout,
    )
    duration = perf_counter() - start
    result = SubprocessResult(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=duration,
    )

    if check and completed.returncode != 0:
        message = f"Command {command!r} exited with exit code {completed.returncode}."
        if completed.stderr:
            message = f"{message} Stderr: {completed.stderr.strip()}"
        raise UUTELError(message, provider="subprocess")

    return result


def stream_subprocess_lines(
    command: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    encoding: str = "utf-8",
) -> Iterator[str]:
    """Yield stdout lines from a subprocess as they are produced."""

    logger.debug(
        "Streaming subprocess", command=list(command), cwd=str(cwd) if cwd else None
    )
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=_build_env(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding=encoding,
        errors="replace",
        bufsize=1,
    )

    try:
        assert process.stdout is not None  # for type-checkers
        for line in iter(process.stdout.readline, ""):
            yield line.rstrip("\r\n")

        # Wait for completion to capture errors/timeout
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - exercised via raise
        process.kill()
        process.wait()
        raise UUTELError(
            f"Command {command!r} timed out after {timeout} seconds",
            provider="subprocess",
        ) from exc
    finally:
        if process.stdout is not None:
            process.stdout.close()
    stderr_text = ""
    if process.stderr is not None:
        stderr_text = process.stderr.read()
        process.stderr.close()

    if process.returncode not in (0, None):
        message = f"Command {command!r} exited with {process.returncode}"
        if stderr_text:
            message = f"{message}. Stderr: {stderr_text.strip()}"
        raise UUTELError(message, provider="subprocess")


async def astream_subprocess_lines(
    command: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    encoding: str = "utf-8",
) -> AsyncIterator[str]:
    """Asynchronously yield stdout lines from a subprocess."""

    logger.debug(
        "Async streaming subprocess",
        command=list(command),
        cwd=str(cwd) if cwd else None,
    )

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        env=_build_env(env),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    assert process.stdout is not None
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        yield line.decode(encoding, errors="replace").rstrip("\r\n")

    stderr_text = ""
    if process.stderr is not None:
        stderr_bytes = await process.stderr.read()
        stderr_text = stderr_bytes.decode(encoding, errors="replace")

    returncode = await process.wait()
    if returncode != 0:
        message = f"Command {command!r} exited with {returncode}"
        if stderr_text:
            message = f"{message}. Stderr: {stderr_text.strip()}"
        raise UUTELError(message, provider="subprocess")
