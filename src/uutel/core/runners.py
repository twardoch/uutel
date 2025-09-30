# this_file: src/uutel/core/runners.py
"""Subprocess execution utilities with streaming support."""

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
    """Container for subprocess execution metadata."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


def _build_env(extra_env: Mapping[str, str] | None) -> dict[str, str]:
    """Merge provided environment variables with process defaults."""

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
    """Run a subprocess and return captured output.

    Raises:
        UUTELError: if the command exits with non-zero status and ``check`` is True
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
