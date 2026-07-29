"""The store write lock.

A store tolerates one writing run at a time. Without this, launching the same
sweep twice by accident lets two processes interleave region writes into the
same arrays, and hours of engine time are lost to results nobody can trust.

Readers are never blocked: inspecting partial results and the status variable
while a sweep runs is the supported way to follow it.

Deliberately absent: heartbeats and automatic staleness detection. A dead
process on another host cannot be probed reliably, and a wrong guess either
blocks a legitimate run or lets two writers proceed. A manual override is
honest and one flag away.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import time
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any

from .errors import StoreLockedError

__all__ = ["LOCK_SUFFIX", "StoreLock"]

LOCK_SUFFIX = ".lock"


class StoreLock:
    """Hold the exclusive write lock on a store directory.

    Parameters
    ----------
    root
        The store directory.
    force
        Break a lock left behind by a dead run instead of refusing.
    """

    def __init__(self, root: str | Path, *, force: bool = False) -> None:
        # Beside the store, not inside it: a stray file in a zarr
        # directory makes every open warn about an unrecognised component.
        root = Path(root)
        self.path = root.with_name(root.name + LOCK_SUFFIX)
        self.force = force
        self._held = False
        self._previous: Any = None

    def acquire(self) -> None:
        """Take the lock, or refuse naming whoever holds it.

        Raises
        ------
        StoreLockedError
            If another run holds the lock and ``force`` is not set.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.force:
            self.path.unlink(missing_ok=True)
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "since": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise StoreLockedError(
                f"{self.path.with_suffix('')} is being written by another run "
                f"({self._owner()}). Wait for it to finish, or pass "
                "force_unlock=True if that run is dead"
            ) from exc
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        self._held = True
        self._install_signal_handler()

    def release(self) -> None:
        """Give the lock back, if this object holds it."""
        if not self._held:
            return
        self.path.unlink(missing_ok=True)
        self._held = False
        self._restore_signal_handler()

    def _owner(self) -> str:
        """Describe the current holder, for the refusal message."""
        try:
            meta = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return "unknown owner"
        return f"pid {meta.get('pid')} on {meta.get('host')} since {meta.get('since')}"

    def _install_signal_handler(self) -> None:
        """Release on SIGTERM, which would otherwise skip every finally."""

        def handler(signum: int, frame: FrameType | None) -> None:
            self.release()
            raise KeyboardInterrupt(f"interrupted by signal {signum}")

        try:
            self._previous = signal.signal(signal.SIGTERM, handler)
        except ValueError:
            # Not the main thread; the context manager still covers normal
            # exits and exceptions, which is the common case.
            self._previous = None

    def _restore_signal_handler(self) -> None:
        """Put back whatever handler was installed before."""
        if self._previous is None:
            return
        try:
            signal.signal(signal.SIGTERM, self._previous)
        except ValueError:
            pass
        self._previous = None

    def __enter__(self) -> StoreLock:
        """Acquire on entry."""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release on exit, including when the body raised."""
        self.release()
