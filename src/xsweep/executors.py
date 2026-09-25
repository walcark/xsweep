"""Executors: how calls are dispatched, never what they mean.

The protocol is deliberately narrow. A sweep is embarrassingly parallel at
the work-item level, so a map is all the core needs, and keeping the surface
this small is what lets dask stay an optional backend rather than a
foundation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Protocol, TypeVar

from .errors import PolicyError

__all__ = [
    "Executor",
    "ProcessExecutor",
    "SerialExecutor",
    "ThreadExecutor",
    "build_executor",
]

T = TypeVar("T")
R = TypeVar("R")


class Executor(Protocol):
    """Dispatch calls and stream back results as they complete."""

    def map_unordered(
        self, fn: Callable[[T], R], items: Sequence[T]
    ) -> Iterator[tuple[int, R]]:
        """Apply ``fn`` to each item, yielding ``(position, result)`` pairs.

        Results may arrive in any order, hence the position: it is what lets
        the caller place each result without relying on completion order.
        """
        ...


class SerialExecutor:
    """Run everything in the calling process, in order.

    The right default: GPU callees want one call at a time per device anyway,
    and a serial run keeps tracebacks readable.
    """

    def map_unordered(
        self, fn: Callable[[T], R], items: Sequence[T]
    ) -> Iterator[tuple[int, R]]:
        """Apply ``fn`` to each item in order."""
        for position, item in enumerate(items):
            yield position, fn(item)


class ProcessExecutor:
    """Run calls in a process pool.

    Parameters
    ----------
    max_workers
        Worker count; ``None`` leaves the runtime default.

    Notes
    -----
    The wrapped callable and every static argument must be picklable, which
    in practice means module-level functions. A pickling failure is caught
    and re-raised naming the culprit, because the raw error names only the
    type and sends users hunting.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers

    def map_unordered(
        self, fn: Callable[[T], R], items: Sequence[T]
    ) -> Iterator[tuple[int, R]]:
        """Apply ``fn`` to each item across worker processes."""
        import pickle

        try:
            pickle.dumps(fn)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise PolicyError(
                "the process executor requires a picklable callable, and "
                f"{getattr(fn, '__qualname__', fn)!r} is not: {exc}. Define "
                "the swept function at module level, or use the serial "
                "executor"
            ) from exc

        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(fn, item): position for position, item in enumerate(items)
            }
            from concurrent.futures import as_completed

            for future in as_completed(futures):
                yield futures[future], future.result()


class ThreadExecutor:
    """Run calls in a thread pool.

    Parameters
    ----------
    max_workers
        Worker count; ``None`` leaves the runtime default.

    Notes
    -----
    For a callee that releases the GIL: a subprocess, a C extension, a wait
    on I/O. Nothing is pickled, so a closure over live objects runs here
    where the process executor refuses it, which is the usual shape when the
    swept function is a method or captures a loaded dataset. A callee that
    holds the GIL gains nothing from this executor.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers

    def map_unordered(
        self, fn: Callable[[T], R], items: Sequence[T]
    ) -> Iterator[tuple[int, R]]:
        """Apply ``fn`` to each item across worker threads."""
        from concurrent.futures import as_completed

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(fn, item): position for position, item in enumerate(items)
            }
            for future in as_completed(futures):
                yield futures[future], future.result()


def build_executor(spec: str | Any, *, max_workers: int | None = None) -> Executor:
    """Turn a policy value into an executor.

    Parameters
    ----------
    spec
        ``"serial"``, ``"thread"``, ``"process"``, ``"dask"``, or an object
        already satisfying the protocol.
    max_workers
        Worker count, honoured by the thread and process executors.

    Returns
    -------
    Executor
        The executor to dispatch with.

    Raises
    ------
    PolicyError
        If the name is unknown or the object lacks ``map_unordered``.
    """
    if not isinstance(spec, str):
        if not hasattr(spec, "map_unordered"):
            raise PolicyError(
                f"{type(spec).__name__} is not an executor: it must expose "
                "map_unordered(fn, items)"
            )
        custom: Executor = spec
        return custom
    if spec == "serial":
        return SerialExecutor()
    if spec == "thread":
        return ThreadExecutor(max_workers)
    if spec == "process":
        return ProcessExecutor(max_workers)
    if spec == "dask":
        return _build_dask_executor(max_workers)
    raise PolicyError(
        f"unknown executor {spec!r}; expected 'serial', 'thread', 'process', "
        "'dask', or an object exposing map_unordered"
    )


def _build_dask_executor(max_workers: int | None) -> Executor:
    """Build the dask-backed executor, importing dask only here."""
    try:
        from dask.distributed import Client  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise PolicyError(
            "the dask executor needs the optional dependency; install it with "
            "pip install 'xsweep[dask]'"
        ) from exc
    raise PolicyError(
        "the dask executor is designed but not implemented in v0; use "
        "'serial' or 'process'"
    )
