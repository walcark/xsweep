"""Executor selection and dispatch: the paths a real run would take."""

from __future__ import annotations

import pytest

from xsweep.errors import PolicyError
from xsweep.executors import ProcessExecutor, SerialExecutor, build_executor


def test_a_non_picklable_callable_is_refused_with_the_culprit_named() -> None:
    """The raw pickling error names only a type; this names the function."""

    def local_closure(x: int) -> int:
        return x * 2

    executor = ProcessExecutor()
    with pytest.raises(PolicyError, match="requires a picklable callable"):
        next(executor.map_unordered(local_closure, [1, 2, 3]))


def test_build_executor_accepts_a_custom_object() -> None:
    """Anything exposing map_unordered is a valid executor, not just the two names."""

    class Custom:
        def map_unordered(self, fn, items):  # type: ignore[no-untyped-def]
            for i, item in enumerate(items):
                yield i, fn(item)

    custom = Custom()
    assert build_executor(custom) is custom


def test_build_executor_rejects_an_object_without_map_unordered() -> None:
    """A typo'd or unrelated object must fail before any call, not mid-run."""
    with pytest.raises(PolicyError, match="must expose"):
        build_executor(object())


def test_build_executor_serial_and_process() -> None:
    """The two built-in names resolve to their executors."""
    assert isinstance(build_executor("serial"), SerialExecutor)
    assert isinstance(build_executor("process"), ProcessExecutor)


def test_build_executor_unknown_name_is_refused() -> None:
    """A typo'd executor name is refused, not silently defaulted."""
    with pytest.raises(PolicyError, match="unknown executor"):
        build_executor("threaded")


def test_dask_executor_names_the_missing_extra() -> None:
    """dask is optional: the message must say how to get it, not just fail.

    Only meaningful without dask installed; skipped otherwise rather than
    asserting a message that would then legitimately not apply.
    """
    try:
        import dask  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("dask is installed; the missing-extra message does not apply")
    with pytest.raises(PolicyError, match="optional dependency"):
        build_executor("dask")
