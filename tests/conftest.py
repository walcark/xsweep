"""Shared fixtures.

The central tool is :class:`CountingCallee`, which records every call it
receives. Most acceptance criteria in the specification are stated as call
counts ("exactly 12 calls", "zero calls on re-run", "the plan never calls the
function"), so nearly every test asserts on one of these recorders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import xarray as xr


@dataclass
class Call:
    """One recorded invocation.

    Attributes
    ----------
    args
        Positional arguments received.
    kwargs
        Keyword arguments received.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class CountingCallee:
    """Wrap a function and record every call made to it.

    Parameters
    ----------
    func
        The function to wrap.
    fail_on
        Optional predicate on the keyword arguments; when it returns ``True``
        the call raises :class:`RuntimeError` instead of returning.
    """

    func: Any
    fail_on: Any = None
    calls: list[Call] = field(default_factory=list)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record the call, then delegate to the wrapped function."""
        self.calls.append(Call(args, dict(kwargs)))
        if self.fail_on is not None and self.fail_on(kwargs):
            raise RuntimeError(f"callee failed on {kwargs}")
        return self.func(*args, **kwargs)

    @property
    def n_calls(self) -> int:
        """Return the number of calls recorded so far."""
        return len(self.calls)

    def reset(self) -> None:
        """Forget every recorded call."""
        self.calls.clear()


@pytest.fixture
def cartesian_space() -> xr.Dataset:
    """Return a small Cartesian space: ``a`` of size 3 and ``b`` of size 4."""
    return xr.Dataset(
        {
            "a": ("a", np.array([0.1, 0.2, 0.3])),
            "b": ("b", np.array([10.0, 20.0, 30.0, 40.0])),
        }
    )


@pytest.fixture
def map_space() -> xr.Dataset:
    """Return a zipped space: two variables sharing the dims ``(y, x)``.

    The values repeat on purpose, so deduplication has something to collapse:
    four unique ``(aot, rh)`` rows among six pixels.
    """
    aot = np.array([[0.1, 0.2, 0.1], [0.3, 0.2, 0.1]])
    rh = np.array([[30.0, 70.0, 30.0], [50.0, 70.0, 30.0]])
    return xr.Dataset(
        {
            "aot": (("y", "x"), aot),
            "rh": (("y", "x"), rh),
        }
    )
