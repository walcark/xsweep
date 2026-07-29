"""Minimal, content-agnostic parameter sweeps for xarray.

Lift an expensive point function into a gridded, cached, resumable xarray
computation:

    >>> import numpy as np, xarray as xr
    >>> from xsweep import sweep
    >>> @sweep("loop(a, b) -> out()")
    ... def f(a: float, b: float) -> float:
    ...     return a * b
    >>> space = xr.Dataset({"a": ("a", [1.0, 2.0]), "b": ("b", [10.0, 20.0])})
    >>> result = f(space)

Semantics come from xarray, not from a second description: variables sharing
a dim vary together, variables on distinct dims multiply.
"""

from .contract import Contract, LoopVar, OutVar, VecVar
from .errors import (
    ContractError,
    PointFailed,
    PolicyError,
    SpaceError,
    StoreError,
    StoreLockedError,
    XsweepError,
)
from .module import SweepModule
from .plan import Plan
from .policy import SweepPolicy
from .sweeper import Sweeper, sweep

__all__ = [
    "Contract",
    "ContractError",
    "LoopVar",
    "OutVar",
    "Plan",
    "PointFailed",
    "PolicyError",
    "SpaceError",
    "StoreError",
    "StoreLockedError",
    "SweepModule",
    "SweepPolicy",
    "Sweeper",
    "VecVar",
    "XsweepError",
    "sweep",
]
__version__ = "0.1.0"
