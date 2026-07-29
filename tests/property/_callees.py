"""Module-level callees for the property suite.

They live in a module rather than inside a test because the process executor
pickles the sweeper by reference: a worker re-imports this module and finds
the very same object.
"""

from __future__ import annotations

import xarray as xr

from xsweep import sweep


@sweep("loop(aot, rh) -> rho()", version="1")
def scalar_engine(aot: float, rh: float) -> float:
    """Stand in for a scalar-only engine call."""
    return aot * 1000.0 + rh


@sweep("loop(aot) vec(wl) -> t(wl)", version="1")
def vector_engine(aot: float, wl: xr.DataArray) -> xr.DataArray:
    """Stand in for an engine accepting a whole wavelength axis."""
    return wl * aot
