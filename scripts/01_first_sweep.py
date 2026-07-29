"""Sweep a point function over a Cartesian space, and cache it.

Run with::

    pixi run -e dev python scripts/01_first_sweep.py

Shows the three things a first user needs: the space builds the grid, the
plan tells you the cost before you pay it, and the second run pays nothing.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import xarray as xr

from xsweep import SweepPolicy, sweep

CALLS = {"n": 0}


@sweep("loop(aot, rh, sza) vec(wl @ 8) -> tdir(wl)", version="1")
def tdir(aot: float, rh: float, sza: float, wl: xr.DataArray) -> xr.DataArray:
    """Stand in for an engine call that would take seconds."""
    CALLS["n"] += 1
    time.sleep(0.002)
    return np.exp(-aot * (1.0 + rh / 100.0) / np.cos(np.radians(sza))) * wl / wl[0]


def main() -> None:
    """Run the sweep twice against the same store."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    space = xr.Dataset(
        {
            "aot": ("aot", [0.05, 0.1, 0.3]),
            "rh": ("rh", [30.0, 70.0]),
            "sza": ("sza", [0.0, 30.0, 60.0]),
            "wl": ("wl", np.arange(400.0, 900.0, 25.0)),
        }
    )

    store = Path(tempfile.mkdtemp()) / "tdir.zarr"
    try:
        print("=== What it will cost, without calling anything ===\n")
        print(tdir.explain(space))

        print("\n=== First run ===\n")
        result = tdir(space, policy=SweepPolicy(store=str(store)))
        print(f"\ncalls made: {CALLS['n']}")
        print(f"result: {dict(result.sizes)}")
        print(f"still lazy: {type(result.tdir.variable._data).__name__}")

        print("\n=== Second run, same store ===\n")
        CALLS["n"] = 0
        again = tdir(space, policy=SweepPolicy(store=str(store)))
        print(f"\ncalls made: {CALLS['n']}")
        print(f"same values: {np.array_equal(result.tdir.values, again.tdir.values)}")
    finally:
        shutil.rmtree(store.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
