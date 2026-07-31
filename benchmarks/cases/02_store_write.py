"""Persisted sweep: what the store's write path costs per point.

The same 40000 points as the in-memory case, now streamed into a zarr store.
The difference between the two is the bookkeeping the store adds, which is
what the chunk-grid design in docs/reference/limitations.md is about.

Also times the second pass over a finished store, which makes no calls at all
and measures only the "already done" path.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import xarray as xr

import xsweep

sys.path.insert(0, str(Path(xsweep.__file__).resolve().parents[2]))

from benchmarks._lib.timing import record  # noqa: E402
from xsweep import SweepPolicy, sweep  # noqa: E402

SIDE = 200


@sweep("loop(a, b) -> out()")
def cheap(a: float, b: float) -> float:
    """As close to free as a Python call gets."""
    return a * b


def main() -> None:
    """Time a persisted sweep, then a fully cached second pass."""
    space = xr.Dataset(
        {
            "a": ("a", np.linspace(0.0, 1.0, SIDE)),
            "b": ("b", np.linspace(0.0, 1.0, SIDE)),
        }
    )
    run_dir = Path(tempfile.mkdtemp())
    policy = SweepPolicy(store=run_dir / "grid.zarr")

    try:
        start = time.perf_counter()
        cheap(space, policy=policy)
        record(
            "store_write",
            n_points=SIDE * SIDE,
            n_calls=SIDE * SIDE,
            wall_time_s=time.perf_counter() - start,
        )

        start = time.perf_counter()
        cheap(space, policy=policy)
        record(
            "store_fully_cached",
            n_points=SIDE * SIDE,
            n_calls=0,
            wall_time_s=time.perf_counter() - start,
        )
    finally:
        shutil.rmtree(run_dir)


if __name__ == "__main__":
    main()
