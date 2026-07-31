"""Resume: what a partly finished store costs to pick back up.

A 40000-point store where a quarter of the points failed, then a second pass
that recomputes only those. The interesting number is the resume pass, which
has to read 40000 statuses to find the 10000 it still owes.
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
STATE = {"broken": True}


@sweep("loop(a, b) -> out()")
def flaky(a: float, b: float) -> float:
    """Fails on a quarter of the grid until it is repaired."""
    if STATE["broken"] and a > 0.75:
        raise RuntimeError("engine unavailable")
    return a * b


def main() -> None:
    """Time a run with failures, then the resume pass that repairs it."""
    space = xr.Dataset(
        {
            "a": ("a", np.linspace(0.0, 1.0, SIDE)),
            "b": ("b", np.linspace(0.0, 1.0, SIDE)),
        }
    )
    n_points = SIDE * SIDE
    run_dir = Path(tempfile.mkdtemp())
    policy = SweepPolicy(store=run_dir / "flaky.zarr")

    try:
        start = time.perf_counter()
        first = flaky(space, policy=policy)
        record(
            "resume_first_pass",
            n_points=n_points,
            n_calls=n_points,
            wall_time_s=time.perf_counter() - start,
        )
        n_failed = int(np.isnan(first.out.values).sum())

        STATE["broken"] = False
        start = time.perf_counter()
        second = flaky(space, policy=policy)
        record(
            "resume_second_pass",
            n_points=n_points,
            n_calls=n_failed,
            wall_time_s=time.perf_counter() - start,
        )

        assert not np.isnan(second.out.values).any()
    finally:
        shutil.rmtree(run_dir)


if __name__ == "__main__":
    main()
