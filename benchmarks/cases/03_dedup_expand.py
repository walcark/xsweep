"""Deduplicated pixel map: unique-row extraction and expansion at scale.

90000 pixels carrying 60 distinct rows. With a near-free callee the engine
time is negligible by construction, so what is measured is the deduplication
machinery itself: hashing the rows, computing 60 results, and filling the
89940 duplicated positions, in memory and then into a store.
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

SIDE = 300
N_CLASSES = 60


@sweep("loop(a, b) -> out()")
def cheap(a: float, b: float) -> float:
    """As close to free as a Python call gets."""
    return a * b


def main() -> None:
    """Time a deduplicated map in memory, then persisted."""
    rows = np.arange(SIDE)[:, None] // 10
    cols = np.arange(SIDE)[None, :] // 10
    class_id = (rows + cols) % N_CLASSES
    a_map = np.linspace(0.1, 1.0, N_CLASSES)[class_id]
    b_map = np.linspace(1.0, 2.0, N_CLASSES)[class_id]

    space = xr.Dataset({"a": (("y", "x"), a_map), "b": (("y", "x"), b_map)})
    n_points = SIDE * SIDE

    start = time.perf_counter()
    cheap(space, policy=SweepPolicy(dedup=True))
    record(
        "dedup_in_memory",
        n_points=n_points,
        n_calls=N_CLASSES,
        wall_time_s=time.perf_counter() - start,
    )

    run_dir = Path(tempfile.mkdtemp())
    try:
        start = time.perf_counter()
        cheap(space, policy=SweepPolicy(dedup=True, store=run_dir / "map.zarr"))
        record(
            "dedup_to_store",
            n_points=n_points,
            n_calls=N_CLASSES,
            wall_time_s=time.perf_counter() - start,
        )
    finally:
        shutil.rmtree(run_dir)


if __name__ == "__main__":
    main()
