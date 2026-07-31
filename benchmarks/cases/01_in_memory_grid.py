"""In-memory Cartesian sweep: the floor of xsweep's per-point cost.

A near-free callee over 40000 points, with no store and no deduplication.
Whatever this costs is planning, delivery and result assembly, so it is the
number to watch when any of those change.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

import xsweep

sys.path.insert(0, str(Path(xsweep.__file__).resolve().parents[2]))

from benchmarks._lib.timing import record  # noqa: E402
from xsweep import sweep  # noqa: E402

SIDE = 200


@sweep("loop(a, b) -> out()")
def cheap(a: float, b: float) -> float:
    """As close to free as a Python call gets."""
    return a * b


def main() -> None:
    """Time a 200 x 200 in-memory sweep."""
    space = xr.Dataset(
        {
            "a": ("a", np.linspace(0.0, 1.0, SIDE)),
            "b": ("b", np.linspace(0.0, 1.0, SIDE)),
        }
    )

    start = time.perf_counter()
    result = cheap(space)
    elapsed = time.perf_counter() - start

    assert result.out.shape == (SIDE, SIDE)
    record(
        "in_memory_grid",
        n_points=SIDE * SIDE,
        n_calls=SIDE * SIDE,
        wall_time_s=elapsed,
    )


if __name__ == "__main__":
    main()
