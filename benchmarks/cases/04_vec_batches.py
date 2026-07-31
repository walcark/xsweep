"""Batched vec axis: the work-item path when one point becomes many calls.

50 loop points against a 20000-element vec axis cut into batches of 200,
which is 5000 work items for 50 points. The callee is elementwise and cheap,
so what is timed is the batching itself: slicing the axis, dispatching a work
item per slice, and stitching the pieces back into one array per point.
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
from xsweep import SweepPolicy, sweep  # noqa: E402

N_LOOP = 50
N_WL = 20_000
BATCH = 200


@sweep("loop(a) vec(wl) -> out(wl)")
def spectral(a: float, wl: xr.DataArray) -> xr.DataArray:
    """Elementwise along the whole axis, so a batch is as valid as the axis."""
    return wl * a


def main() -> None:
    """Time the same sweep whole, then batched."""
    space = xr.Dataset(
        {
            "a": ("a", np.linspace(0.1, 1.0, N_LOOP)),
            "wl": ("wl", np.linspace(0.4, 2.4, N_WL)),
        }
    )

    start = time.perf_counter()
    whole = spectral(space)
    record(
        "vec_whole_axis",
        n_points=N_LOOP,
        n_calls=N_LOOP,
        wall_time_s=time.perf_counter() - start,
    )

    start = time.perf_counter()
    batched = spectral(space, policy=SweepPolicy(chunks={"wl": BATCH}))
    record(
        "vec_batched",
        n_points=N_LOOP,
        n_calls=N_LOOP * (N_WL // BATCH),
        wall_time_s=time.perf_counter() - start,
    )

    np.testing.assert_array_equal(whole.out.values, batched.out.values)


if __name__ == "__main__":
    main()
