"""
Resuming a sweep after ~10% of it fails
==========================================

The scenario that justifies the store, applied at a size where the payoff
has a real number attached: 900 points, an engine that fails on 3 of 30 AOT
columns (90 points, 10%) until it is "fixed", and a resume pass that only
recomputes what failed. See ``scripts/03_failure_and_resume.py`` for the
smaller, more narrated version of the same idiom.

Run with::

    pixi run -e dev python benchmarks/examples/10_resume_after_interruption.py
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

# sphinx-gallery executes examples without `__file__` set, so the repo root
# is located from the already-installed xsweep package instead.
sys.path.insert(0, str(Path(xsweep.__file__).resolve().parents[2]))

from benchmarks._lib.timing import record  # noqa: E402
from xsweep import SweepPolicy, sweep  # noqa: E402

STATE = {"calls": 0, "flaky": True, "bad_aot": frozenset()}


@sweep("loop(aot, sza) -> transmission()")
def transmission(aot: float, sza: float) -> float:
    """Beer-Lambert transmission (case 1), failing on 10% of AOT columns."""
    STATE["calls"] += 1
    if STATE["flaky"] and aot in STATE["bad_aot"]:
        raise RuntimeError(f"engine blew up on aot={aot}")
    time.sleep(0.002)
    airmass = 1.0 / np.cos(np.radians(sza))
    return float(np.exp(-aot * airmass))


def _report(result: xr.Dataset, label: str, calls: int) -> None:
    """Print the outcome counts held in the status variable."""
    status = result.status.values
    print(
        f"{label}: {calls:3d} calls | "
        f"ok={int((status == 1).sum())} "
        f"failed={int((status == 2).sum())} "
        f"pending={int((status == 0).sum())}"
    )


def main() -> None:
    """Run a 30x30 sweep with 10% failures, then resume after fixing it."""
    aot = np.linspace(0.02, 0.6, 30)
    sza = np.linspace(0.0, 80.0, 30)
    STATE["bad_aot"] = frozenset(aot[::10])  # 3 of 30 columns, 90/900 points

    space = xr.Dataset({"aot": ("aot", aot), "sza": ("sza", sza)})
    store = Path(tempfile.mkdtemp()) / "transmission.zarr"
    policy = SweepPolicy(store=str(store))

    try:
        print("=== First run, the engine fails on 10% of the space ===\n")
        print(transmission.explain(space, policy=policy))

        STATE["calls"] = 0
        start = time.perf_counter()
        first = transmission(space, policy=policy)
        full_elapsed = time.perf_counter() - start
        full_calls = STATE["calls"]
        _report(first, "run 1", full_calls)

        print("\n=== Engine fixed, same store, resume ===\n")
        STATE["flaky"] = False
        STATE["calls"] = 0
        start = time.perf_counter()
        second = transmission(space, policy=policy)
        resume_elapsed = time.perf_counter() - start
        resume_calls = STATE["calls"]
        _report(second, "run 2", resume_calls)

        print(
            f"\nfull run: {full_elapsed:.3f}s for {full_calls} calls\n"
            f"resume:   {resume_elapsed:.3f}s for {resume_calls} calls "
            f"({full_calls / max(resume_calls, 1):.0f}x fewer)"
        )

        record(
            "10_resume_after_interruption_full",
            n_calls=full_calls,
            wall_time_s=full_elapsed,
        )
        bench = record(
            "10_resume_after_interruption_resume",
            n_calls=resume_calls,
            wall_time_s=resume_elapsed,
        )
        print(f"\nrecorded: {bench.wall_time_s:.4f}s for {bench.n_calls} calls")
    finally:
        shutil.rmtree(store.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
