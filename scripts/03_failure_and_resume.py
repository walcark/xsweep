"""Survive a failing engine, then resume without recomputing anything.

Run with::

    pixi run -e dev python scripts/03_failure_and_resume.py

The scenario is the one that justifies the store: a long sweep where some
calls fail and the process dies halfway. Nothing computed is ever lost, and
the status variable always says what happened.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from xsweep import SweepPolicy, sweep

STATE = {"calls": 0, "flaky": True}


@sweep("loop(aot, sza) -> rho()", version="1")
def rho(aot: float, sza: float) -> float:
    """Fail on one slice of the space while the engine is misbehaving."""
    STATE["calls"] += 1
    if STATE["flaky"] and aot == 0.2:
        raise RuntimeError(f"engine blew up on aot={aot}")
    return float(aot * 100.0 + sza)


def report(result: xr.Dataset, label: str) -> None:
    """Print the outcome counts held in the status variable."""
    status = result.status.values
    print(
        f"{label}: {STATE['calls']:3d} calls | "
        f"ok={int((status == 1).sum())} "
        f"failed={int((status == 2).sum())} "
        f"pending={int((status == 0).sum())}"
    )


def main() -> None:
    """Run a failing sweep, then fix the engine and resume."""
    space = xr.Dataset(
        {
            "aot": ("aot", [0.05, 0.1, 0.2, 0.3]),
            "sza": ("sza", [0.0, 20.0, 40.0, 60.0]),
        }
    )
    store = Path(tempfile.mkdtemp()) / "rho.zarr"
    policy = SweepPolicy(store=str(store))

    try:
        print("=== First run, the engine fails on aot=0.2 ===")
        STATE["calls"] = 0
        first = rho(space, policy=policy)
        report(first, "run 1")
        failed = first.status.values == 2
        print(
            f"failed points hold NaN: {bool(np.isnan(first.rho.values[failed]).all())}"
        )
        print(
            f"others are finite:      {bool(np.isfinite(first.rho.values[~failed]).all())}"
        )

        print("\n=== Engine fixed, same store, relaunch ===")
        STATE["flaky"] = False
        STATE["calls"] = 0
        second = rho(space, policy=policy)
        report(second, "run 2")
        print("only the failed points were recomputed")

        print("\n=== Third run, nothing left to do ===")
        STATE["calls"] = 0
        third = rho(space, policy=policy)
        report(third, "run 3")

        print("\n=== Fail fast instead, when you want to stop and look ===")
        STATE["flaky"] = True
        fresh = Path(tempfile.mkdtemp()) / "rho2.zarr"
        try:
            rho(space, policy=SweepPolicy(store=str(fresh), on_error="raise"))
        except Exception as exc:  # noqa: BLE001 - showing the message is the point
            print(f"{type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(fresh.parent, ignore_errors=True)
    finally:
        shutil.rmtree(store.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
