"""Package a sweep as a class, and see the three temporalities separated.

Run with::

    pixi run -e dev python scripts/05_module_and_policy.py

The contract belongs to the class (physics), the policy to the instance
(run), the space and statics to the call (data). Watching one change without
disturbing the others is the point of the split.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from xsweep import ContractError, SweepModule, SweepPolicy

CALLS = {"n": 0}


class RhoAtm(SweepModule):
    """Atmospheric reflectance, as adjeff's samplers are shaped."""

    contract = "loop(aot, rh) vec(wl @ 16) -> rho_atm(wl)"

    def forward(
        self, aot: float, rh: float, wl: xr.DataArray, *, n_ph: int
    ) -> xr.DataArray:
        """Pure physics: no store, no policy, no bookkeeping."""
        CALLS["n"] += 1
        return np.exp(-aot * (1.0 + rh / 100.0)) * (wl / 500.0) / np.sqrt(n_ph)


def main() -> None:
    """Show contract, policy and data varying independently."""
    space = xr.Dataset(
        {
            "aot": (("y", "x"), np.array([[0.05, 0.2], [0.05, 0.4]])),
            "rh": (("y", "x"), np.array([[30.0, 80.0], [30.0, 50.0]])),
            "wl": ("wl", np.arange(400.0, 500.0, 5.0)),
        }
    )
    store = Path(tempfile.mkdtemp())

    try:
        print("=== forward is testable on its own, no sweep involved ===")
        alone = RhoAtm().forward(0.1, 50.0, xr.DataArray([500.0], dims="wl"), n_ph=100)
        print(f"forward(0.1, 50.0, [500.0], n_ph=100) = {float(alone[0]):.5f}\n")

        print("=== Same class, two policies: only the cost changes ===")
        plain = RhoAtm()
        tuned = RhoAtm(SweepPolicy(store=str(store / "rho.zarr"), dedup=True))

        CALLS["n"] = 0
        a = plain(space, n_ph=1000)
        calls_plain = CALLS["n"]

        CALLS["n"] = 0
        b = tuned(space, n_ph=1000)
        print(f"default policy : {calls_plain} calls")
        print(f"dedup + store  : {CALLS['n']} calls")
        print(f"identical result: {np.array_equal(a.rho_atm.values, b.rho_atm.values)}")

        print("\n=== A call-level override beats the instance ===")
        plan = tuned.explain(space, policy=SweepPolicy(dedup=False), n_ph=1000)
        print(f"instance says dedup, the call says no: n_unique={plan.n_unique}")

        print("\n=== A static that changes physics changes the identity ===")
        try:
            tuned(space, n_ph=2000)
        except Exception as exc:  # noqa: BLE001 - the message is the lesson
            print(f"{type(exc).__name__}: {str(exc)[:120]}...")

        print("\n=== A missing contract fails when the class is defined ===")
        try:

            class Broken(SweepModule):
                def forward(self, a: float) -> float:
                    return a

        except ContractError as exc:
            print(f"ContractError: {exc}")
    finally:
        shutil.rmtree(store, ignore_errors=True)


if __name__ == "__main__":
    main()
