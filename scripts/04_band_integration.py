"""Hand a whole axis to a vectorised engine, with a response table as context.

Run with::

    pixi run -e dev python scripts/04_band_integration.py

Band integration is the case that motivates both the `vec` and the `const`
clauses, and it is also the case where batching would silently corrupt the
result. The script shows the correct contract and the refusal you get for the
wrong one.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from xsweep import ContractError, SweepPolicy, sweep

CALLS = {"n": 0}


@sweep("loop(aot, rh) vec(wl) const(srf) -> radiance(band)", version="1")
def radiance(
    aot: float, rh: float, wl: xr.DataArray, srf: xr.DataArray
) -> xr.DataArray:
    """Integrate a spectrum against a response function.

    ``wl`` arrives whole because the integration reduces over it, and ``srf``
    arrives whole because it is context rather than a swept parameter.
    """
    CALLS["n"] += 1
    spectrum = np.exp(-aot * (1.0 + rh / 100.0)) * (wl / 500.0)
    return (spectrum * srf).sum("wl")


def main() -> None:
    """Run a band integration, then show what the contract refuses."""
    wl = np.arange(400.0, 900.0, 5.0)
    centres, width = np.array([490.0, 560.0, 665.0]), 40.0
    weights = np.exp(-(((wl[None, :] - centres[:, None]) / width) ** 2))
    weights /= weights.sum(axis=1, keepdims=True)

    space = xr.Dataset(
        {
            "aot": ("aot", [0.05, 0.2, 0.4]),
            "rh": ("rh", [30.0, 80.0]),
            "wl": ("wl", wl),
            "srf": (("band", "wl"), weights),
        }
    )

    print("=== The plan shows what each call receives ===\n")
    print(radiance.explain(space))

    print("\n=== Running ===")
    result = radiance(space, policy=SweepPolicy(load=True))
    print(f"{CALLS['n']} calls, result {dict(result.sizes)}")
    print("\nradiance by band, at aot=0.05, rh=30:")
    print(np.round(result.radiance.sel(aot=0.05, rh=30.0).values, 4))

    print("\n=== What the contract refuses, and why ===\n")
    try:

        @sweep("loop(aot) vec(wl @ 8) const(srf) -> radiance(band)")
        def wrong(aot: float, wl: xr.DataArray, srf: xr.DataArray) -> xr.DataArray:
            return (wl * srf).sum("wl")

    except ContractError as exc:
        print(f"ContractError: {exc}")
        print(
            "\nBatching wl would integrate each batch separately and sum "
            "nothing\ncorrectly. The refusal happens at decoration, before "
            "any engine time."
        )


if __name__ == "__main__":
    main()
