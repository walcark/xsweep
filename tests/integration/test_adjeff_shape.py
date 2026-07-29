"""SC-007: an adjeff sampler expressed with its signature unchanged.

The motivating consumer is adjeff's `SceneModuleSweep`, whose point
functions already split their arguments into swept parameters and static
configuration. If xsweep needed those signatures rewritten, the migration
would not be worth doing.

The engine is stubbed: what is under test is the shape of the API, not the
physics.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from xsweep import SweepModule, SweepPolicy


def sph_alb(
    wl: xr.DataArray,
    aot: float,
    rh: float,
    h: float,
    href: float,
    species: str,
    afgl_type: str,
    remove_rayleigh: bool,
    n_ph: int,
) -> xr.DataArray:
    """Stand in for adjeff's Smart-G spherical-albedo sampler.

    The signature is copied verbatim from
    ``adjeff/modules/samplers/_smartg.py``: swept parameters first, static
    configuration after. Nothing about it changes to run under xsweep.
    """
    scale = 1.0 if remove_rayleigh else 2.0
    return scale * aot * (1.0 + rh / 100.0) * (h - href) * wl / n_ph


class SphAlbSampler(SweepModule):
    """The adjeff module, re-expressed on top of xsweep.

    In adjeff this needed three ClassVars (`scalar_dims`, `vector_dims`,
    `output_vars`) plus `_get_configs`, `_compute` and `_apply_bundle`. Here
    the contract carries all of it, and `forward` is the engine call.
    """

    contract = "loop(aot, rh, h, href) vec(wl @ 8) -> sph_alb(wl)"

    def forward(
        self,
        aot: float,
        rh: float,
        h: float,
        href: float,
        wl: xr.DataArray,
        *,
        species: str,
        afgl_type: str,
        remove_rayleigh: bool,
        n_ph: int,
    ) -> xr.DataArray:
        """Call the engine with its own signature, untouched."""
        return sph_alb(
            wl=wl,
            aot=aot,
            rh=rh,
            h=h,
            href=href,
            species=species,
            afgl_type=afgl_type,
            remove_rayleigh=remove_rayleigh,
            n_ph=n_ph,
        )


def _space() -> xr.Dataset:
    """Return the atmospheric space an AtmoConfig would produce."""
    return xr.Dataset(
        {
            "aot": ("aot", [0.05, 0.2]),
            "rh": ("rh", [30.0, 70.0]),
            "h": ("h", [8.0]),
            "href": ("href", [0.0]),
            "wl": ("wl", np.arange(400.0, 500.0, 5.0)),
        }
    )


def _statics() -> dict[str, object]:
    """Return what adjeff passes through `__init__` and forwards to the call."""
    return {
        "species": "continental",
        "afgl_type": "afgl_exp_h8km",
        "remove_rayleigh": False,
        "n_ph": int(2e7),
    }


def test_the_sampler_runs_with_its_signature_unchanged() -> None:
    """The whole point of SC-007: no rewrite required at the engine boundary."""
    module = SphAlbSampler()
    result = module(_space(), **_statics())

    assert dict(result.sizes) == {"aot": 2, "rh": 2, "h": 1, "href": 1, "wl": 20}
    assert (result.status.values == 1).all()

    expected = 2.0 * 0.05 * 1.3 * 8.0 * 400.0 / int(2e7)
    got = result.sph_alb.sel(aot=0.05, rh=30.0).values.ravel()[0]
    assert got == expected


def test_statics_that_change_physics_change_the_store_identity(tmp_path) -> None:
    """`remove_rayleigh` and `n_ph` change results, so they change the cache."""
    module = SphAlbSampler(SweepPolicy(store=str(tmp_path / "s.zarr")))
    statics = _statics()
    module(_space(), **statics)

    from xsweep.errors import StoreError

    try:
        module(_space(), **{**statics, "remove_rayleigh": True})
    except StoreError as exc:
        assert "different configuration" in str(exc)
    else:  # pragma: no cover - the assertion above is the expected path
        raise AssertionError("a physics-changing static must invalidate the store")


def test_the_class_replaces_the_adjeff_plumbing() -> None:
    """One contract stands in for three ClassVars and three methods."""
    contract = SphAlbSampler._contract
    assert [v.name for v in contract.loop] == ["aot", "rh", "h", "href"]
    assert [v.name for v in contract.vec] == ["wl"]
    assert contract.outputs == ("sph_alb",)


def test_dedup_applies_to_the_sampler_unchanged() -> None:
    """The adjeff use case: a map of atmospheric parameters with duplicates."""
    module = SphAlbSampler(SweepPolicy(dedup=True))
    rng = np.random.default_rng(0)
    space = xr.Dataset(
        {
            "aot": (("y", "x"), rng.choice([0.05, 0.2], size=(4, 4))),
            "rh": (("y", "x"), rng.choice([30.0, 70.0], size=(4, 4))),
            "h": ((), 8.0),
            "href": ((), 0.0),
            "wl": ("wl", np.arange(400.0, 420.0, 5.0)),
        }
    )
    plain = SphAlbSampler()(space, **_statics())
    deduped = module(space, **_statics())
    assert np.array_equal(plain.sph_alb.values, deduped.sph_alb.values)
