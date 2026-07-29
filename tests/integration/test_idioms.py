"""Quickstart scenario 9: primitives, and the two documented idioms.

These are the cases the design deliberately does NOT solve with machinery,
solving them with a convention instead. A test keeps the convention honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xsweep import SweepPolicy, sweep
from xsweep.errors import SpaceError


def test_string_axes_are_first_class() -> None:
    """adjeff coerced every axis to float, which broke profile names."""
    seen: list[str] = []

    @sweep("loop(profile) -> out()")
    def f(profile: str) -> float:
        seen.append(profile)
        return float(len(profile))

    space = xr.Dataset({"profile": ("profile", ["afgl_ms", "afgl_t"])})
    result = f(space)
    assert seen == ["afgl_ms", "afgl_t"]
    assert result.out.values.tolist() == [7.0, 6.0]


def test_datetime_axes_are_first_class() -> None:
    """A time axis is an ordinary swept parameter, not a special case."""

    @sweep("loop(when) -> out()")
    def f(when: int) -> float:
        return float(when > 0)

    stamps = np.array(["2026-01-01", "2026-07-01"], dtype="datetime64[ns]")
    result = f(xr.Dataset({"when": ("when", stamps)}))
    assert dict(result.sizes) == {"when": 2}


def test_object_values_point_at_the_label_idiom() -> None:
    """The refusal must teach the workaround, not just say no."""

    @sweep("loop(cfg) -> out()")
    def f(cfg: object) -> float:
        return 1.0

    space = xr.Dataset({"cfg": ("cfg", np.array([object()], dtype=object))})
    with pytest.raises(SpaceError, match="label string"):
        f(space)


def test_the_seed_carrier_idiom_makes_repetitions_distinct() -> None:
    """Without a carrier variable, identical calls would collapse to one.

    That is the trap: the standard deviation across repetitions would be
    exactly zero, and the Monte-Carlo convergence study would silently
    measure nothing.
    """

    @sweep("loop(aot, seed) -> rho()")
    def mc(aot: float, seed: int) -> float:
        rng = np.random.default_rng(int(seed))
        return float(aot + rng.normal(0.0, 0.1))

    space = xr.Dataset(
        {
            "aot": ("aot", [0.1, 0.3]),
            "seed": ("rep", np.arange(20)),
        }
    )
    result = mc(space)
    assert dict(result.sizes) == {"aot": 2, "rep": 20}

    noise = result.rho.std("rep")
    assert (noise > 0).all()

    # Same seeds, same values: each repetition stays reproducible.
    again = mc(space)
    assert np.array_equal(result.rho.values, again.rho.values)


def test_the_seed_idiom_survives_dedup() -> None:
    """Distinct seeds mean distinct rows, so nothing collapses."""

    @sweep("loop(aot, seed) -> rho()")
    def mc(aot: float, seed: int) -> float:
        return float(aot * 100 + seed)

    space = xr.Dataset({"aot": ("aot", [0.1, 0.1]), "seed": ("rep", np.arange(5))})
    plain = mc(space)
    deduped = mc(space, policy=SweepPolicy(dedup=True))
    assert np.array_equal(plain.rho.values, deduped.rho.values)


def test_comparing_two_versions_is_a_store_level_workflow(tmp_path) -> None:
    """Version selects code, not data, so it is never a sweep axis."""

    @sweep("loop(a) -> rho()", version="1")
    def v1(a: float) -> float:
        return a * 2.0

    @sweep("loop(a) -> rho()", version="2")
    def v2(a: float) -> float:
        return a * 3.0

    space = xr.Dataset({"a": ("a", [1.0, 2.0])})
    first = v1(space, policy=SweepPolicy(store=str(tmp_path / "v1.zarr")))
    second = v2(space, policy=SweepPolicy(store=str(tmp_path / "v2.zarr")))

    compared = xr.concat(
        [first.rho, second.rho], dim=pd.Index(["1", "2"], name="version")
    )
    delta = compared.diff("version")
    assert delta.values.tolist() == [[1.0, 2.0]]


def test_a_static_mapping_carries_its_own_cache_token(tmp_path) -> None:
    """The other half of the label idiom: the mapping must be visible."""
    calls: list[str] = []

    class Profiles:
        """A static lookup that cannot be JSON-serialised on its own."""

        def __init__(self, tag: str, table: dict[str, float]) -> None:
            self.tag = tag
            self.table = table

        def __cache_token__(self) -> str:
            return self.tag

    @sweep("loop(profile) -> rho()", version="1")
    def f(profile: str, *, profiles: Profiles) -> float:
        calls.append(profile)
        return profiles.table[profile]

    space = xr.Dataset({"profile": ("profile", ["ms", "t"])})
    store = str(tmp_path / "s.zarr")
    table = Profiles("v1", {"ms": 1.0, "t": 2.0})

    first = f(space, policy=SweepPolicy(store=store), profiles=table)
    assert first.rho.values.tolist() == [1.0, 2.0]

    calls.clear()
    f(space, policy=SweepPolicy(store=store), profiles=table)
    assert calls == []
