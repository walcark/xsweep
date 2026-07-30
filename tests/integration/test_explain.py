"""Quickstart scenario 6: the plan, and the honesty rules it must obey.

A plan exists so a user can decide whether to spend engine hours. A plan
that spent one to build itself, or that invented a duration, would defeat
its own purpose.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from xsweep import SweepPolicy, sweep


def test_explaining_costs_nothing(map_space: xr.Dataset) -> None:
    """The counter is the proof: not one call, whatever the policy."""
    calls: list[float] = []

    @sweep("loop(aot, rh) -> rho()")
    def f(aot: float, rh: float) -> float:
        calls.append(aot)
        return aot

    for policy in (
        SweepPolicy(),
        SweepPolicy(dedup=True),
        SweepPolicy(executor="process"),
    ):
        f.explain(map_space, policy=policy)
    assert calls == []


def test_the_probe_result_is_kept_not_discarded(tmp_path) -> None:
    """Discarding it would waste an engine call, which is the whole argument."""
    calls: list[float] = []

    @sweep("loop(a) -> band_int(band)")
    def f(a: float) -> xr.DataArray:
        calls.append(a)
        return xr.DataArray([a, a, a], dims="band")

    space = xr.Dataset({"a": ("a", [1.0])})
    plan = f.explain(space)
    assert not plan.determined

    result = f(space, policy=SweepPolicy(store=str(tmp_path / "s.zarr")))
    assert len(calls) == 1
    assert dict(result.sizes) == {"a": 1, "band": 3}


def test_no_duration_is_invented(map_space: xr.Dataset) -> None:
    """The library cannot know what a call costs, so it must not pretend."""

    @sweep("loop(aot, rh) -> rho()")
    def f(aot: float, rh: float) -> float:
        return aot

    text = repr(f.explain(map_space))
    assert "elapsed" not in text
    assert "estimated" not in text


def test_the_report_shows_counts_batching_and_shapes() -> None:
    """Everything FR-030 requires must actually appear."""

    @sweep("loop(aot) vec(wl) const(srf) -> band_int(band)", version="3")
    def f(aot: float, wl: xr.DataArray, srf: xr.DataArray) -> xr.DataArray:
        return ((aot * wl) * srf).sum("wl")

    space = xr.Dataset(
        {
            "aot": ("aot", [0.1, 0.2]),
            "wl": ("wl", np.arange(20.0)),
            "srf": (("band", "wl"), np.ones((3, 20))),
        }
    )
    text = repr(f.explain(space, n_ph=1000))

    assert "v3" in text
    assert "points          2" in text
    assert "aot   loop" in text
    assert "wl    vec" in text
    assert "srf   const" in text
    assert "n_ph  static" in text
    assert "band_int" in text
    assert "EXECUTOR  serial" in text


def test_a_forgotten_dedup_is_visible_before_launching(map_space: xr.Dataset) -> None:
    """The single most expensive mistake, made obvious at zero cost."""

    @sweep("loop(aot, rh) -> rho()")
    def f(aot: float, rh: float) -> float:
        return aot

    without = f.explain(map_space)
    with_dedup = f.explain(map_space, policy=SweepPolicy(dedup=True))
    assert with_dedup.n_calls < without.n_calls
    assert "dedup  disabled" in repr(without)
    assert "unique" in repr(with_dedup)


def test_an_accidental_product_is_visible_as_extra_axes() -> None:
    """Meaning zip but writing a product shows up as one axis per variable."""

    @sweep("loop(aot, rh) -> rho()")
    def f(aot: float, rh: float) -> float:
        return aot

    zipped = xr.Dataset(
        {
            "aot": (("y", "x"), np.zeros((3, 3))),
            "rh": (("y", "x"), np.zeros((3, 3))),
        }
    )
    product = xr.Dataset({"aot": ("aot", np.zeros(9)), "rh": ("rh", np.zeros(9))})

    assert f.explain(zipped).n_points == 9
    assert f.explain(product).n_points == 81
    assert len(f.explain(product).axes) == 2


def test_a_batch_size_of_one_shows_its_call_count() -> None:
    """The third costly mistake: the axis length multiplies the calls."""

    @sweep("loop(a) vec(wl @ 1) -> t(wl)")
    def f(a: float, wl: xr.DataArray) -> xr.DataArray:
        return wl

    space = xr.Dataset({"a": ("a", [1.0, 2.0]), "wl": ("wl", np.arange(50.0))})
    plan = f.explain(space)
    assert plan.n_points == 2
    assert plan.n_calls == 100
