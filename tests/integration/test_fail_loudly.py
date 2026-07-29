"""Quickstart scenario 11: every refusal arrives before the first call.

Constitution principle VII. The wrapped calls are expensive, so an error
caught after the sweep started has already cost real hours. Each case below
asserts both the refusal AND that the function was never invoked.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xsweep import SweepPolicy, XsweepError, sweep
from xsweep.errors import ContractError, PolicyError, SpaceError, StoreError


@pytest.fixture
def counter() -> list[float]:
    """Return the list every callee below appends to."""
    return []


def _sweeper(counter: list[float], spec: str = "loop(a, b) -> out()"):
    """Build a counting sweeper on the given contract."""

    @sweep(spec, version="1")
    def f(**kwargs: float) -> float:
        counter.append(1.0)
        return 1.0

    return f


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        ("batch(wl) -> t(wl)", "unknown clause"),
        ("loop(a)", "no '->' clause"),
        ("vec(a) const(a) -> out()", "appears in both"),
        ("loop(a) -> out(), out()", "declared twice"),
        ("loop(a) vec(wl @ 8) -> band_int(band)", "not among the declared"),
        ("loop(a) -> band(band)", "its own dims"),
    ],
)
def test_contract_errors_arrive_at_decoration(spec: str, fragment: str) -> None:
    """Definition time, so a bad contract cannot survive to a run."""
    with pytest.raises(ContractError, match=fragment):

        @sweep(spec)
        def f(**kwargs: object) -> float:
            return 1.0


def test_a_missing_variable_is_refused_before_calling(counter: list[float]) -> None:
    """The space must serve the contract, and the message lists what it holds."""
    f = _sweeper(counter)
    space = xr.Dataset({"a": ("a", [1.0])})
    with pytest.raises(SpaceError, match="available variables: a"):
        f(space)
    assert counter == []


def test_an_object_coordinate_is_refused_before_calling(
    counter: list[float],
) -> None:
    """Objects break fingerprint, store and dedup alike."""
    f = _sweeper(counter, "loop(cfg) -> out()")
    space = xr.Dataset({"cfg": ("cfg", np.array([object()], dtype=object))})
    with pytest.raises(SpaceError, match="sweep a label string"):
        f(space)
    assert counter == []


@pytest.mark.parametrize(
    "policy",
    [
        SweepPolicy(dedup=("z",)),
        SweepPolicy(chunks={"z": 2}),
        SweepPolicy(retries=-1),
        SweepPolicy(max_workers=0),
    ],
)
def test_policy_errors_arrive_before_calling(
    policy: SweepPolicy, counter: list[float]
) -> None:
    """Policy is validated against the space during planning, so it costs zero."""
    f = _sweeper(counter)
    space = xr.Dataset({"a": ("a", [1.0]), "b": ("b", [2.0])})
    with pytest.raises(PolicyError):
        f(space, policy=policy)
    assert counter == []


def test_a_reserved_static_name_is_refused_before_calling(
    counter: list[float],
) -> None:
    """Statics share a namespace with the sweeper's own arguments.

    Only ``space`` is actually reachable: ``policy`` is keyword-only in the
    signature, so a static of that name binds to the parameter and never
    reaches the statics. It stays reserved for symmetry and clarity.
    """
    f = _sweeper(counter)
    space = xr.Dataset({"a": ("a", [1.0]), "b": ("b", [2.0])})
    with pytest.raises(PolicyError, match="reserved by the sweeper"):
        f(space, space=1)
    assert counter == []


def test_a_non_policy_passed_as_policy_says_what_to_do(
    counter: list[float],
) -> None:
    """Otherwise the failure surfaces as an opaque TypeError from vars()."""
    f = _sweeper(counter)
    space = xr.Dataset({"a": ("a", [1.0]), "b": ("b", [2.0])})
    with pytest.raises(PolicyError, match="must be a SweepPolicy"):
        f(space, policy=1)  # type: ignore[arg-type]
    assert counter == []


def test_an_unserialisable_static_is_refused_before_calling(
    tmp_path, counter: list[float]
) -> None:
    """A cache that cannot see a change is worse than no cache."""
    f = _sweeper(counter)
    space = xr.Dataset({"a": ("a", [1.0]), "b": ("b", [2.0])})

    class Opaque:
        pass

    with pytest.raises(StoreError, match="__cache_token__"):
        f(space, policy=SweepPolicy(store=str(tmp_path / "s.zarr")), engine=Opaque())


def test_a_changed_space_is_refused_on_resume(tmp_path) -> None:
    """Resuming against a different axis would misalign every result."""
    calls: list[float] = []

    @sweep("loop(a) -> out()", version="1")
    def f(a: float) -> float:
        calls.append(a)
        return a

    store = str(tmp_path / "s.zarr")
    f(xr.Dataset({"a": ("a", [1.0, 2.0])}), policy=SweepPolicy(store=store))
    calls.clear()

    with pytest.raises(StoreError, match="different 'a' axis"):
        f(xr.Dataset({"a": ("a", [1.0, 2.0, 3.0])}), policy=SweepPolicy(store=store))
    assert calls == []


def test_every_refusal_shares_one_base_class() -> None:
    """Users catch one thing; the subclass says which phase to fix."""
    for error in (ContractError, SpaceError, PolicyError, StoreError):
        assert issubclass(error, XsweepError)
