"""The public surface, checked against contracts/public-api.md.

Principle II caps this list, so a test that counts it is a guard against
accidental growth.
"""

from __future__ import annotations

import pytest
import xarray as xr

import xsweep
from xsweep import Contract, Plan, Sweeper, SweepPolicy, sweep
from xsweep.errors import ContractError, PolicyError


def test_public_names_are_exactly_those_documented() -> None:
    """Growing this list is a constitution decision, not an accident."""
    assert set(xsweep.__all__) == {
        "Contract",
        "ContractError",
        "LoopVar",
        "OutVar",
        "Plan",
        "PointFailed",
        "PolicyError",
        "SpaceError",
        "StoreError",
        "StoreLockedError",
        "SweepModule",
        "SweepPolicy",
        "Sweeper",
        "VecVar",
        "XsweepError",
        "sweep",
    }


def test_every_public_name_is_importable() -> None:
    """A name in __all__ that cannot be imported is a broken promise."""
    for name in xsweep.__all__:
        assert getattr(xsweep, name) is not None


def test_plan_is_annotatable_by_users() -> None:
    """explain() returns a Plan, so users must be able to name that type."""

    def review(plan: Plan) -> int:
        return plan.n_calls

    @sweep("loop(a) -> out()")
    def f(a: float) -> float:
        return a

    space = xr.Dataset({"a": ("a", [1.0, 2.0])})
    assert review(f.explain(space)) == 2


def test_malformed_contract_raises_at_decoration() -> None:
    """Contract errors must explode at import, not after minutes of runs."""
    with pytest.raises(ContractError):

        @sweep("loop(a)")
        def f(a: float) -> float:
            return a


def test_decorator_produces_a_sweeper_exposing_the_contract() -> None:
    """The contract is introspectable, which the future DAG layer needs."""

    @sweep("loop(a) vec(wl) -> t(wl)")
    def f(a: float, wl: xr.DataArray) -> xr.DataArray:
        return wl

    assert isinstance(f, Sweeper)
    assert isinstance(f.contract, Contract)
    assert f.contract.inputs == ("a", "wl")
    assert f.contract.outputs == ("t",)


def test_reserved_static_names_are_refused() -> None:
    """Statics share a namespace with the sweeper's own arguments."""

    @sweep("loop(a) -> out()")
    def f(a: float, **kw: object) -> float:
        return a

    space = xr.Dataset({"a": ("a", [1.0])})
    with pytest.raises(PolicyError, match="reserved by the sweeper"):
        f(space, policy=SweepPolicy(), space=1)  # type: ignore[call-arg]
