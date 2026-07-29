"""Quickstart scenario 5: the class facade.

The point of validating at ``__init_subclass__`` is timing: a bad contract
must explode when the module is imported, not after twenty minutes of engine
time.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xsweep import SweepModule, SweepPolicy, sweep
from xsweep.errors import ContractError


class Rho(SweepModule):
    """Reference module used across the tests below."""

    contract = "loop(aot, rh) -> rho()"

    def forward(self, aot: float, rh: float) -> float:
        """Return a value identifying its inputs unambiguously."""
        return aot * 100 + rh


def test_missing_contract_fails_at_class_definition() -> None:
    """The error arrives at import, which is the whole point."""
    with pytest.raises(ContractError, match="declares no 'contract'"):

        class Broken(SweepModule):
            def forward(self, a: float) -> float:
                return a


def test_abstract_base_needs_no_contract() -> None:
    """Intermediate bases exist, so they get an explicit escape hatch."""

    class Base(SweepModule, abstract=True):
        """A shared base with no physics of its own."""

    class Concrete(Base):
        contract = "loop(a) -> out()"

        def forward(self, a: float) -> float:
            return a

    assert Concrete._contract.outputs == ("out",)


def test_contract_is_inherited_and_overridable() -> None:
    """Subclasses refine physics; the contract follows."""

    class Child(Rho):
        """Same contract, different physics."""

        def forward(self, aot: float, rh: float) -> float:
            return aot + rh

    class Other(Rho):
        contract = "loop(aot) -> rho()"

        def forward(self, aot: float) -> float:
            return aot

    assert Child._contract == Rho._contract
    assert Other._contract.inputs == ("aot",)


def test_forward_is_testable_on_its_own() -> None:
    """Pure physics must not need a sweep to be exercised."""
    assert Rho().forward(0.5, 10.0) == pytest.approx(60.0)


def test_module_and_decorator_agree(map_space: xr.Dataset) -> None:
    """Two surfaces, one engine: they must produce identical results."""

    @sweep("loop(aot, rh) -> rho()")
    def as_function(aot: float, rh: float) -> float:
        return aot * 100 + rh

    from_function = as_function(map_space)
    from_module = Rho()(map_space)
    assert np.array_equal(from_function.rho.values, from_module.rho.values)
    assert from_function.rho.dims == from_module.rho.dims


def test_policy_given_at_instantiation_applies(map_space: xr.Dataset) -> None:
    """Run configuration belongs to the instance, physics to the class."""
    plan = Rho(SweepPolicy(dedup=True)).explain(map_space)
    assert plan.n_unique is not None and plan.n_unique < plan.n_points


def test_call_level_policy_overrides_the_instance(map_space: xr.Dataset) -> None:
    """Precedence is call over instance over decorator default."""
    module = Rho(SweepPolicy(dedup=True))
    plan = module.explain(map_space, policy=SweepPolicy(dedup=False))
    assert plan.n_unique is None
