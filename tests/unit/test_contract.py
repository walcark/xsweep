"""Contract coherence tests: the rules that need no space to check."""

from __future__ import annotations

import pytest

from xsweep.contract import Contract, LoopVar, OutVar, coerce
from xsweep.errors import ContractError


def test_name_in_two_clauses_is_rejected() -> None:
    """A variable cannot be consumed two ways at once."""
    with pytest.raises(ContractError, match="both vec and const"):
        Contract.parse("vec(wl) const(wl) -> t(wl)")


def test_duplicate_output_is_rejected() -> None:
    """Two outputs cannot share a name, since the result is a Dataset."""
    with pytest.raises(ContractError, match="output 'rho' declared twice"):
        Contract.parse("loop(a) -> rho(), rho()")


def test_batching_a_reduced_dim_is_rejected() -> None:
    """Batching an axis the callable reduces over silently corrupts results.

    Without a space, this error cannot distinguish "the function reduces over
    wl" from "wl is multi-dim", so the message names both readings.
    """
    with pytest.raises(ContractError, match="not among the declared output dims"):
        Contract.parse("loop(aot) vec(wl @ 8) const(srf) -> band_int(band)")


def test_whole_axis_over_a_reduced_dim_is_allowed() -> None:
    """Passing the whole axis to a reducing callable is the correct form."""
    c = Contract.parse("loop(aot) vec(wl) const(srf) -> band_int(band)")
    assert c.vec[0].max_batch is None


def test_inputs_and_outputs_introspection() -> None:
    """Introspection is what the future pipeline layer will wire on."""
    c = Contract.parse("loop(aot, rh) vec(wl @ 4) const(srf) -> rho(wl), t(wl)")
    assert c.inputs == ("aot", "rh", "wl", "srf")
    assert c.outputs == ("rho", "t")
    assert c.out_dims == ("wl",)


def test_version_defaults_and_enters_equality() -> None:
    """Version is part of the contract, so it changes identity."""
    a = Contract.parse("loop(a) -> out()")
    b = Contract.parse("loop(a) -> out()", version="2")
    assert a.version == "0"
    assert a != b


def test_coerce_accepts_both_forms() -> None:
    """A string and an object are interchangeable wherever a contract is taken."""
    obj = Contract(loop=(LoopVar("a"),), out=(OutVar("out", (), ()),))
    assert coerce("loop(a) -> out()") == obj
    assert coerce(obj) is obj


def test_coerce_rejects_conflicting_versions() -> None:
    """Declaring the version twice with different values is a real conflict."""
    obj = Contract.parse("loop(a) -> out()", version="3")
    with pytest.raises(ContractError, match="declare it in one place only"):
        coerce(obj, version="4")


def test_output_named_after_its_own_dim_is_rejected() -> None:
    """In a Dataset such a variable would be its own coordinate.

    Found by running a band-integration sweep: allocation collided in the
    store, which is far too late for an error the contract can catch.
    """
    with pytest.raises(ContractError, match="its own dims"):
        Contract.parse("loop(aot) vec(wl) const(srf) -> band(band)")
