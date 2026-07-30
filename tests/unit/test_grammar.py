"""Parser tests: every production, plus one case per error-catalogue row."""

from __future__ import annotations

import pytest

from xsweep.contract import ConstVar, Contract, LoopVar, OutVar, VecVar
from xsweep.errors import ContractError


def test_full_contract() -> None:
    """Parse a contract using every clause."""
    c = Contract.parse("loop(aot, rh, sza) vec(wl @ 8) const(srf) -> t(wl)")
    assert c.loop == (LoopVar("aot"), LoopVar("rh"), LoopVar("sza"))
    assert c.vec == (VecVar("wl", 8),)
    assert c.const == (ConstVar("srf"),)
    assert c.out == (OutVar("t", ("wl",), (None,)),)


def test_const_protected_dims() -> None:
    """A const variable may protect specific dims from batch alignment."""
    c = Contract.parse("loop(a) vec(wl) const(srf, bias(x, y)) -> t(wl)")
    assert c.const == (ConstVar("srf"), ConstVar("bias", ("x", "y")))


def test_const_protected_dims_render_round_trips() -> None:
    """The protected-dims form renders back to an equal contract."""
    c = Contract.parse("loop(a) const(bias(x, y)) -> t()")
    assert Contract.parse(c.render()) == c
    assert "bias(x, y)" in c.render()


def test_scalar_output_has_no_dims() -> None:
    """Empty output parentheses mean one scalar per call."""
    c = Contract.parse("loop(a, b) -> out()")
    assert c.out == (OutVar("out", (), ()),)
    assert c.out_dims == ()


def test_vec_without_batch_means_whole_axis() -> None:
    """An unannotated vec variable takes its whole axis in one call."""
    c = Contract.parse("loop(a) vec(wl) -> t(wl)")
    assert c.vec == (VecVar("wl", None),)


def test_multiple_outputs_keep_declaration_order() -> None:
    """Several outputs are ordered, since tuples map onto that order."""
    c = Contract.parse("loop(a) -> rho(), t()")
    assert c.outputs == ("rho", "t")


def test_clause_order_is_free() -> None:
    """Clauses may appear in any order."""
    a = Contract.parse("const(srf) vec(wl) loop(aot) -> t(wl)")
    b = Contract.parse("loop(aot) vec(wl) const(srf) -> t(wl)")
    assert a == b


def test_whitespace_is_insignificant() -> None:
    """Spacing does not change the parse."""
    a = Contract.parse("loop(aot,rh)vec(wl@8)->t(wl)")
    b = Contract.parse("  loop( aot , rh )  vec( wl @ 8 )  ->  t( wl )  ")
    assert a == b


def test_render_round_trips() -> None:
    """The canonical rendering parses back to an equal contract."""
    c = Contract.parse("loop(aot, rh) vec(wl @ 8) const(srf) -> t(wl)")
    assert Contract.parse(c.render()) == c


def test_contract_is_hashable() -> None:
    """Contracts enter the fingerprint, so they must hash."""
    c = Contract.parse("loop(a) -> out()")
    assert hash(c) == hash(Contract.parse("loop(a) -> out()"))


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        ("batch(wl @ 8) -> t(wl)", "unknown clause 'batch'"),
        ("loop(a)", "no '->' clause"),
        ("vec(a) vec(b) -> t()", "clause 'vec' appears twice"),
        ("loop(a) -> t(", "expected a dim name"),
        ("loop(2wl) -> t()", "expected a variable name"),
        ("loop(a) -> t() extra()", "after the output clause"),
        ("loop(a) ! -> t()", "unexpected character '!'"),
        ("vec(wl @ 0) -> t(wl)", "batch size must be >= 1"),
    ],
)
def test_parse_errors(spec: str, fragment: str) -> None:
    """Each malformed contract raises with the documented message."""
    with pytest.raises(ContractError, match=fragment):
        Contract.parse(spec)


def test_parse_error_carries_offset() -> None:
    """Errors point at the character that caused them."""
    with pytest.raises(ContractError, match="offset 0"):
        Contract.parse("batch(wl) -> t(wl)")
