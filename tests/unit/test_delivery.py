"""Argument delivery and return normalisation tests."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xsweep.contract import Contract, LoopVar, OutVar
from xsweep.delivery import (
    GROUP_DIM,
    assemble_args,
    assemble_group_args,
    normalise_return,
    split_group_return,
)
from xsweep.errors import ContractError


def test_loop_values_arrive_as_native_scalars() -> None:
    """External engines want plain Python, not 0-d arrays."""
    contract = Contract.parse("loop(aot, profile) -> out()")
    args = assemble_args(
        contract,
        loop_values={"aot": 0.1, "profile": "afgl_ms"},
        arrays={},
        statics={},
    )
    assert args == {"aot": 0.1, "profile": "afgl_ms"}
    assert isinstance(args["aot"], float)


def test_deliver_array_keeps_a_dataarray() -> None:
    """The per-variable override exists for callables that want labels."""
    contract = Contract(
        loop=(LoopVar("aot", deliver="array"),), out=(OutVar("out", (), ()),)
    )
    args = assemble_args(contract, loop_values={"aot": 0.1}, arrays={}, statics={})
    assert isinstance(args["aot"], xr.DataArray)


def test_vec_and_const_arrive_as_arrays_and_statics_verbatim() -> None:
    """Each clause has its own delivery, and statics pass through untouched."""
    contract = Contract.parse("loop(a) vec(wl @ 2) const(srf) -> t(wl)")
    wl = xr.DataArray([400.0, 405.0], dims="wl")
    srf = xr.DataArray(np.ones((2, 3)), dims=("band", "wl"))
    args = assemble_args(
        contract,
        loop_values={"a": 1.0},
        arrays={"wl": wl, "srf": srf},
        statics={"n_ph": 1000},
    )
    assert args["wl"].sizes == {"wl": 2}
    assert args["srf"].sizes == {"band": 2, "wl": 3}
    assert args["n_ph"] == 1000


def test_unnamed_dataarray_is_named_from_the_contract() -> None:
    """This kills adjeff's "func must return an unnamed DataArray" constraint."""
    contract = Contract.parse("loop(a) -> tdir(wl)")
    out = normalise_return(xr.DataArray([1.0, 2.0], dims="wl"), contract)
    assert list(out.data_vars) == ["tdir"]


def test_bare_scalar_is_accepted_for_a_scalar_output() -> None:
    """A callable returning a float is legitimate when the output has no dims."""
    out = normalise_return(3.5, Contract.parse("loop(a) -> out()"))
    assert float(out["out"]) == 3.5


def test_dict_maps_onto_declared_names() -> None:
    """Multi-output contracts read a dict by name, order does not matter."""
    contract = Contract.parse("loop(a) -> rho(wl), t(wl)")
    rho = xr.DataArray([1.0, 2.0], dims="wl")
    t = xr.DataArray([3.0, 4.0], dims="wl")
    out = normalise_return({"t": t, "rho": rho}, contract)
    assert list(out.data_vars) == ["rho", "t"]
    assert float(out["t"][0]) == 3.0


def test_dict_missing_an_output_is_rejected() -> None:
    """A wrong or missing key is a contract violation, not a silent omission."""
    contract = Contract.parse("loop(a) -> rho(wl), t(wl)")
    with pytest.raises(ContractError, match="missing \\['t'\\]"):
        normalise_return({"rho": xr.DataArray([1.0], dims="wl")}, contract)


def test_a_tuple_for_multiple_outputs_is_refused() -> None:
    """Positional matching would let a swapped pair go undetected, so it is gone."""
    contract = Contract.parse("loop(a) -> rho(wl), t(wl)")
    rho = xr.DataArray([1.0, 2.0], dims="wl")
    t = xr.DataArray([3.0, 4.0], dims="wl")
    with pytest.raises(ContractError, match="not a tuple"):
        normalise_return((rho, t), contract)


def test_dataset_is_validated_by_name() -> None:
    """An explicit Dataset is checked nominatively, extras are dropped."""
    contract = Contract.parse("loop(a) -> rho(wl)")
    ds = xr.Dataset(
        {"rho": ("wl", [1.0]), "debug": ("wl", [9.0])},
    )
    out = normalise_return(ds, contract)
    assert list(out.data_vars) == ["rho"]


def test_dataset_missing_an_output_is_rejected() -> None:
    """A wrong name is a contract violation, not a silent omission."""
    contract = Contract.parse("loop(a) -> rho(wl), t(wl)")
    ds = xr.Dataset({"rho": ("wl", [1.0])})
    with pytest.raises(ContractError, match="missing \\['t'\\]"):
        normalise_return(ds, contract)


@pytest.mark.parametrize(
    ("result", "spec", "fragment"),
    [
        ((1.0, 2.0), "loop(a) -> out()", "returned a tuple of 2"),
        (1.0, "loop(a) -> rho(), t()", "not a tuple"),
        ((1.0,), "loop(a) -> rho(), t()", "not a tuple"),
    ],
)
def test_return_shape_mismatches(result: object, spec: str, fragment: str) -> None:
    """Mismatches between declaration and return are named precisely."""
    with pytest.raises(ContractError, match=fragment):
        normalise_return(result, Contract.parse(spec))


# --- batched delivery ---


def test_group_args_stack_batched_variables() -> None:
    """Batched variables arrive as one aligned array over the group dim."""
    contract = Contract.parse("batch(aot, rh) -> t()")

    args = assemble_group_args(
        contract,
        group_values=[{"aot": 0.1, "rh": 50.0}, {"aot": 0.4, "rh": 80.0}],
        arrays={},
        statics={},
    )

    assert args["aot"].dims == (GROUP_DIM,)
    np.testing.assert_allclose(args["aot"].values, [0.1, 0.4])
    np.testing.assert_allclose(args["rh"].values, [50.0, 80.0])


def test_group_args_keep_a_scalar_loop_variable_scalar() -> None:
    """A loop variable shared by the group is not stacked."""
    contract = Contract.parse("loop(sza) batch(aot) -> t()")

    args = assemble_group_args(
        contract,
        group_values=[{"sza": 40.0, "aot": 0.1}, {"sza": 40.0, "aot": 0.4}],
        arrays={},
        statics={},
    )

    assert args["sza"] == 40.0
    assert args["aot"].sizes[GROUP_DIM] == 2


def test_a_scalar_loop_variable_may_not_vary_within_a_group() -> None:
    """Grouping points that disagree on a scalar variable is ambiguous."""
    contract = Contract.parse("loop(sza) batch(aot) -> t()")

    with pytest.raises(ContractError, match="takes 2 values"):
        assemble_group_args(
            contract,
            group_values=[{"sza": 40.0, "aot": 0.1}, {"sza": 60.0, "aot": 0.4}],
            arrays={},
            statics={},
        )


def test_split_drops_the_group_dim() -> None:
    """Each point's Dataset looks exactly like an unbatched return."""
    result = xr.Dataset(
        {"t": xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=[GROUP_DIM, "wl"])}
    )

    parts = split_group_return(result, 2)

    assert len(parts) == 2
    assert parts[0]["t"].dims == ("wl",)
    np.testing.assert_allclose(parts[1]["t"].values, [3.0, 4.0])
