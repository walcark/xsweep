"""Space validation and loop-grid enumeration tests."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xsweep.contract import Contract
from xsweep.errors import SpaceError
from xsweep.space import build_grid, validate_space


def test_distinct_dims_give_a_cartesian_product(cartesian_space: xr.Dataset) -> None:
    """Variables on distinct dims multiply."""
    grid = build_grid(cartesian_space, Contract.parse("loop(a, b) -> out()"))
    assert grid.dims == ("a", "b")
    assert grid.shape == (3, 4)
    assert grid.n_points == 12
    assert not any(axis.is_zipped for axis in grid.axes)


def test_shared_dims_give_a_zip(map_space: xr.Dataset) -> None:
    """Variables sharing dims vary together, they do not multiply."""
    grid = build_grid(map_space, Contract.parse("loop(aot, rh) -> out()"))
    assert grid.dims == ("y", "x")
    assert grid.n_points == 6
    assert all(axis.is_zipped for axis in grid.axes)
    assert grid.axes[0].carriers == ("aot", "rh")


def test_values_are_native_python_scalars(cartesian_space: xr.Dataset) -> None:
    """Loop values reach the callable as plain Python, not 0-d arrays."""
    grid = build_grid(cartesian_space, Contract.parse("loop(a, b) -> out()"))
    values = grid.values_at((0, 2))
    assert values == {"a": pytest.approx(0.1), "b": pytest.approx(30.0)}
    assert isinstance(values["a"], float)


def test_zip_reads_the_same_position_for_both_variables(
    map_space: xr.Dataset,
) -> None:
    """A zipped point takes each variable at the same grid position."""
    grid = build_grid(map_space, Contract.parse("loop(aot, rh) -> out()"))
    assert grid.values_at((1, 0)) == {"aot": pytest.approx(0.3), "rh": 50.0}


def test_scalar_variable_creates_no_axis() -> None:
    """A 0-d variable is fixed-but-present: delivered, never swept."""
    space = xr.Dataset({"a": ("a", [1.0, 2.0]), "sza": ((), 35.0)})
    grid = build_grid(space, Contract.parse("loop(a, sza) -> out()"))
    assert grid.dims == ("a",)
    assert grid.values_at((0,)) == {"a": 1.0, "sza": 35.0}


def test_string_and_datetime_axes_are_first_class() -> None:
    """Non-numeric axes must work; adjeff coerced everything to float."""
    space = xr.Dataset(
        {
            "profile": ("profile", ["afgl_ms", "afgl_t"]),
            "when": ("when", np.array(["2026-01-01", "2026-07-01"], "datetime64[ns]")),
        }
    )
    grid = build_grid(space, Contract.parse("loop(profile, when) -> out()"))
    validate_space(space, Contract.parse("loop(profile, when) -> out()"))
    assert grid.shape == (2, 2)
    assert grid.values_at((0, 0))["profile"] == "afgl_ms"


def test_missing_variable_names_what_is_available() -> None:
    """The error must be actionable, so it lists the real variables."""
    space = xr.Dataset({"a": ("a", [1.0])})
    with pytest.raises(SpaceError, match="available variables: a"):
        validate_space(space, Contract.parse("loop(a, missing) -> out()"))


def test_object_dtype_points_at_the_label_idiom() -> None:
    """Objects break fingerprint, store and dedup alike."""
    space = xr.Dataset({"cfg": ("cfg", np.array([object()], dtype=object))})
    with pytest.raises(SpaceError, match="sweep a label string"):
        validate_space(space, Contract.parse("loop(cfg) -> out()"))


def test_empty_loop_clause_gives_one_degenerate_point() -> None:
    """A contract with no loop clause still runs once."""
    space = xr.Dataset({"A": (("y", "x"), np.zeros((2, 2)))})
    grid = build_grid(space, Contract.parse("vec(A) -> C(y, x)"))
    assert grid.n_points == 1
    assert grid.indices() == [()]
