"""Store discipline: layout, chunk grid, fingerprint and status.

The chunk grid is what makes parallel region writes safe without any
coordination, so it is checked directly rather than inferred from behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import zarr

from xsweep import SweepPolicy, sweep
from xsweep.errors import StoreError


@sweep("loop(a, b) -> out()", version="1")
def _f(a: float, b: float) -> float:
    """Multiply, so values identify their position unambiguously."""
    return a * b


def test_chunk_grid_is_one_per_loop_point(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """One loop point per chunk is what lets two writers never collide."""
    store = str(tmp_path / "s.zarr")
    _f(cartesian_space, policy=SweepPolicy(store=store))

    group = zarr.open_group(store, mode="r")
    array = group["out"]
    assert isinstance(array, zarr.Array)
    assert array.chunks == (1, 1)
    assert array.shape == (3, 4)


def test_status_is_allocated_over_the_loop_dims(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """Resume reads this array, so its shape must match the loop grid."""
    store = str(tmp_path / "s.zarr")
    result = _f(cartesian_space, policy=SweepPolicy(store=store))
    assert result.status.dims == ("a", "b")
    assert result.status.dtype == np.uint8
    assert (result.status.values == 1).all()


def test_metadata_records_the_configuration(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """The fingerprint is what lets a resume tell same from different."""
    store = str(tmp_path / "s.zarr")
    _f(cartesian_space, policy=SweepPolicy(store=store))

    group = zarr.open_group(store, mode="r")
    meta = dict(group.attrs["xsweep_meta"])
    assert meta["contract"] == "loop(a, b) -> out()"
    assert meta["version"] == "1"
    assert len(meta["fingerprint"]) == 32


def test_a_different_version_refuses_the_store(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """Bumping the version must never silently reuse stale physics."""
    store = str(tmp_path / "s.zarr")
    _f(cartesian_space, policy=SweepPolicy(store=store))

    @sweep("loop(a, b) -> out()", version="2")
    def bumped(a: float, b: float) -> float:
        return a * b

    with pytest.raises(StoreError, match="different configuration"):
        bumped(cartesian_space, policy=SweepPolicy(store=store))


def test_a_different_static_refuses_the_store(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """Statics change results, so they are part of the store's identity."""

    @sweep("loop(a, b) -> out()", version="1")
    def g(a: float, b: float, *, n_ph: int) -> float:
        return a * b * n_ph

    store = str(tmp_path / "s.zarr")
    g(cartesian_space, policy=SweepPolicy(store=store), n_ph=10)
    with pytest.raises(StoreError, match="different configuration"):
        g(cartesian_space, policy=SweepPolicy(store=store), n_ph=20)


def test_an_extended_axis_refuses_the_store(tmp_path) -> None:
    """Resuming against a changed space would misalign every result."""
    small = xr.Dataset({"a": ("a", [1.0, 2.0]), "b": ("b", [10.0])})
    store = str(tmp_path / "s.zarr")
    _f(small, policy=SweepPolicy(store=store))

    larger = xr.Dataset({"a": ("a", [1.0, 2.0, 3.0]), "b": ("b", [10.0])})
    with pytest.raises(StoreError, match="different 'a' axis"):
        _f(larger, policy=SweepPolicy(store=store))


def test_a_foreign_directory_is_refused(tmp_path) -> None:
    """Writing into someone else's zarr would corrupt it."""
    store = tmp_path / "foreign.zarr"
    zarr.open_group(str(store), mode="w")
    space = xr.Dataset({"a": ("a", [1.0]), "b": ("b", [1.0])})
    with pytest.raises(StoreError, match="not written by xsweep"):
        _f(space, policy=SweepPolicy(store=str(store)))


def test_result_is_lazy_by_default_and_needs_no_dask(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """Laziness here is xarray's own indexing, dask is not involved."""
    store = str(tmp_path / "s.zarr")
    result = _f(cartesian_space, policy=SweepPolicy(store=store))
    backend = type(result.out.variable._data).__name__
    assert "Array" in backend
    assert "dask" not in backend.lower()


def test_load_materialises_on_request(tmp_path, cartesian_space: xr.Dataset) -> None:
    """Materialising is the user's choice, never the library's."""
    store = str(tmp_path / "s.zarr")
    result = _f(cartesian_space, policy=SweepPolicy(store=store, load=True))
    assert isinstance(result.out.variable._data, np.ndarray)
