"""Store discipline: layout, chunk grid, fingerprint and status.

The loop-dim chunk grid governs write cost, not concurrency: execution has
exactly one writer (the parent process collecting results), so the grid is
sized from a memory budget instead of being pinned to one point per chunk.
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


def test_loop_chunk_grid_defaults_to_a_memory_budget(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """A sweep this small fits the whole loop grid in one chunk."""
    store = str(tmp_path / "s.zarr")
    _f(cartesian_space, policy=SweepPolicy(store=store))

    group = zarr.open_group(store, mode="r")
    array = group["out"]
    assert isinstance(array, zarr.Array)
    assert array.chunks == (3, 4)
    assert array.shape == (3, 4)


def test_store_chunks_can_be_set_explicitly(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """The policy override wins over the memory-budget default."""
    store = str(tmp_path / "s.zarr")
    _f(cartesian_space, policy=SweepPolicy(store=store, store_chunks={"a": 1}))

    group = zarr.open_group(store, mode="r")
    array = group["out"]
    assert isinstance(array, zarr.Array)
    assert array.chunks == (1, 4)


def test_a_wide_chunk_flushes_once_not_once_per_point(monkeypatch, tmp_path) -> None:
    """The whole point of buffering: one flush per chunk, not per point."""
    from xsweep.store import Store

    calls: list[float] = []

    @sweep("loop(a) -> out()", version="1")
    def f(a: float) -> float:
        calls.append(a)
        return a

    flushes = {"n": 0}
    original = Store.flush_chunk

    def counting(self: Store, region: object, buffer: object) -> None:
        flushes["n"] += 1
        original(self, region, buffer)  # type: ignore[arg-type]

    monkeypatch.setattr(Store, "flush_chunk", counting)

    space = xr.Dataset({"a": ("a", np.arange(100.0))})
    store = str(tmp_path / "s.zarr")
    f(space, policy=SweepPolicy(store=store, store_chunks={"a": 100}))
    assert len(calls) == 100
    assert flushes["n"] == 1


def test_status_shares_the_loop_chunk_grid(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """A zarr write's cost is dominated by its per-call overhead, not size.

    Status is one byte per point, but chunking it finer than the data it
    accompanies would still fan one buffered write out into many small ones,
    so it follows the exact same grid.
    """
    store = str(tmp_path / "s.zarr")
    _f(cartesian_space, policy=SweepPolicy(store=store))

    group = zarr.open_group(store, mode="r")
    status = group["status"]
    assert isinstance(status, zarr.Array)
    assert status.chunks == (3, 4)


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
