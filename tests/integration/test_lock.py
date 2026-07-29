"""Quickstart scenario 8: one writing run per store.

The failure this prevents is not a crash but a silence: two processes
interleaving region writes produce a store nobody can trust, and the loss is
measured in engine hours.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xsweep import SweepPolicy, sweep
from xsweep.errors import StoreLockedError
from xsweep.lock import LOCK_NAME, StoreLock


@sweep("loop(a, b) -> out()", version="1")
def _f(a: float, b: float) -> float:
    """Multiply, so a value identifies its position."""
    return a * b


def test_a_second_writer_is_refused_and_told_who_holds_the_lock(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """The message must name the owner, otherwise it is not actionable."""
    store = tmp_path / "s.zarr"
    with StoreLock(store):
        with pytest.raises(StoreLockedError, match="being written by another run"):
            _f(cartesian_space, policy=SweepPolicy(store=str(store)))


def test_the_refusal_names_the_owning_process(tmp_path) -> None:
    """Knowing the pid and host is what lets a user decide what to do."""
    store = tmp_path / "s.zarr"
    with StoreLock(store):
        with pytest.raises(StoreLockedError, match=r"pid \d+ on \S+ since"):
            StoreLock(store).acquire()


def test_force_unlock_breaks_a_stale_lock(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """A crashed run must not lock a store forever."""
    store = tmp_path / "s.zarr"
    stale = StoreLock(store)
    stale.acquire()
    stale._held = False  # simulate a process that died without releasing

    result = _f(
        cartesian_space, policy=SweepPolicy(store=str(store), force_unlock=True)
    )
    assert (result.status.values == 1).all()


def test_the_lock_is_released_after_a_successful_run(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """A finished run must leave the store usable by the next one."""
    store = tmp_path / "s.zarr"
    _f(cartesian_space, policy=SweepPolicy(store=str(store)))
    assert not (store / LOCK_NAME).exists()


def test_the_lock_is_released_even_when_the_run_raises(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """Fail-fast must not also cost the user their store."""

    @sweep("loop(a, b) -> out()")
    def boom(a: float, b: float) -> float:
        raise RuntimeError("engine blew up")

    store = tmp_path / "s.zarr"
    with pytest.raises(Exception, match="engine blew up"):
        boom(cartesian_space, policy=SweepPolicy(store=str(store), on_error="raise"))
    assert not (store / LOCK_NAME).exists()


def test_reading_is_never_blocked(tmp_path, cartesian_space: xr.Dataset) -> None:
    """Following a running sweep by reading its status is the documented way."""
    store = tmp_path / "s.zarr"
    _f(cartesian_space, policy=SweepPolicy(store=str(store)))

    with StoreLock(store):
        ds = xr.open_zarr(str(store), chunks=None, consolidated=True)
        assert (ds.status.values == 1).all()
        assert np.isfinite(ds.out.values).all()


def test_the_lock_file_records_its_owner(tmp_path) -> None:
    """The payload is what the refusal message reads back."""
    store = tmp_path / "s.zarr"
    with StoreLock(store):
        meta = json.loads((store / LOCK_NAME).read_text())
    assert {"pid", "host", "since"} <= set(meta)


def test_the_in_memory_mode_takes_no_lock(cartesian_space: xr.Dataset) -> None:
    """Nothing is shared, so nothing needs protecting."""
    result = _f(cartesian_space)
    assert (result.status.values == 1).all()
