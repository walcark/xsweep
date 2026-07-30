"""Quickstart scenario 8: one writing run per store.

The failure this prevents is not a crash but a silence: two processes
interleaving region writes produce a store nobody can trust, and the loss is
measured in engine hours.
"""

from __future__ import annotations

import json
import os
import signal
import threading

import numpy as np
import pytest
import xarray as xr

from xsweep import SweepPolicy, sweep
from xsweep.errors import StoreLockedError
from xsweep.lock import LOCK_SUFFIX, StoreLock


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
    assert not store.with_name(store.name + LOCK_SUFFIX).exists()


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
    assert not store.with_name(store.name + LOCK_SUFFIX).exists()


def test_reading_is_never_blocked(tmp_path, cartesian_space: xr.Dataset) -> None:
    """Following a running sweep by reading its status is the documented way."""
    store = tmp_path / "s.zarr"
    _f(cartesian_space, policy=SweepPolicy(store=str(store)))

    with StoreLock(store):
        ds = xr.open_zarr(str(store), chunks=None, consolidated=True)
        assert (ds.status.values == 1).all()
        assert np.isfinite(ds.out.values).all()


def test_the_lock_sits_beside_the_store_not_inside_it(tmp_path) -> None:
    """A stray file inside a zarr directory makes every open warn."""
    store = tmp_path / "s.zarr"
    with StoreLock(store):
        assert store.with_name(store.name + LOCK_SUFFIX).exists()
        assert not (store / "xsweep-lock.json").exists()


def test_the_lock_file_records_its_owner(tmp_path) -> None:
    """The payload is what the refusal message reads back."""
    store = tmp_path / "s.zarr"
    with StoreLock(store):
        meta = json.loads(store.with_name(store.name + LOCK_SUFFIX).read_text())
    assert {"pid", "host", "since"} <= set(meta)


def test_the_in_memory_mode_takes_no_lock(cartesian_space: xr.Dataset) -> None:
    """Nothing is shared, so nothing needs protecting."""
    result = _f(cartesian_space)
    assert (result.status.values == 1).all()


def test_releasing_twice_is_a_no_op(tmp_path) -> None:
    """A second release must not raise or touch a file it no longer owns."""
    store = tmp_path / "s.zarr"
    lock = StoreLock(store)
    lock.acquire()
    lock.release()
    assert not lock.path.exists()
    lock.release()  # already released: must not raise


def test_releasing_a_lock_never_acquired_is_a_no_op(tmp_path) -> None:
    """A lock object that never took the lock has nothing to give back."""
    StoreLock(tmp_path / "s.zarr").release()


def test_the_owner_is_unknown_when_the_lock_file_is_corrupted(tmp_path) -> None:
    """A refusal must still name something, even if the payload is unreadable."""
    store = tmp_path / "s.zarr"
    lock_path = store.with_name(store.name + LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("not json")

    with pytest.raises(StoreLockedError, match="unknown owner"):
        StoreLock(store).acquire()


def test_sigterm_releases_the_lock_and_raises_keyboard_interrupt(tmp_path) -> None:
    """A killed run must not leave the store locked forever."""
    store = tmp_path / "s.zarr"
    lock = StoreLock(store)
    lock.acquire()
    try:
        with pytest.raises(KeyboardInterrupt, match="interrupted by signal"):
            os.kill(os.getpid(), signal.SIGTERM)
        assert not lock.path.exists()
    finally:
        lock._held = False  # already released by the handler


def test_installing_the_handler_off_the_main_thread_degrades_quietly(
    tmp_path,
) -> None:
    """signal.signal() only works on the main thread; a worker must not crash."""
    store = tmp_path / "s.zarr"
    outcome: dict[str, object] = {}

    def run() -> None:
        lock = StoreLock(store)
        try:
            lock.acquire()
            outcome["previous"] = lock._previous
            lock.release()
        except Exception as exc:  # noqa: BLE001 - surfaced to the main thread
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    assert "error" not in outcome
    assert outcome["previous"] is None


def test_restoring_the_handler_tolerates_a_value_error(tmp_path, monkeypatch) -> None:
    """A restore that fails (e.g. the thread context shifted) must not crash release().

    signal.signal() only raises ValueError off the main thread, which acquire()
    already turned into `_previous = None`; forcing a failure here covers the
    symmetric, harder-to-provoke case where install succeeded but restore does not.
    """
    import signal as signal_module

    store = tmp_path / "s.zarr"
    lock = StoreLock(store)
    lock.acquire()
    assert lock._previous is not None

    def broken_signal(signalnum: int, handler: object) -> object:
        raise ValueError("simulated: cannot restore here")

    monkeypatch.setattr(signal_module, "signal", broken_signal)
    lock.release()  # must not raise despite the failing restore
    assert not lock.path.exists()
