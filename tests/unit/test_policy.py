"""Policy resolution tests.

The interesting cases are the falsy ones: a plain default cannot tell "not
specified" from "specified as False", and getting that wrong would silently
ignore a user's explicit choice.
"""

from __future__ import annotations

import pytest

from xsweep.errors import PolicyError
from xsweep.policy import DEFAULTS, SweepPolicy, check_static_names, resolve, validate


def test_defaults_when_no_layer_sets_anything() -> None:
    """Resolving nothing yields the documented defaults."""
    assert resolve() == DEFAULTS
    assert resolve(None, SweepPolicy()) == DEFAULTS


def test_later_layer_wins() -> None:
    """Precedence runs left to right: defaults, decorator, instance, call."""
    got = resolve(
        SweepPolicy(store="a.zarr", retries=1),
        SweepPolicy(retries=2),
        SweepPolicy(retries=3),
    )
    assert got.store == "a.zarr"
    assert got.retries == 3


def test_explicit_false_overrides_a_truthy_lower_layer() -> None:
    """An explicit False must win, which a plain default could not express."""
    got = resolve(SweepPolicy(dedup=True), SweepPolicy(dedup=False))
    assert got.dedup is False


def test_explicit_zero_overrides_a_nonzero_lower_layer() -> None:
    """Same problem with 0, the other falsy value users legitimately set."""
    got = resolve(SweepPolicy(retries=5), SweepPolicy(retries=0))
    assert got.retries == 0


def test_unset_field_inherits_rather_than_resetting() -> None:
    """A layer that says nothing about a field leaves it alone."""
    got = resolve(SweepPolicy(store="a.zarr"), SweepPolicy(retries=2))
    assert got.store == "a.zarr"


def test_in_memory_is_derived_from_store() -> None:
    """No store named means the run keeps everything in memory."""
    assert resolve().in_memory is True
    assert resolve(SweepPolicy(store="a.zarr")).in_memory is False


@pytest.mark.parametrize(
    ("policy", "fragment"),
    [
        (SweepPolicy(chunks={"z": 4}), "chunks names dim 'z'"),
        (SweepPolicy(chunks={"y": 0}), "chunks\\['y'\\] must be 'auto' or an int"),
        (SweepPolicy(chunks={"y": "soon"}), "chunks\\['y'\\] must be 'auto' or an int"),
        (SweepPolicy(dedup=("z",)), "dedup names dim 'z'"),
        (SweepPolicy(retries=-1), "retries must be >= 0"),
        (SweepPolicy(max_workers=0), "max_workers must be >= 1"),
    ],
)
def test_validation_errors(policy: SweepPolicy, fragment: str) -> None:
    """Invalid policies fail before any expensive call, naming the fix."""
    with pytest.raises(PolicyError, match=fragment):
        validate(resolve(policy), available_dims=("y", "x"))


def test_unknown_dim_error_lists_what_is_available() -> None:
    """The message must be actionable, so it lists the real dims."""
    with pytest.raises(PolicyError, match="available dims: y, x"):
        validate(resolve(SweepPolicy(chunks={"z": 4})), available_dims=("y", "x"))


@pytest.mark.parametrize("name", ["space", "policy"])
def test_reserved_static_names_are_rejected(name: str) -> None:
    """Statics share a namespace with the sweeper's own arguments."""
    with pytest.raises(PolicyError, match="reserved by the sweeper"):
        check_static_names({name: 1})


def test_ordinary_static_names_pass() -> None:
    """Anything not reserved is accepted verbatim."""
    check_static_names({"n_ph": 1000, "species": "afgl"})
