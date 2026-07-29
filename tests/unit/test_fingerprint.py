"""Fingerprint tests: what the cache can and cannot see.

The fingerprint is the whole cache-invalidation mechanism, so the tests below
are really about honesty: anything that changes results must change it, and
anything it cannot see must be refused rather than ignored.
"""

from __future__ import annotations

import pytest

from xsweep.contract import Contract
from xsweep.errors import StoreError
from xsweep.store import fingerprint

CONTRACT = Contract.parse("loop(a, b) -> out()")


def test_same_configuration_gives_the_same_digest() -> None:
    """Determinism is what makes a resume possible at all."""
    assert fingerprint(CONTRACT, {"n_ph": 10}) == fingerprint(CONTRACT, {"n_ph": 10})


def test_bumping_the_version_changes_the_digest() -> None:
    """This is the whole point of the version field."""
    bumped = Contract.parse("loop(a, b) -> out()", version="2")
    assert fingerprint(CONTRACT, {}) != fingerprint(bumped, {})


def test_changing_the_contract_changes_the_digest() -> None:
    """A different call shape is a different computation."""
    other = Contract.parse("loop(a) -> out()")
    assert fingerprint(CONTRACT, {}) != fingerprint(other, {})


def test_changing_a_static_changes_the_digest() -> None:
    """Statics change results, so they belong to the identity."""
    assert fingerprint(CONTRACT, {"n_ph": 10}) != fingerprint(CONTRACT, {"n_ph": 20})


def test_reordering_statics_does_not_change_the_digest() -> None:
    """Keyword order carries no meaning, so it must carry no identity."""
    a = fingerprint(CONTRACT, {"n_ph": 10, "species": "afgl"})
    b = fingerprint(CONTRACT, {"species": "afgl", "n_ph": 10})
    assert a == b


def test_nested_serialisable_statics_are_supported() -> None:
    """Configuration is often a dict or a list, not a scalar."""
    assert fingerprint(CONTRACT, {"bands": [400, 500], "opts": {"x": 1}})


def test_unserialisable_static_is_refused_by_name() -> None:
    """A cache that cannot see a change is worse than no cache at all."""

    class Opaque:
        pass

    with pytest.raises(StoreError, match="__cache_token__"):
        fingerprint(CONTRACT, {"engine": Opaque()})


def test_cache_token_makes_an_object_usable() -> None:
    """The documented escape for objects that cannot be serialised."""

    class Engine:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def __cache_token__(self) -> str:
            return self.tag

    one = fingerprint(CONTRACT, {"engine": Engine("v1")})
    two = fingerprint(CONTRACT, {"engine": Engine("v1")})
    three = fingerprint(CONTRACT, {"engine": Engine("v2")})
    assert one == two
    assert one != three
