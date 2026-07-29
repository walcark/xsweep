"""Exception hierarchy.

Each subclass maps to the phase that raises it, which is also the phase the
user must fix. Messages state what to change, not only what is wrong
(constitution principle VII).
"""

from __future__ import annotations


class XsweepError(Exception):
    """Base class for every error raised by xsweep."""


class ContractError(XsweepError):
    """Raise when a contract is malformed, incoherent, or violated.

    Raised at definition time (decoration or class creation) for grammar and
    coherence problems, and at execution time when a wrapped callable returns
    something the contract did not declare.
    """


class SpaceError(XsweepError):
    """Raise when the sweep space cannot serve the contract.

    Raised during planning: a missing variable, a non-primitive dtype,
    coordinates that do not align exactly.
    """


class PolicyError(XsweepError):
    """Raise when a run policy cannot apply to this space.

    Raised during planning: an unknown dim in ``dedup`` or ``chunks``, a
    reserved name used as a static, an out-of-range field.
    """


class StoreError(XsweepError):
    """Raise when a store cannot be used as requested.

    Raised on open or during writes: fingerprint mismatch, changed space,
    output shape incompatible with the allocation.
    """


class StoreLockedError(StoreError):
    """Raise when a store is already being written by another run."""


class PointFailed(XsweepError):
    """Raise when a point call fails and the policy is to fail fast."""
