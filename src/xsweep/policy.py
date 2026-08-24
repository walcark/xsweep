"""Run policy: what changes the cost of a sweep, never its result.

Two types on purpose. :class:`SweepPolicy` is what users write, so every field
may be left unspecified; :class:`ResolvedPolicy` is what the rest of the
library consumes, so every field is concrete. Resolution folds the layers
defaults, decorator, instance, call, each overriding only the fields it set.

The sentinel is what makes that possible: a plain default cannot tell "not
specified" from "specified as ``False``" or ``0``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeVar

from .errors import PolicyError

__all__ = ["UNSET", "ResolvedPolicy", "SweepPolicy"]

OnError = Literal["nan", "raise"]
SkipPredicate = Callable[[Mapping[str, Any]], bool]

#: Names the sweeper takes for itself, so they cannot be static kwargs.
RESERVED_STATIC_NAMES = ("space", "policy")


class _Sentinel(Enum):
    """Single-member enum, the typing-friendly way to spell a sentinel."""

    UNSET = "UNSET"

    def __repr__(self) -> str:
        """Render as ``UNSET`` in dataclass reprs."""
        return "UNSET"


UNSET = _Sentinel.UNSET
_Unset = Literal[_Sentinel.UNSET]

T = TypeVar("T")


@dataclass(frozen=True)
class SweepPolicy:
    """Run configuration, with every field optional.

    Parameters
    ----------
    store
        Where results are persisted. ``None`` runs in memory, without cache or
        resume.
    chunks
        Batch sizes keyed by DIM, overriding any ``@ N`` in the contract.
        A dim must still be named explicitly to be batched at all, which is
        what lets a function that needs its whole axis (a convolution, a
        moving average) stay correct by simply not being named; the value
        for a named dim may be ``"auto"`` to size it from a memory budget
        instead of a fixed number.
    store_chunks
        Chunk width per loop dim in the store, overriding the automatic
        memory-budget sizing. Affects write cost only, never results: a
        wider chunk means fewer, larger writes, at the price of buffering
        more computed points in memory before they are flushed.
    batch_size
        How many points a ``batch``-delivered call receives at once. One
        number rather than one per dim: a batched call carries whole points,
        which span the product of every loop dim, not positions along one of
        them. Ignored by a contract that declares no ``batch`` clause.
    dedup
        ``True`` deduplicates over every loop dim, a tuple over the named ones.
    executor
        ``"serial"``, ``"process"``, or an object satisfying the executor
        protocol.
    max_workers
        Worker count for the process executor.
    on_error
        ``"nan"`` records a failure and continues, ``"raise"`` fails fast.
    retries
        Attempts per point before it is marked failed.
    skip_where
        Predicate on a point's values; ``True`` marks it skipped without
        calling. There is no automatic NaN-skip: an unexpected NaN is more
        often a bug than a mask.
    load
        Materialise the result instead of returning a lazy handle.
    force_unlock
        Break a lock left behind by a dead run.
    """

    store: str | Path | None | _Unset = UNSET
    chunks: Mapping[str, int | Literal["auto"]] | _Unset = UNSET
    store_chunks: Mapping[str, int] | _Unset = UNSET
    batch_size: int | _Unset = UNSET
    dedup: bool | tuple[str, ...] | _Unset = UNSET
    executor: str | Any | _Unset = UNSET
    max_workers: int | None | _Unset = UNSET
    on_error: OnError | _Unset = UNSET
    retries: int | _Unset = UNSET
    skip_where: SkipPredicate | None | _Unset = UNSET
    load: bool | _Unset = UNSET
    force_unlock: bool | _Unset = UNSET


@dataclass(frozen=True)
class ResolvedPolicy:
    """A policy with every field concrete, consumed by planning and execution."""

    store: str | Path | None = None
    chunks: Mapping[str, int | Literal["auto"]] = field(default_factory=dict)
    store_chunks: Mapping[str, int] = field(default_factory=dict)
    batch_size: int = 64
    dedup: bool | tuple[str, ...] = False
    executor: str | Any = "serial"
    max_workers: int | None = None
    on_error: OnError = "nan"
    retries: int = 0
    skip_where: SkipPredicate | None = None
    load: bool = False
    force_unlock: bool = False

    @property
    def in_memory(self) -> bool:
        """Return whether this run keeps results in memory only."""
        return self.store is None


DEFAULTS = ResolvedPolicy()


def resolve(*layers: SweepPolicy | None) -> ResolvedPolicy:
    """Fold policy layers into a concrete policy.

    Parameters
    ----------
    *layers
        Partial policies in increasing order of precedence, typically
        decorator, then instance, then call. ``None`` layers are ignored.

    Returns
    -------
    ResolvedPolicy
        The concrete policy, defaults filling whatever no layer set.
    """
    values = dataclasses.asdict(DEFAULTS)
    for layer in layers:
        if layer is None:
            continue
        if not isinstance(layer, SweepPolicy):
            raise PolicyError(
                f"policy must be a SweepPolicy, got {type(layer).__name__}. "
                "Run configuration goes through policy=SweepPolicy(...), "
                "while physics parameters are ordinary keyword arguments"
            )
        for name, value in vars(layer).items():
            if value is not UNSET:
                values[name] = value
    return ResolvedPolicy(**values)


def validate(policy: ResolvedPolicy, *, available_dims: tuple[str, ...]) -> None:
    """Check a resolved policy against the space it will run on.

    Parameters
    ----------
    policy
        The concrete policy.
    available_dims
        Dims the space actually has, used to make errors actionable.

    Raises
    ------
    PolicyError
        If a named dim does not exist, or a numeric field is out of range.
    """
    listed = ", ".join(available_dims) or "none"

    for dim, value in policy.chunks.items():
        if dim not in available_dims:
            raise PolicyError(
                f"chunks names dim {dim!r}, which the space does not have; "
                f"available dims: {listed}"
            )
        if value != "auto" and (not isinstance(value, int) or value < 1):
            raise PolicyError(
                f"chunks[{dim!r}] must be 'auto' or an int >= 1, got {value!r}"
            )
    for dim, size in policy.store_chunks.items():
        if dim not in available_dims:
            raise PolicyError(
                f"store_chunks names dim {dim!r}, which the space does not "
                f"have; available dims: {listed}"
            )
        if size < 1:
            raise PolicyError(f"store_chunks[{dim!r}] must be >= 1, got {size}")
    if isinstance(policy.dedup, tuple):
        for dim in policy.dedup:
            if dim not in available_dims:
                raise PolicyError(
                    f"dedup names dim {dim!r}, which the space does not have; "
                    f"available dims: {listed}"
                )
    if policy.retries < 0:
        raise PolicyError(f"retries must be >= 0, got {policy.retries}")
    if policy.max_workers is not None and policy.max_workers < 1:
        raise PolicyError(f"max_workers must be >= 1, got {policy.max_workers}")
    if policy.skip_where is not None and not callable(policy.skip_where):
        raise PolicyError("skip_where must be callable")
    if policy.on_error not in ("nan", "raise"):
        raise PolicyError(f"on_error must be 'nan' or 'raise', got {policy.on_error!r}")


def check_static_names(statics: Mapping[str, Any]) -> None:
    """Reject static kwargs that collide with names the sweeper takes.

    Parameters
    ----------
    statics
        The keyword arguments forwarded to every call.

    Raises
    ------
    PolicyError
        If a static uses a reserved name.
    """
    for name in statics:
        if name in RESERVED_STATIC_NAMES:
            raise PolicyError(
                f"{name!r} is reserved by the sweeper and cannot be a static "
                "argument; rename the parameter of your function"
            )
