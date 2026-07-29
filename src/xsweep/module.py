"""The class facade: physics in ``forward``, orchestration in ``__call__``.

Modelled on the pytorch idiom, and for the same reason: ``forward`` stays
pure and independently testable, while everything about running it lives
elsewhere.

The contract is validated in ``__init_subclass__`` rather than by a
metaclass. No metaclass conflicts with ABCs or third-party bases, and the
contract is parsed AT IMPORT: a malformed one explodes at class definition,
not after twenty minutes of engine time.
"""

from __future__ import annotations

from typing import Any, ClassVar

import xarray as xr

from .contract import Contract, coerce
from .errors import ContractError
from .plan import Plan
from .policy import SweepPolicy
from .sweeper import Sweeper

__all__ = ["SweepModule"]


class SweepModule:
    """Base class for a sweep packaged as an object.

    Subclasses declare a ``contract`` class attribute and implement
    ``forward``. Intermediate bases that declare nothing pass
    ``abstract=True``.

    Parameters
    ----------
    policy
        Instance-level run policy, overridable per call.
    """

    contract: ClassVar[str | Contract]
    _contract: ClassVar[Contract]

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        """Parse and validate the class contract at import time."""
        super().__init_subclass__(**kwargs)
        if abstract:
            return
        spec = getattr(cls, "contract", None)
        if spec is None:
            raise ContractError(
                f"{cls.__name__} declares no 'contract'. Add one, e.g. "
                "contract = 'loop(aot, rh) -> rho()', or declare the class "
                "abstract with 'class Base(SweepModule, abstract=True)'"
            )
        cls._contract = coerce(spec)

    def __init__(self, policy: SweepPolicy | None = None) -> None:
        self.policy = policy
        self._sweeper = Sweeper(type(self)._contract, self.forward, policy)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Compute one point. Pure physics, testable on its own."""
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def explain(
        self,
        space: xr.Dataset,
        /,
        *,
        policy: SweepPolicy | None = None,
        **statics: Any,
    ) -> Plan:
        """Resolve the sweep without calling ``forward`` once."""
        return self._sweeper.explain(space, policy=policy, **statics)

    def __call__(
        self,
        space: xr.Dataset,
        /,
        *,
        policy: SweepPolicy | None = None,
        **statics: Any,
    ) -> xr.Dataset:
        """Plan the sweep, then execute it."""
        return self._sweeper(space, policy=policy, **statics)
