# Public API contract: xsweep v0

Everything importable from `xsweep`. Anything not listed here is internal and
may change without notice. Growing this list requires a constitution check
(principle II).

```python
from xsweep import sweep, Sweeper, SweepPolicy, SweepModule, Contract, Plan
from xsweep import XsweepError, ContractError, SpaceError, PolicyError
from xsweep import StoreError, StoreLockedError, PointFailed
```

`Plan` is exported as a sixth name. It is a frozen data holder returned by
`explain()`, not machinery: returning a type the user cannot import to
annotate would be a defect, and the future pipeline layer will want it
typed.

## `sweep`

```python
def sweep(
    contract: str | Contract,
    *,
    version: str = "0",
    **policy_fields: Any,
) -> Callable[[Callable[..., Any]], Sweeper]: ...
```

Decorator. Parses and validates the contract immediately, so a malformed
contract raises at import. `version` is a contract field, not a policy field
(FR-018). Remaining keywords build the decorator-level default policy.

```python
@sweep(
    "loop(aot, rh, sza) vec(wl @ 8) -> tdir_down(wl)",
    store="runs/tdir.zarr",
    version="1",
)
def tdir_down(
    aot: float, rh: float, sza: float, wl: xr.DataArray, *, n_ph: int
) -> xr.DataArray: ...
```

## `Sweeper`

```python
class Sweeper:
    def __init__(
        self,
        contract: str | Contract,
        func: Callable[..., Any],
        policy: SweepPolicy | None = None,
    ) -> None: ...

    contract: Contract
    func: Callable[..., Any]

    def __call__(
        self, space: xr.Dataset, *, policy: SweepPolicy | None = None, **statics: Any
    ) -> xr.Dataset: ...

    def explain(
        self, space: xr.Dataset, *, policy: SweepPolicy | None = None, **statics: Any
    ) -> Plan: ...
```

`__call__` plans then executes. `explain` plans and stops; it never calls the
wrapped function (FR-029, FR-031). Both accept call-level policy overrides
through `policy=`; `space` and `policy` are reserved static names.

Return value: a Dataset whose variables are the contract outputs plus
`status`. Lazy when a store is used and `load` is false, materialised
otherwise.

## `SweepPolicy`

```python
@dataclass(frozen=True)
class SweepPolicy:
    store: str | Path | None = UNSET
    chunks: Mapping[str, int] = UNSET
    dedup: bool | tuple[str, ...] = UNSET
    executor: str | Executor = UNSET
    max_workers: int | None = UNSET
    on_error: Literal["nan", "raise"] = UNSET
    retries: int = UNSET
    skip_where: Callable[[Mapping[str, Any]], bool] | None = UNSET
    load: bool = UNSET
    force_unlock: bool = UNSET
```

Resolved defaults, precedence and rationale: [research R1 and
R2](../research.md). Layers fold as defaults, decorator, instance, call, each
overriding only the fields it set.

## `SweepModule`

```python
class SweepModule:
    contract: ClassVar[str | Contract]

    def __init_subclass__(cls, abstract: bool = False, **kw: Any) -> None: ...
    def __init__(self, policy: SweepPolicy | None = None) -> None: ...

    def __call__(
        self, space: xr.Dataset, *, policy: SweepPolicy | None = None, **statics: Any
    ) -> xr.Dataset: ...
    def explain(
        self, space: xr.Dataset, *, policy: SweepPolicy | None = None, **statics: Any
    ) -> Plan: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
```

A concrete subclass without `contract` raises `ContractError` at class
definition. `abstract=True` exempts intermediate bases. The contract is
inherited and overridable. `__call__` orchestrates; `forward` is pure physics
and independently testable.

## `Contract`

```python
@dataclass(frozen=True)
class Contract:
    loop: tuple[LoopVar, ...]
    vec: tuple[VecVar, ...]
    const: tuple[str, ...]
    out: tuple[OutVar, ...]
    version: str = "0"

    @classmethod
    def parse(cls, spec: str, *, version: str = "0") -> Contract: ...

    @property
    def inputs(self) -> tuple[str, ...]: ...
    @property
    def outputs(self) -> tuple[str, ...]: ...
```

Frozen and hashable, so it enters the fingerprint. Wherever a contract is
expected, `str | Contract` is accepted and coerced. Programmatic
construction is supported and is how engine-aware planners (radtrans) will
build contracts.

## Return-value shapes accepted from the wrapped callable

| Contract outputs | Accepted return | Handling |
|------------------|-----------------|----------|
| exactly one | a DataArray, even unnamed | named after the declared output |
| several | an ordered tuple | mapped to declaration order |
| any | a Dataset | validated by name; wrong names raise `ContractError` |

## Argument delivery

| Clause | Delivered as |
|--------|--------------|
| `loop` | native Python scalar (`float`, `int`, `str`, `bool`), or DataArray with `deliver="array"` |
| `vec` | DataArray covering the current batch |
| `const` | DataArray, whole |
| statics | verbatim |

This is the only delivery mode in v0. An opt-out handing the raw per-call
Dataset to the callable is deliberately absent (FR-011) and can be added
later without breaking anything.

## Errors

```text
XsweepError
├── ContractError      definition time: grammar, coherence, return shape
├── SpaceError         planning: missing variable, object dtype, alignment
├── PolicyError        planning: unknown dim, reserved static name, bad field
├── StoreError         store: fingerprint mismatch, space change, corruption
│   └── StoreLockedError
└── PointFailed        execution: a call raised (fail-fast mode)
```

Every message states what to change, not only what is wrong.
