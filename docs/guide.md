# Guide

A progressive tour of xsweep, from the smallest possible sweep to the knobs
you reach for only once a run gets expensive. Each section links to a
runnable companion script in `scripts/`; run them as you read.

## 1. Why xsweep

You already have the tools to do this without xsweep: `xr.apply_ufunc`
vectorises a function over a `Dataset`, and it does that well. xsweep exists
for what `apply_ufunc` does not do:

- **Deduplication.** A satellite scene's per-pixel atmospheric parameters
  repeat constantly (clear-sky pixels, flat regions). xsweep collapses
  identical input rows before calling anything, and expands the result back
  afterwards. `apply_ufunc` calls once per pixel, always.
- **A cache that is also the output.** Persist a sweep to a zarr store and
  it becomes memoisation, resumability and bounded memory in one mechanism.
  Interrupt a run at point 4000 of 5000 and relaunch: only the missing
  points are recomputed. `apply_ufunc` has no notion of "already done".
- **Concurrent writes handled for you.** A parallel sweep writing into a
  shared store is a coordination problem (which chunk, which writer, what
  happens to a duplicate). xsweep resolves this once, centrally, rather
  than leaving every caller to reinvent it.
- **A declaration of what one call consumes**, separate from how the data
  happens to be laid out. `loop(aot, rh) vec(wl) const(srf) -> t(wl)` reads
  the same whether `aot` is a 1-D axis or a 2-D pixel map; `apply_ufunc`
  makes you encode that in `input_core_dims` by hand, differently each time.

If your function is already vectorised, cheap, and you never need to stop
and resume, `apply_ufunc` alone is the right tool and xsweep would add
nothing. xsweep earns its place when at least one of the above is true:
expensive calls, a map with repeats, or a run you cannot afford to restart
from zero.

## 2. The sweep in five minutes

The smallest useful sweep: a scalar function, a Cartesian space, one call.

```python
from xsweep import sweep
import xarray as xr


@sweep("loop(a, b) -> out()")
def f(a: float, b: float) -> float:
    return a * b


space = xr.Dataset({"a": ("a", [1.0, 2.0, 3.0]), "b": ("b", [10.0, 20.0])})
result = f(space)  # out(a: 3, b: 2), six calls
```

No store, no policy: everything runs in memory and `f(space)` returns a
lazy-looking but fully in-memory `Dataset`. This is the shape most sweeps
start in, before persistence or dedup are needed at all. Run
`scripts/01_first_sweep.py` to see it end to end, including what
`f.explain(space)` reports before a single call happens.

## 3. Semantics come from your data, not from the library

The contract never says how many points there are, or whether two variables
vary together. That comes entirely from the dims your `xr.Dataset` already
has:

```python
# distinct dims: Cartesian product, 3 x 2 = 6 points
xr.Dataset({"a": ("a", [1.0, 2.0, 3.0]), "b": ("b", [10.0, 20.0])})

# shared dims: zip, one point per pixel, not a product
xr.Dataset({"aot": (("y", "x"), aot_map), "rh": (("y", "x"), rh_map)})
```

A contract written as `loop(aot, rh) -> rho()` serves both spaces unchanged.
This is also where an accidental product hides: two variables meant to vary
together but declared on different dims silently explode into every
combination. `f.explain(space)` shows the axes and their origin (zip or
product) before any call runs, which is the cheapest place to catch it.

## 4. Passing more than a scalar: vec and const

Not every function wants one value per call. `vec` hands a whole axis (or
batches of it) to the callable; `const` hands context data that is not
itself being swept:

```python
@sweep("loop(aot, rh) vec(wl) const(srf) -> radiance(band)")
def radiance(aot, rh, wl, srf):
    spectrum = physics(aot, rh, wl)
    return (spectrum * srf).sum("wl")  # srf integrates the whole axis
```

`srf` here needs the whole `wl` axis to integrate correctly; batching `wl`
would integrate each batch separately and silently corrupt the sum. The
contract refuses that combination at decoration time, before any call. If a
`const` variable happens to share a dim with something that IS batched
elsewhere, it auto-aligns to the active batch instead of causing a shape
mismatch; `const(bias(x, y))` protects specific dims from that if the
callee genuinely needs them whole regardless. See
`scripts/04_band_integration.py` for the full contract, including the
refusal.

## 5. Making it persistent: the store

Point a policy at a path and the same sweep gains a cache, bounded memory,
and resumability, with no change to the function or the contract:

```python
from xsweep import SweepPolicy

result = f(space, policy=SweepPolicy(store="runs/f.zarr"))
```

A `status` variable rides along with the result (`ok`, `failed`, `skipped`,
`pending`), which is what makes "what happened" inspectable without
re-running anything. Interrupt a long sweep and call it again with the same
store: only points without an `ok` status are recomputed, and a fresh run
against a finished store makes zero calls. This is the mechanism `apply_ufunc`
has no equivalent for. See `scripts/03_failure_and_resume.py` for a failing
engine, a resume, and the fail-fast alternative (`on_error="raise"`).

## 6. Not paying twice: dedup

A satellite scene rarely has as many distinct pixel values as it has
pixels. `dedup=True` computes each unique input row once and copies the
result onto every pixel that shares it:

```python
result = f(space, policy=SweepPolicy(dedup=True))
```

On a 3600-pixel map with 25 unique rows, this divides the number of calls
by 144. With an expensive engine that is nearly the whole story; the
duplicate positions still need filling in the store, which xsweep does in
one batched pass rather than one write per pixel (see
[limitations](limitations.md) for the numbers). `dedup` never changes a
value or a shape, only how many times the callee runs to produce them; see
`scripts/02_pixel_map_dedup.py`.

## 7. Tuning cost without touching the result: SweepPolicy

Everything in `SweepPolicy` is a cost decision, never a value one: a
release gate in the test suite checks that the result is bit-identical
across every policy combination. Three temporalities stay separate on
purpose:

- **Contract** = physics, fixed when the function is written.
- **Policy** = run configuration, fixed at call, instance, or decorator
  level (call wins, then instance, then decorator default).
- **Data** = the space and static kwargs, fixed at call time.

The knobs that matter once a sweep gets big:

- `chunks={"wl": 500}` overrides a `vec` batch size, keyed by dim.
  `chunks={"wl": "auto"}` batches it too, sized from a memory budget instead
  of a number you pick yourself; the dim still has to be named, since only
  you know whether your function is safe to run on pieces of that axis (a
  convolution or a moving average is not, even when its output has the same
  shape as its input).
- `store_chunks={"y": 256}` sizes the store's loop-dim chunk grid; by
  default it is sized from a memory budget, not exposed unless you need to
  override it.
- `executor="process"` / `max_workers=N` parallelises across processes.
- `on_error="raise"` stops at the first failure instead of recording it and
  continuing; `retries=N` gives a flaky call another chance first.
- `skip_where=lambda point: ...` excludes points from a predicate on their
  loop values, without ever calling the function for them.

```python
plain = f(space)
tuned = f(space, policy=SweepPolicy(dedup=True, store_chunks={"a": 2}))
# tuned costs less to write; plain and tuned agree exactly on every value
```

## 8. State that survives across calls: the class facade

A bare `@sweep` function has nowhere clean to put state built once and
reused across calls (an engine handle, a loaded lookup table): a closure
breaks the pickling the process executor needs, and a module global is not
much better. `SweepModule` gives that state a home, modelled on the
PyTorch idiom:

```python
from xsweep import SweepModule


class RhoAtm(SweepModule):
    contract = "loop(aot, rh) vec(wl @ 8) -> rho_atm(wl)"

    def __init__(self, policy=None, engine=None):
        super().__init__(policy)
        self.engine = engine  # built once, reused by every call

    def forward(self, aot, rh, wl, *, n_ph):
        return self.engine.run(aot, rh, wl, n_ph)


mod = RhoAtm(
    SweepPolicy(store="runs/rho.zarr", executor="process"), engine=load_engine()
)
result = mod(space, n_ph=int(1e6))
```

`forward` stays pure physics, testable with no sweep involved
(`RhoAtm(engine=...).forward(0.1, 50.0, wl, n_ph=100)`); `__call__` does the
orchestration. This is also the combination that most needs
`executor="process"` (an expensive engine, built once, run in parallel),
and the instance's own state now survives the trip into worker processes
intact. If a bare function is enough for your case (no per-instance setup),
the decorator is simpler and gets you the same contract and policy
machinery; reach for the class when `__init__` earns its keep. See
`scripts/05_module_and_policy.py`.

## 9. Going further

A few situations are solved with a convention rather than new machinery,
documented in full in [idioms](idioms.md):

- **Replication / Monte-Carlo repeats**: a bare `rep` dim does not work,
  since a contract loops over variables, not dims; a seed *variable* on
  that dim does.
- **Comparing two versions of the same physics**: `version` selects code,
  never becomes a sweep axis, and one store per version keeps the
  comparison explicit.
- **Object-valued parameters**: sweep coordinates must be primitives; sweep
  a label string and pass the object as a static exposing
  `__cache_token__()`.

Two more advanced contract features, covered in
[contract-dsl.md](../specs/001-xsweep-v0/contracts/contract-dsl.md):

- `const(bias(x, y))` protects specific dims of a context variable from
  ever being sliced, even if a `vec` variable elsewhere shares that dim.
- Batching a multi-dim `vec` variable goes through `policy.chunks`, keyed
  by dim rather than the `@ N` shorthand, which only applies to 1-D
  variables.

## 10. Where to go next

- [Idioms](idioms.md): the recipes above, in full.
- [Limitations](limitations.md): what xsweep deliberately does not do, and
  the measured cost of the store's chunk grid.
- [Design reference](design/xsweep.md): why it is built this way, and the
  alternatives that were rejected.
- [Implementation findings](implementation-findings.md): bugs and
  measurements that corrected the design after it was built.
