# Knowing what a run will cost

`explain` resolves the whole sweep and stops. It reads the space, applies the
policy, opens the store if there is one, and returns a `Plan` without calling
the function once.

```python
plan = layer.explain(space, policy=SweepPolicy(store="runs/layer.zarr"))
print(plan)
```

```text
╭─ layer ──────────────────────────────────────────────────────────────╮
│ loop(tau, ssa) -> reflectance(), transmittance()                     │
╰──────────────────────────────────────────────────────────────── v0 ──╯

SPACE
tau    axis     12
ssa    axis      5
points          60
dedup  disabled

CALLS
to compute 30
cached     30
skipped     0
total      60

ARGUMENTS
name  kind  dtype    shape
───────────────────────────
tau   loop  float64
ssa   loop  float64

RESULT
name           dims             dtype
────────────────────────────────────────
reflectance    tau: 12, ssa: 5  float64
transmittance  tau: 12, ssa: 5  float64

STORE
path      runs/layer.zarr
chunk tau                    12
chunk ssa                     5
regions                      60
EXECUTOR  serial
```

## Estimating the total time

The plan gives the count; you supply the cost of one call.

```python
start = time.perf_counter()
engine(one_representative_point)
per_call = time.perf_counter() - start

plan = layer.explain(space, policy=policy)
print(
    f"{plan.n_to_compute} calls x {per_call:.1f} s = {plan.n_to_compute * per_call / 3600:.1f} h"
)
```

`n_to_compute` is the honest number: it excludes what the store already has
and what `skip_where` removes. `n_points` is the size of the grid,
`n_calls` what a full run would take from scratch.

Two caveats worth saying out loud. Per-call cost is rarely uniform across a
parameter space, so time a point near the expensive end rather than the
cheap corner. And with `executor="process"` the wall clock divides by
something smaller than the worker count, never by the worker count itself.

## What else the plan is for

- **Catching an accidental product.** A point count with five too many digits
  is the symptom; see [semantics](semantics.md).
- **Seeing what one call will receive.** The arguments block gives the kind,
  dtype and shape of every argument, including the batch length of a `vec`
  variable. If that shape is not what the callee expects, you know now.
- **Checking the store lines up.** The store block shows the chunk grid and
  region count a run will write into.
- **Auditing a finished run.** Against a complete store, `n_to_compute` is
  zero and every point is cached.

Execution consumes the same plan object that `explain` returns, which is what
guarantees the report and the run cannot drift apart.

## Skipping points

`skip_where` takes a predicate on a point's loop values and excludes it
without ever calling the function:

```python
SweepPolicy(skip_where=lambda point: point["tau"] > 1.5)
```

Skipped points get a `skipped` status rather than a value, and a later run
without the predicate will compute them.

---

Worked in [Start here](../auto_examples/01_why_a_sweep_library.rst) and
[the store](../auto_examples/02_store_cache_and_resume.rst).
