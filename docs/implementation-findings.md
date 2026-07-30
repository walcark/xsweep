# Implementation findings

What building v0 revealed that reviewing the design had not. Every entry
below was found by running code, not by rereading a document, which is the
reason to write them down: they are the things a specification cannot catch.

Dates: 2026-07-29, against zarr 3.2.1, xarray 2026.7.0, numpy 2.5.1.

## Bugs and design errors found while implementing

### 1. xarray's zarr writer makes dask mandatory

`Dataset.to_zarr(compute=False)` returns a delayed object and therefore
imports dask. So does `Dataset.chunk()`. Both were the obvious way to
pre-allocate a store and declare its chunk grid, and both would have turned
an optional dependency into a required one, in direct violation of principle
X.

Allocation now goes through the zarr API (`zarr.open_group`, then
`create_array` with `dimension_names` and a `fill_value`), followed by
`zarr.consolidate_metadata`. Region writes and lazy reads need no dask.

Recorded in research R6. A useful corollary for anyone touching `store.py`:
any xarray API that could return a delayed object is off limits there.

### 2. The process executor could never have worked

The decorator rebinds the module-level name to the `Sweeper`, so the wrapped
function is no longer reachable at its own qualified name. Every
process-backed sweep died at pickling, with an error blaming the user's
function.

The sweeper now pickles by reference (`__reduce__` returning a module plus
qualname lookup), so a worker re-imports the module and finds the same
object, function included. The per-item callable is a `functools.partial`
rather than a lambda, for the same reason.

The lesson generalises: a decorator that replaces a name breaks every
mechanism that resolves objects by name, and multiprocessing is one of them.

### 3. Coordinates carried back by a call were written outside their region

A callable returning `wl * aot` hands back the `wl` coordinate along with the
values. Writing that dataset by region then tried to write the coordinate
too, which spans the whole axis rather than the region.

Coordinates are now dropped from a call output before writing: the store
already holds them from allocation.

### 4. An output named after one of its own dims collides in the store

`-> band(band)` looks natural for a band integration, and it cannot work: in
a Dataset a variable sharing a dim's name IS that dim's coordinate, so the
output and its own axis are the same key. It surfaced as
`ContainsArrayError` at allocation, long after the contract could have caught
it.

Now refused at definition, with a message proposing the rename.

### 5. A contract with no loop clause produced an inconsistent status array

With no `loop` clause the loop grid is empty, so the status array is 0-d.
Its chunk grid was being built as `(1,)`, which zarr rejects for a shape of
`()`. Found by the multi-dim vec test, which is exactly the contract shape
that has no loop clause.

### 6. Passing a non-policy as `policy=` gave an opaque TypeError

`vars()` on an int fails with a message naming neither the argument nor the
fix. Now a `PolicyError` explaining that run configuration goes through
`SweepPolicy` while physics parameters are ordinary keyword arguments.

### 7. The lock made the store believe it already existed

The write lock creates the store directory before anything is allocated in
it, and store detection tested directory existence. Every locked run then
tried to open a group that was not there yet.

Detection now tests for the group metadata (`zarr.json`), which is also the
more accurate question: an empty directory is not a store.

### 8. `space` had to become positional-only

A static named `space` collided with the parameter before the reserved-name
check could run, producing `TypeError: got multiple values for argument
'space'`. Making `space` positional-only lets such a static reach the
statics, where the check produces an actionable message.

A related discovery: `policy` is keyword-only, so a static of that name binds
to the parameter and can never reach the statics. Its entry in the reserved
list is therefore unreachable. It is kept for symmetry, and the test says so.

### 9. A const variable sharing a batched dim could mismatch shapes

`const` was documented and implemented as "always whole", full stop. That is
too strong: if a const variable happens to share a dim with a batched `vec`
variable, handing it the whole axis while the vec variable arrives sliced
gives the callee two arrays of different sizes on what should be the same
axis, either a hard shape error or, worse, a silent misalignment if xarray
resolves it through coordinate-based alignment instead.

Const variables now auto-align to the active batch on any dim they share
with one, the same slicing `vec` already gets. An escape hatch stays
available for the case where the callee genuinely needs the full axis of a
const array regardless of what else is being batched (a normalisation, a
reduction): `const(bias(x, y))` protects `x` and `y` on `bias` from ever
being sliced. `explain()`'s reported shape for a const argument now reflects
whichever applies.

### 10. The reduced-dim batching guard only covered the 1-D shorthand

The `@ N` marker refuses to batch a dim absent from the contract's declared
outputs, since the function would then be reducing over a dim being split
into pieces, corrupting the result. `policy.chunks`, the multi-dim
equivalent for batching a `vec` variable with more than one dim, checked
only that the named dim belonged to some vec variable, not that it survived
to the output. The same corruption was reachable through the multi-dim path
with no refusal at all. Both now share the same `dim in contract.out_dims`
check.

### 11. SweepModule combined with the process executor was broken (2026-07-30)

The README's own example (`RhoAtm(SweepPolicy(..., executor="process"))`)
does not work as written. `Sweeper.__reduce__` pickles by reference (module
plus qualname), added so a decorator-rebound module-level function stays
reachable by a worker. For a `SweepModule` instance, the wrapped callable is
a bound method (`self.forward`), whose `__qualname__` is class-qualified
(`"RhoAtm.forward"`) with no trace of the instance. The by-reference lookup
therefore resolves to the unbound function on the class, which is not a
`Sweeper`, so every worker in the pool raised at the `isinstance` check and
the whole pool died with `BrokenProcessPool`.

This is exactly the combination the class facade exists for: `__init__`-
built state (an engine handle, a loaded table) has nowhere clean to live
with the bare decorator, so it is the natural candidate for
`executor="process"`. The bug made that combination impossible, silently
promised by the README's own example, with nothing in `test_module.py`
covering `executor="process"` to catch it.

Checked before fixing: a bound method already pickles correctly on its own,
through its instance, as long as the class is importable by qualname and
the instance state is picklable. The by-reference lookup was solving a
problem bound methods do not have. Fixed by making `__reduce__` conditional
on `hasattr(self.func, "__self__")`: a bound method now reconstructs through
the constructor (contract, the already-unpickled bound method, policy,
version, name), and a plain function keeps the by-reference lookup.

### 12. A multi-output tuple return could swap two values undetected (2026-07-30)

`normalise_return` accepted a bare tuple for a multi-output contract,
matched to `contract.outputs` by position. An `xr.Dataset` return was
already checked by name, so a Dataset return could never silently swap
two outputs, but a tuple could: reordering two return values without
touching the contract passed every check and produced a result with the
values under the wrong names.

Reviewing the API surface for this reason turned up that the safe path
(name-checked) already existed for `Dataset`, just not for anything
lighter, so `forward` had to either accept the position-only risk or take
on an xarray dependency just to return two values safely. A plain dict
keyed by output name is now accepted alongside `Dataset`, validated the
same way (missing name is a clear `ContractError`), and the tuple path is
removed outright for a multi-output contract rather than kept as a
still-available foot-gun. Single-output contracts are unaffected: there is
nothing to swap with one value.

## Measurements that corrected the specification

### The per-point cost was five times the estimate

The specification assumed roughly one millisecond per loop point. Measured:
ten. The surprise was where it sat: the store-less mode cost the same, so it
was not the filesystem but `Dataset.to_zarr(region=...)`, which revalidates
and re-encodes the whole dataset on every call.

Writing straight into the zarr arrays, and dropping the per-point
`expand_dims` round trip, brought it to about five milliseconds. The spec
assumption was corrected rather than quietly left standing.

At the target scale this is still comfortable: 1e4 points cost about a minute
of bookkeeping against hours of engine time, so SC-009 holds.

### Deduplication initially gained nothing at all

On a 3600-pixel map with 25 unique rows and a free callee:

| | wall time | calls | writes |
|---|---|---|---|
| without dedup | 17.5 s | 3600 | 3600 |
| with dedup | 17.6 s | 25 | 3600 |

Calls divided by 144, wall time divided by exactly one. Duplicated points
were being written one region each, and those writes cost everything the
deduplication had saved.

Duplicated points are now filled in a single pass at the end, in slabs sized
from a memory budget rather than a point count, because one expanded row is
as wide as the output's call dims: a scalar output makes rows of eight bytes,
a hundred-channel spectrum makes them a hundred times bigger. A fixed point
count would swing the memory cost by two orders of magnitude between
contracts.

The pass is unconditional, because a run that resumes with every
representative already computed still has duplicates to fill.

After the change, on the same map: 8.1 s, so 2.1x. Better, and still not the
144x the call count suggests.

### The remaining floor is the chunk grid, and it scales with pixels

| map | time | per pixel |
|---|---|---|
| 900 px | 2.1 s | 2.36 ms |
| 3600 px | 8.1 s | 2.24 ms |
| 14400 px | 31.1 s | 2.16 ms |

Linear in PIXELS, flat per pixel, independent of how many unique rows there
are. Extrapolating: 1e5 pixels take four minutes, 1e6 take thirty-seven, a
full Sentinel-2 tile would take three days.

The cause is the store chunk grid: one chunk per loop point. The stated
reason was that this is what makes concurrent region writes safe with no
coordination. A million pixels means a million chunk files, and filling them
costs a million writes whatever deduplication saved on calls.

This does not show with a real engine below about 1e5 points: the same 3600
pixel map with a ten-second callee takes ten hours without dedup and four
minutes with it. Beyond that scale it becomes visible even with an expensive
callee.

### The chunk-per-point justification did not hold in the implementation, and coarser chunks are now the default (2026-07-30)

The "safe concurrent writers" reasoning above was checked against the actual
execution loop and found false: `_run` streams outcomes back from the
executor and writes every one of them in the parent process. There is
exactly one writer, always, so a chunk grid sized for concurrent writers was
buying nothing. The real reason chunk-per-point "worked" is unrelated to
concurrency: a write into a size-1 chunk is one independent write, while a
write into a shared chunk is a read-modify-write of the whole chunk, and
that cost was paid once per point instead of once per chunk.

The fix implemented: the store's loop-dim chunk grid is now sized from a
memory budget (64 MB by default), not pinned to one point per chunk, and
overridable per dim with `SweepPolicy(loop_chunks=...)`. Execution buffers
outcomes per chunk, reading it once, placing every point it produces, and
writing it back once when every runnable point that chunk owns has an
outcome, so a wider grid amortises the read-modify-write across a whole
chunk instead of paying it per point. A one-point chunk (the default's
floor when a row is wide relative to the budget, or an explicit
`loop_chunks` override of 1) skips the read entirely and writes straight
through, matching the old cost exactly: nothing else can share that chunk,
so there is nothing to preserve.

The first measurement covered only the data arrays and looked like enough:
a 200x200 Cartesian sweep (40,000 points, a near-free callee) went from 69 s
(1.7 ms/point) at chunk-per-point to 39 s with the wider grid. Isolating the
status writes (stubbed to a no-op) showed the data path alone had actually
dropped to 4.2 s; **status writes, still unbuffered on the reasoning that
one byte is too cheap to bother with, accounted for the other 35 s.** A
zarr write's cost is dominated by its fixed per-call overhead (encoding,
storage put, journaling), not by payload size, so a 1-byte write costs
about as much as a several-KB one. Status is now buffered and flushed the
same way as the data, sharing the same chunk grid. Final measurement: **3.8
s, 0.09 ms/point, about 18x** over the chunk-per-point baseline.

`Store.expand`'s duplicate-filling pass (the 144x-calls-but-1.0x-wall-time
case above) needed no change at all: it already slabs along the first loop
dim from a memory budget, and a slab spanning several store chunks is only
cheap when the chunk grid itself is coarse. It now inherits the wider grid
automatically.

A related but distinct gap surfaced while discussing the fix: `_stream`
collected every outcome into a list before `_run` wrote any of them, so
`on_error="raise"` losing a point mid-run discarded every success computed
earlier in that same run, not just the failing one. `_stream` is now a
generator, and `_run` writes (or buffers) each outcome as it arrives; a
raise now flushes whatever is buffered before propagating, so nothing
already computed is lost, matching what the failure/resume story already
promised for separate runs but did not actually hold for a single one.

This closes the v1 candidate previously recorded here (coarser loop-dim
chunks, buffered per chunk); see [limitations](limitations.md) for the
current numbers and the remaining trade (more computed-but-unflushed points
sit in memory before a chunk completes, so a hard kill mid-chunk recomputes
more, though it never loses a persisted result).
