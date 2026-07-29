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

The cause is the store chunk grid: one chunk per loop point, which is what
makes concurrent region writes safe with no coordination. A million pixels
means a million chunk files, and filling them costs a million writes whatever
deduplication saved on calls.

This does not show with a real engine below about 1e5 points: the same 3600
pixel map with a ten-second callee takes ten hours without dedup and four
minutes with it. Beyond that scale it becomes visible even with an expensive
callee.

**The v1 candidate**: coarser chunks along the loop dims, say 256 by 256 on a
map, which would bring a Sentinel-2 tile from 1.2e8 chunks to about 1800. The
cost is that two workers could then share a chunk during the computation
phase, so writes would need serialising or buffering per chunk. Resume
granularity and the status variable would coarsen with it. That trade is
deliberately out of v0 and is stated in
[limitations](limitations.md).
