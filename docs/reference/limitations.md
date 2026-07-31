# v0 limitations

What xsweep v0 deliberately does not do, and where the reasoning lives. Each
entry is stated in exactly one place, so this page points rather than
restates (spec FR-026).

## Scope of the persistent mode

The persistent mode is calibrated for callees costing seconds to minutes,
where engine time dominates by construction. Fast callees are NOT excluded: a
fast function is usually vectorisable, so it belongs in `vec` (one call for a
whole axis, not a million loop points), and the store-less mode removes
bookkeeping cost entirely. For a cheap, scalar-only, very-many-point sweep
that still needs persistence, execution buffers writes into coarse store
chunks (see below), which keeps bookkeeping a small fraction of a real run
even at large point counts.

## The store chunk grid

Execution has exactly one writer: the parent process collecting outcomes as
they complete. The store's loop-dim chunk grid is therefore sized from a
memory budget (64 MB by default, overridable per dim with
`SweepPolicy(store_chunks=...)`), not pinned to one point per chunk. A size-1
grid turns every point's write into a read-modify-write of that whole chunk;
execution's write buffer reads a chunk once, places every point it produces,
and writes it back once when every runnable point that chunk owns has an
outcome. Status is buffered the same way, since a zarr write's cost is
dominated by its fixed per-call overhead, not by the one byte a status code
carries.

Measured on a 200x200 Cartesian sweep (40,000 points, a near-free callee):
the one-point-per-chunk grid costs about 69 s (1.7 ms/point); the buffered,
memory-budget grid costs about 3.8 s (0.09 ms/point), roughly 18x. With a
real engine the calls still dominate by orders of magnitude either way, so
this only matters when bookkeeping was already competing with engine time.

This also benefits a large deduplicated map for free: `Store.expand` fills
duplicated positions in slabs along the first loop dim, and a slab spanning
several store chunks was exactly as costly as one point per chunk under the
old grid. It now inherits the same coarse grid with no change to `expand`
itself.

One point per chunk is still available, and is what `place()` uses
automatically whenever every loop dim's chunk width is 1 (including an
explicit `store_chunks` override of 1): with nothing else in the chunk to
preserve, reading it first would only add a read no direct write pays for.

A wider chunk means more computed-but-unflushed points sit in memory before
a chunk completes, and more of them are recomputed (not lost, since nothing
un-flushed was ever persisted) if the process is killed hard mid-chunk. A
`PointFailed` raise flushes every open chunk before propagating, so an
`on_error="raise"` sweep never loses a result it already computed, buffered
or not; only an unrecoverable kill can lose a partial chunk's progress, and
even then only what that one chunk was still holding.

## Store and concurrency

- **One store per sweep configuration.** A store carries a fingerprint of
  contract, version and statics, and refuses to open under a different one
  (spec FR-032). Sharing one store between instances is not supported.
- **One writing run at a time**, enforced by a lock (spec FR-036). Readers
  are never blocked. Multi-process cooperation, where cluster array tasks
  claim points atomically, is out of scope: the documented answer is to
  partition the space at submission, one store per task, and merge
  afterwards.
- **No automatic staleness detection.** A dead process on another host cannot
  be probed reliably, so `force_unlock=True` is the honest override.
- **No store reindexing.** Extending an axis and resuming is refused rather
  than silently misaligned (edge case 9).
- **No automatic per-version namespacing.** See [idioms](../guide/idioms.md).

## Batching

- **Batch sizes are explicit** (spec FR-035): the contract's `@ N` shorthand
  for 1-D vec variables, or dim-keyed policy chunks. There is no automatic
  sizing. A deterministic form derived from a declared memory budget is
  designed for later, but it would have to decide once and persist the
  choice, since batch size determines the store chunk grid.
- **No halo or overlap granularity.** A function that is local-with-
  neighbourhood along a dim (a spatial convolution) cannot be batched
  correctly yet; pass the axis whole.

## Data model

- **Sweep values are primitives**: float, int, str, bool, datetime64. See
  [idioms](../guide/idioms.md) for the label workaround.
- **Rectangular axes only.** A wavelength support that depends on the band
  does not fit. The clean workaround is a union grid with zero weights in the
  spectral response.
- **No union output grids.** Calls returning different shapes are detected
  and reported, not padded.
- **Loop variables are assumed in memory.** A lazily-backed one is computed
  with a warning rather than streamed.
- **An output cannot be named after one of its own dims**: in a Dataset such
  a variable IS that dim's coordinate.

## Surfaces

- **No raw per-call Dataset delivery** (spec FR-011). Arguments always come
  named from the contract. Adding the opt-out later is backward compatible.
- **No progress callback** (spec FR-027). Observability is standard-library
  logging; live progress is obtained by reading the status variable from the
  store while the sweep runs.
- **No instance-level output rename.** Designed, and implemented together
  with the pipeline layer, because that is what makes it necessary.
- **No contract cross-check against type annotations.** The redundancy is
  real and usable later as a decoration-time check, never as a source of
  truth.

## Execution

- **Dask is an executor backend, not implemented in v0.** The core never
  imports it. Note that "lazy result" does not mean "dask": without dask, the
  returned handle is lazily indexed by xarray, and dask would only add
  chunked parallel reads.
- **The process executor requires picklable callables**, which in practice
  means functions defined at module level. The sweeper itself pickles by
  reference, so a decorated module-level function works; a function defined
  inside another function does not.
- **No streaming deduplication.** Unique rows are computed in memory.
