# v0 limitations

What xsweep v0 deliberately does not do, and where the reasoning lives. Each
entry is stated in exactly one place, so this page points rather than
restates (spec FR-026).

## Scope of the persistent mode

The persistent mode is calibrated for around 1e4 loop points with callees
costing seconds to minutes. One write region per loop point is negligible at
that ratio, and the regime is self-limiting anyway: a million points at ten
seconds each would run for months.

Fast callees are NOT excluded. A fast function is usually vectorisable, so it
belongs in `vec` (one call for a whole axis, not a million loop points), and
the store-less mode removes the per-point write cost entirely. Only the
conjunction is a non-goal: cheap, scalar-only, very many points, AND needing
persistence. Serving it would need coalesced regions, which would coarsen
resume granularity and the status variable.

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
- **No automatic per-version namespacing.** See [idioms](idioms.md).

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
  [idioms](idioms.md) for the label workaround.
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
