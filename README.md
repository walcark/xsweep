# xsweep

Minimal, content-agnostic parameter sweeps for xarray: lift an expensive
"point function" into a gridded, cached, resumable xarray computation.

One core mechanism, three surfaces: a `@sweep` decorator, a pytorch-style
`SweepModule` base class, and a `Contract` object describing what one call
consumes and produces.

Status: design phase. No code yet. The full design, its rationale, worked
examples and known limits live in `docs/design/xsweep.md`.

Motivating consumers: adjeff (Smart-G sweeps, replaces its internal
`SweepBundle`/`UniqueIndex`), radtrans (the engine-agnostic half of its
planned sweep layer), and short sensitivity studies in Earth observation.
