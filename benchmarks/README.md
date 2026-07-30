# Benchmarks

A library of type cases exercising xsweep on synthetic remote-sensing /
radiative-transfer problems, chosen to vary in sweep size and memory profile
and to cover the contract features (`vec`, `const`, batching, dedup, the
`SweepModule` facade, resumability). Where a case implements a real formula,
its docstring cites the source; anything synthetic (an SRF, a lookup table)
is labelled as such rather than presented as instrument data.

Each case lives in `examples/NN_name.py`, follows the same house style as
`scripts/` (see `docs/guide.md`), and is:

- runnable standalone, e.g. `pixi run -e dev python benchmarks/examples/01_beer_lambert_transmission.py`
- rendered as a page in the example gallery (`pixi run -e docs docs-build`)
- timed, appending its measurement to `results/history.jsonl`

Run every case and refresh the reports in one go:

```bash
pixi run -e dev bench
```

which regenerates `results/TIMING.md` (current numbers, plus a per-case
history across `xsweep` versions) and `docs/site/_static/benchmarks_history.json`
(consumed by the site's evolution chart). Wall-clock times are only
meaningful compared across runs on the same machine (`host` on each record);
they say nothing across different hardware.

See `results/TIMING.md` for the current numbers.
