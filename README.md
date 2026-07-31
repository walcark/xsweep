# xsweep

<p align="center"><em>Content-agnostic parameter sweeps for xarray: lift an
expensive point function into a gridded, cached, resumable computation.</em></p>

<p align="center">
  <a href="https://github.com/walcark/xsweep/actions/workflows/ci.yml"><img src="https://github.com/walcark/xsweep/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://codecov.io/gh/walcark/xsweep"><img src="https://codecov.io/gh/walcark/xsweep/branch/main/graph/badge.svg"></a>
  <a href="https://walcark.github.io/xsweep/"><img src="https://github.com/walcark/xsweep/actions/workflows/docs.yml/badge.svg"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue">
  <a href="https://pixi.sh"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/mypy-checked-2a6db2"></a>
  <img src="https://img.shields.io/badge/tested%20with-pytest-0a9edc?logo=pytest&logoColor=white">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

**[Documentation and examples](https://walcark.github.io/xsweep/)**

Some functions cannot be vectorised along the dims you want to sweep: a
Monte-Carlo solver, an iterative scheme whose stopping point depends on the
data, an external engine that takes one configuration at a time. numpy has
nothing to offer there, and the loop you write instead quietly grows a cache,
a resume path, a deduplication table and a way to guess how long it will all
take.

xsweep is that loop, written once.

```python
import numpy as np, xarray as xr
from xsweep import sweep, SweepPolicy


@sweep("loop(tau, ssa) -> reflectance()", store="runs/layer.zarr", version="1")
def layer(tau: float, ssa: float) -> float:
    return monte_carlo(tau, ssa, n_photons=int(1e6))  # subprocess, GPU kernel, ...


space = xr.Dataset(
    {"tau": ("tau", np.linspace(0.1, 3.0, 40)), "ssa": ("ssa", [0.9, 0.95, 1.0])}
)

print(layer.explain(space))  # what it will cost, zero calls
result = layer(space)  # reflectance(tau: 40, ssa: 3)
```

Re-running that sweep makes zero calls. Interrupt it and relaunch: only the
missing points are recomputed.

## The three ideas

**Semantics come from xarray, not from a second description.** Variables
sharing a dim vary together; variables on distinct dims multiply. A 1000 x
1000 map is a million zipped points, not a trillion product ones.

**The contract describes the call, not the data.** `loop` is one value per
call, `vec` a whole axis, `const` context handed over unchanged, and the
arrow names what one call produces. The same contract reads a parameter study
and a satellite scene.

**Cache, output and resume are the same artefact.** Results stream into a
zarr store with a `status` sidecar, which gives memoisation, bounded memory,
resumability and safe parallel writes at once. Deduplication, batch sizes and
executors are cost decisions on top, and a release gate enforces that they
never move a value.

## Install

```bash
pip install xsweep
```

Python 3.11+, with xarray, zarr and numpy. Dask is an optional executor
backend and no core path imports it.

To work on xsweep itself: `pixi install && pixi run -e dev all`.

## Documentation

Everything lives on the site: **<https://walcark.github.io/xsweep/>**

- [Why xsweep](https://walcark.github.io/xsweep/why.html), including when it
  is the wrong tool
- [Examples](https://walcark.github.io/xsweep/auto_examples/index.html): ten
  pages, one idea each, on a real Monte-Carlo solver, with the actual output
  of every run
- [Guide](https://walcark.github.io/xsweep/guide/contract.html) and
  [API reference](https://walcark.github.io/xsweep/reference/api.html)
- [Limitations](https://walcark.github.io/xsweep/reference/limitations.html):
  what v0 deliberately does not do
- [Benchmarks](https://walcark.github.io/xsweep/benchmarks.html): per-point
  cost tracked across releases

## Status

v0. Motivating consumers: adjeff (Smart-G sweeps, replacing its internal
`SweepBundle` and `UniqueIndex`), radtrans (the engine-agnostic half of its
sweep layer), and short sensitivity studies in Earth observation.

## License

[Apache License 2.0](LICENSE).
