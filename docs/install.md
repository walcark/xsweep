# Installation

Python 3.11 or later, with xarray, zarr 3 and numpy. Dask is an optional
executor backend and no core path imports it.

```bash
pip install xsweep
```

## Working on xsweep itself

The repository is managed with [pixi](https://pixi.sh):

```bash
git clone https://github.com/walcark/xsweep
cd xsweep
pixi install
pixi run -e dev all      # fmt, lint, type-check, test
```

Other tasks worth knowing:

```bash
pixi run -e dev python examples/01_why_a_sweep_library.py   # run one example
pixi run -e docs docs-build                                 # build this site
pixi run -e dev bench                                       # run the benchmarks
```

Formatting, linting and type-checking are all driven by those tasks; there is
nothing to configure by hand.
