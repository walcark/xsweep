# Example scripts

Runnable, self-contained, and ordered so that each one adds a single idea.
Every engine is stubbed with a cheap function, since the library is
content-agnostic and the point is the plumbing.

```bash
pixi run -e dev python scripts/01_first_sweep.py
```

| Script | What it shows |
|--------|---------------|
| `01_first_sweep.py` | Cartesian space, the plan before paying for it, and a second run that makes zero calls |
| `02_pixel_map_dedup.py` | Zipped dims on a pixel map, and what deduplication does and does not buy |
| `03_failure_and_resume.py` | A failing engine, the status variable, resume, and fail-fast |
| `04_band_integration.py` | The `vec` and `const` clauses, and the contract refusing to batch a reduced axis |
| `05_module_and_policy.py` | The class facade, and the three temporalities varying independently |

Each writes to a temporary directory and cleans up after itself, so they can
be run in any order and as many times as you like.
