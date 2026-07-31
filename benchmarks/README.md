# Benchmarks

These are not the examples. The [example gallery](../examples) exists to show
what xsweep is for, so its engine is deliberately expensive and its point
counts are small enough to render a website. These cases want the opposite:
**a callee as close to free as Python allows, over as many points as is
practical**, so that what gets timed is xsweep's own bookkeeping rather than
somebody's physics.

That is the only way a number stays comparable from one release to the next.
If the engine dominated, the measurement would tell you about the engine.

| Case file | What it isolates |
|---|---|
| `cases/01_in_memory_grid.py` | planning, delivery and result assembly, with no store |
| `cases/02_store_write.py` | the store's write path, and a fully cached second pass |
| `cases/03_dedup_expand.py` | unique-row extraction and expansion, 90000 pixels over 60 rows |
| `cases/04_vec_batches.py` | the work-item path when one point becomes 100 batched calls |
| `cases/05_resume.py` | a partly failed store, and the resume pass that repairs it |

## Running them

```bash
pixi run -e dev bench
```

This runs every case, appends its measurements to `results/history.jsonl`,
and regenerates two derived files from that ledger:

- `results/TIMING.md`, the current numbers plus a per-case history;
- `docs/_static/benchmarks_history.json`, which feeds the evolution chart on
  the site.

Both are generated. Edit the cases, never the reports.

Nothing runs these automatically. They are meant for a deliberate run, at a
release or when a change is suspected of costing something, not on every
push: a shared CI runner's timings vary by more than the effects worth
measuring.

## Reading the numbers

Every record carries `host` and `cpu_count`, because **wall-clock times only
mean something compared against other runs on the same machine**. A row from
a laptop and a row from a workstation are two different experiments that
happen to share a case name.

`ms/point` is the number to watch. Absolute wall time moves when a case is
resized; the per-point cost is what a release is allowed to be judged on, and
even then only against the same host.
