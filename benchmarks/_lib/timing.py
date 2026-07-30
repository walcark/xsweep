"""Record one benchmark measurement per case, appended to results/history.jsonl.

Each `benchmarks/examples/NN_*.py` script calls `record()` at the end of its
`main()`. The ledger is append-only and machine-dependent (see `host`), so
`benchmarks/run.py` is what turns it into `results/TIMING.md` and the site's
evolution chart, not this module.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import xsweep

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
_HISTORY_PATH = _RESULTS_DIR / "history.jsonl"

# Sphinx-gallery re-executes every example script on each docs build (to
# render its output/figures), which would otherwise append a fresh line to
# the committed ledger on every CI run, under an ephemeral runner hostname.
# The docs workflow sets this so the gallery still renders, but the ledger
# only grows from a deliberate `pixi run -e dev bench`.
_SKIP_RECORD_ENV = "XSWEEP_BENCH_SKIP_RECORD"


@dataclass(frozen=True)
class BenchResult:
    """One timed run of one benchmark case."""

    case: str
    xsweep_version: str
    date: str
    host: str
    cpu_count: int
    n_calls: int
    wall_time_s: float


def record(case: str, *, n_calls: int, wall_time_s: float) -> BenchResult:
    """Build a `BenchResult` for `case` and append it to `results/history.jsonl`."""
    result = BenchResult(
        case=case,
        xsweep_version=xsweep.__version__,
        date=datetime.now(timezone.utc).date().isoformat(),
        host=platform.node(),
        cpu_count=os.cpu_count() or 1,
        n_calls=n_calls,
        wall_time_s=round(wall_time_s, 6),
    )
    if not os.environ.get(_SKIP_RECORD_ENV):
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with _HISTORY_PATH.open("a") as f:
            f.write(json.dumps(asdict(result)) + "\n")
    return result
