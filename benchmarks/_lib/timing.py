"""Record one benchmark measurement, appended to results/history.jsonl.

Each `benchmarks/cases/NN_*.py` script calls `record()` once per measurement
it makes. The ledger is append-only and machine-dependent (see `host`), so
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


@dataclass(frozen=True)
class BenchResult:
    """One timed run of one benchmark case."""

    case: str
    xsweep_version: str
    date: str
    host: str
    cpu_count: int
    n_points: int
    n_calls: int
    wall_time_s: float


def record(
    case: str, *, n_points: int, n_calls: int, wall_time_s: float
) -> BenchResult:
    """Build a `BenchResult` for `case` and append it to `results/history.jsonl`.

    Parameters
    ----------
    case
        Identifier of the measurement, unique across the whole suite.
    n_points
        Points in the sweep, which is what the per-point cost divides by.
    n_calls
        Calls the run actually made, which differs from `n_points` whenever
        deduplication or a cache was involved.
    wall_time_s
        Wall-clock seconds the measured section took.

    Returns
    -------
    BenchResult
        The record that was appended.
    """
    result = BenchResult(
        case=case,
        xsweep_version=xsweep.__version__,
        date=datetime.now(timezone.utc).date().isoformat(),
        host=platform.node(),
        cpu_count=os.cpu_count() or 1,
        n_points=n_points,
        n_calls=n_calls,
        wall_time_s=round(wall_time_s, 6),
    )
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with _HISTORY_PATH.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")
    print(
        f"{case}: {n_points} points, {n_calls} calls, "
        f"{wall_time_s:.3f} s ({wall_time_s / n_points * 1e3:.4f} ms/point)"
    )
    return result
