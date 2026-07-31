# Benchmark evolution across versions

The gallery examples on this site are kept small and fast, because the site
re-runs every one of them on every build. Timing them would say more about
the runner than about xsweep.

The measurements below come from a separate, deliberately heavier set of
cases under `benchmarks/`, frozen so that the same work is timed release
after release. They are recorded only when
[`pixi run -e dev bench`](https://github.com/walcark/xsweep/blob/main/benchmarks/README.md)
is run on purpose, appended to `benchmarks/results/history.jsonl`, and also
rendered as a plain table in
[TIMING.md](https://github.com/walcark/xsweep/blob/main/benchmarks/results/TIMING.md).

Pick a case to see how its wall-clock time has moved across versions. Times
are only comparable within one host, shown on hover; they say nothing across
different machines.

```{raw} html
<div id="benchmarks-chart">
  <p>
    <label for="benchmarks-case-select">Case:</label>
    <select id="benchmarks-case-select"></select>
  </p>
  <canvas id="benchmarks-canvas" width="760" height="360"
          style="max-width: 100%; height: auto;"></canvas>
  <p id="benchmarks-info" style="font-family: monospace; font-size: 0.9em;"></p>
  <p id="benchmarks-empty" style="display:none;">
    No recorded measurements yet.
  </p>
</div>
<script src="_static/benchmarks.js"></script>
```
