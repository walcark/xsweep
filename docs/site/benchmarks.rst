Benchmark evolution across versions
=====================================

Each case in the gallery is timed, and the measurement is appended to
``benchmarks/results/history.jsonl`` on every deliberate
``pixi run -e dev bench`` (not on every docs build; see
`benchmarks/README.md <https://github.com/walcark/xsweep/blob/main/benchmarks/README.md>`_).
The same ledger is also rendered as a plain table in
`TIMING.md <https://github.com/walcark/xsweep/blob/main/benchmarks/results/TIMING.md>`_.

Pick a case below to see how its wall-clock time has moved across
``xsweep`` versions. Times are only meaningful compared within the same
host (shown on hover); they say nothing across different machines.

.. raw:: html

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
