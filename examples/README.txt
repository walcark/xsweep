Examples
========

Read these in order. Each page adds exactly one idea to the one before it,
and they all drive the same engine: a small Monte-Carlo photon transport
solver, plus a two-stream doubling solver for the spectral pages. Both live
in ``_solvers.py`` next to these scripts, and both are deliberately hard to
vectorise, because a sweep library has nothing to offer a closed form.

Every script is runnable on its own::

    pixi run -e dev python examples/01_why_a_sweep_library.py

The physics is textbook and its sources are cited in ``_solvers.py``; the
limits that have to hold are asserted in
``tests/integration/test_example_solvers.py``. Nothing here is calibrated
against a real instrument, and the photon counts are kept small so that this
site can be built in a couple of minutes. A production engine costs seconds
to minutes per point, which is the regime these pages are really about.
