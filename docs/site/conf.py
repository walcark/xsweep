"""Sphinx configuration for the xsweep example gallery.

Build with::

    pixi run -e docs docs-build

This builds only the benchmark/example gallery (`benchmarks/examples/`), not
the hand-written guides under `docs/*.md`, which stay plain markdown read
directly on GitHub.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import xsweep

project = "xsweep"
copyright = "2026, Kévin Walcarius"
author = "Kévin Walcarius"
release = xsweep.__version__

extensions = [
    "sphinx_gallery.gen_gallery",
]

sphinx_gallery_conf = {
    "examples_dirs": "../../benchmarks/examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/\d+_",
    "download_all_examples": False,
    "remove_config_comments": True,
}

templates_path: list[str] = []
exclude_patterns: list[str] = ["_build"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "xsweep examples"
html_theme_options = {
    "github_url": "https://github.com/walcark/xsweep",
    "navbar_end": ["navbar-icon-links"],
}
