"""Sphinx configuration for the xsweep documentation site.

Build with::

    pixi run -e docs docs-build

This builds the whole site: the hand-written guide and reference pages under
``docs/``, plus the example gallery generated from ``examples/`` by
sphinx-gallery, which re-executes every example and captures its output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xsweep

project = "xsweep"
copyright = "2026, Kévin Walcarius"
author = "Kévin Walcarius"
release = xsweep.__version__

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

# `dollarmath` renders the radiative-transfer equations the examples cite;
# `colon_fence` is what sphinx-design directives are written with in markdown.
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# Only files matching `filename_pattern` are executed and turned into pages;
# `_solvers.py` is a shared helper imported by the examples, not an example.
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/\d+_",
    "ignore_pattern": r"/_[^/]*\.py$",
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "remove_config_comments": True,
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

templates_path: list[str] = []
exclude_patterns: list[str] = ["_build", "README.md"]

html_theme = "shibuya"
html_static_path = ["_static"]
html_title = "xsweep"
html_theme_options = {
    "accent_color": "blue",
    "color_mode": "auto",
    "github_url": "https://github.com/walcark/xsweep",
    "nav_links": [
        {"title": "Guide", "url": "guide"},
        {"title": "Examples", "url": "auto_examples/index"},
        {"title": "Reference", "url": "reference/api"},
    ],
}
