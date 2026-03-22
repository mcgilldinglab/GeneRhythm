from pathlib import Path

project = "GeneRhythm"
author = "GeneRhythm contributors"
copyright = "2026, GeneRhythm contributors"

extensions = [
    "myst_parser",
    "nbsphinx",
    "nbsphinx_link",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

nbsphinx_execute = "never"
nbsphinx_allow_errors = False

html_theme = "sphinx_rtd_theme"
html_title = "GeneRhythm Docs"

html_context = {
    "display_github": False,
}

ROOT = Path(__file__).resolve().parents[1]
