# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

project = 'Lunar-Route-Planner'
copyright = '2026, Elvin Hadi'
author = 'Elvin Hadi'
release = '23-10-2026'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",       # pulls docstrings from code
    "sphinx.ext.napoleon",      # understands NumPy/Google docstrings
    "sphinx.ext.viewcode",      # adds [source] links
    "sphinx.ext.intersphinx",   # links to numpy/scipy docs (optional)
]
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
