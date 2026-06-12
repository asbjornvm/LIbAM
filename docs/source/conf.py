# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
sys.path.insert(0, os.path.abspath('../../src'))

project = 'LIbAM'
copyright = '2026, asbjørn'
author = 'asbjørn'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Pull docstrings from source code
    'sphinx.ext.napoleon',      # Support Google/NumPy style docstrings
    'sphinx.ext.viewcode',      # Add links to highlighted source code
]

autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_favicon = '_static/icon.svg'
html_theme_options = {
    'logo': 'icon.svg',  # Alabaster's own option (path relative to _static)
    'logo_name': True,   # show the project name (LIbAM) under the logo
}
