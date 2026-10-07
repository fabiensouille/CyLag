#!/usr/bin/env python3

import sys
import os

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CYLAG'
copyright = '2023, Fabien Souillé'
author = 'fabien souille'
release = '1.0'

# -- Import test -------------------------------------------------------------
try:
    import cylag
except ImportError:
    raise RuntimeError('Cannot import cylag, please investigate')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.imgmath",
    "sphinx_gallery.gen_gallery",
]

templates_path = ['_templates']
source_suffix = ['.rst'] #['.rst', '.md']
exclude_patterns = ['_build']
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# 'classic', 'alabaster', 'sphinxdoc', 'scrolls', 'nature', 'bizstyle'
html_theme = 'classic'
html_theme_options = {
    "sidebarwidth": '25%',
    "body_min_width": None,
    "body_max_width": 900,
    "globaltoc_collapse": True,
    "globaltoc_maxdepth": 2,
    # Body
    "bodyfont": "Segoe UI, Arial, sans-serif",
    "headbgcolor": "#DCEAF7",
    "headtextcolor": "#1F3A56",
    # Sidebar
    "sidebarbgcolor": "#F0F0F0",
    "sidebartextcolor": "#333333",
    "sidebarlinkcolor": "#2F5D8A",
    "sidebarbtncolor": "#D8D8D8",
    # Relation bars (top and bottom navigation)
    "relbarbgcolor": "#7F9DB9",
    "relbartextcolor": "#FFFFFF",
    "relbarlinkcolor": "#FFFFFF",
    # Footer
    "footerbgcolor": "#DCEAF7",
    "footertextcolor": "#4C5C68",
    # Code blocks
    "codebgcolor": "#EAF7EA",
    "codetextcolor": "#1F2D1F",
}

# option for side bar toc
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}

# logo
html_logo = "_static/logo/logo_cylag_sb.png"

# path for custom
html_static_path = ['_static']

# custom css parameters
html_css_files = [
  'custom.css',
]

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# hide show source opt
html_show_sourcelink = False


# -- Gallery ---------------------------------------------------
# https://sphinx-gallery.github.io/stable/advanced.html#example-2-detecting-image-files-on-disk

sphinx_gallery_conf = {
    "filename_pattern": r"vnv_.*\.py$",
    #"ignore_pattern": r"_.*\.py$",
    # Location of examples
    "examples_dirs": "../../examples",
    # Gallery output
    "gallery_dirs": "auto_examples",
    # Do not execute scripts
    "plot_gallery": True,
    # Optional: don't create backreferences
    "backreferences_dir": None,
    # Sort subsections by README order
    "subsection_order": None,
    # Remove download links if scripts are not intended to run
    "download_all_examples": False,
    # Reset matplotlib after each example exec
    "reset_modules": ("matplotlib"),
    # image scrapper (for "plot_gallery": False)
    #"image_scrapers": ('matplotlib', 'cylag._scraper.png_scraper'),
}

# -- API ---------------------------------------------------

napoleon_numpy_docstring = True
napoleon_include_special_with_doc = True

# to generate api manually: sphinx-apidoc -o docs/source/api cylag
# automatic generation:
def run_apidoc(app):    
    os.system("sphinx-apidoc -f -o source/api ../cylag")

# -- APP ---------------------------------------------------

def setup(app):
    app.connect("builder-inited", run_apidoc)


# -- Options for latex output -------------------------------------------------
latex_documents = [
    (
        "index_pdf",
        "manual.tex",
        "CyLag user manual",
        "Fabien Souillé",
        "manual",
    ),
]
