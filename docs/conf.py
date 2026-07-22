"""Sphinx configuration for the GeoJAX documentation."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "GeoJAX"
author = "GeoJAX contributors"
release = "0.1.0"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = True

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3

nb_execution_mode = "force"
nb_execution_timeout = 180
nb_execution_raise_on_error = True
nb_merge_streams = True

html_theme = "pydata_sphinx_theme"
html_title = "GeoJAX"
html_baseurl = "https://www.kisungyou.com/geojax/"
html_favicon = "_static/brand/geojax-gj-favicon.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "logo": {
        "text": "Geo",
        "image_light": "_static/brand/geojax-gj-mark.png",
        "image_dark": "_static/brand/geojax-gj-mark-dark.png",
        "alt_text": "GeoJAX home",
    },
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/kisungyou/geojax",
            "icon": "fa-brands fa-github",
        }
    ],
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
html_sidebars = {
    "index": [],
    "getting_started/index": [],
    "**": ["search-field", "sidebar-nav-bs"],
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
