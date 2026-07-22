# Documentation conventions

Public objects use NumPy-style docstrings with a one-line summary, parameters,
returns, mathematical notes, shape conventions, numerical limitations, and a
minimal example.

Tutorials are MyST Markdown notebooks. They must be deterministic, import only
the public GeoJAX API, include the mathematics needed to understand the task,
and finish with numerical diagnostics plus at least one informative figure.

The site build executes every tutorial and treats execution warnings as build
failures:

```bash
make website
```

The build executes every tutorial with `myst-nb`, treats Sphinx warnings as
errors, and then audits the rendered HTML for malformed math nodes, leaked TeX,
and broken local links, fragments, scripts, styles, and images.
