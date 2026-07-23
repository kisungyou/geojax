# GeoJAX Makefile
#
# Main workflow:
#   make install   Build a wheel and install it into the current Python env.
#   make website   Execute tutorials and build the documentation website.
#
# Tunable variables:
#   make install PYTHON=python3

PYTHON ?= python
PIP := $(PYTHON) -m pip
PACKAGE ?= geojax
DIST_DIR ?= dist
VERSION ?= $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
RELEASE_DIST_DIR ?= $(DIST_DIR)/$(VERSION)
DOCS_DIR ?= docs
SITE_DIR ?= site
JUPYTER_EXECUTE_DIR ?= jupyter_execute
JUPYTER_CACHE_DIR ?= .jupyter_cache
SPHINXOPTS ?= -E -a
PYTESTOPTS ?= --cov=geojax --cov-report=term-missing

.PHONY: help install test test-float32 test-matrix website serve release-check clean

help:
	@echo "GeoJAX targets"
	@echo "  make install   Build a wheel and install it in the current environment"
	@echo "  make test      Run the full float64 test suite with coverage"
	@echo "  make test-float32"
	@echo "                 Run the full float32 test suite with coverage"
	@echo "  make test-matrix"
	@echo "                 Run the supported Python/JAX/precision matrix with tox"
	@echo "  make website   Execute tutorials and build the documentation website"
	@echo "  make serve     Serve the built website at http://127.0.0.1:8000"
	@echo "  make release-check"
	@echo "                 Run the complete matrix, docs, and package checks"
	@echo "  make clean     Remove build artifacts"
	@echo ""
	@echo "Variables"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  VERSION=$(VERSION)"
	@echo "  PYTESTOPTS=$(PYTESTOPTS)"
	@echo "  SPHINXOPTS=$(SPHINXOPTS)"

install:
	$(PIP) install --upgrade build
	rm -rf build $(DIST_DIR) *.egg-info
	$(PYTHON) -m build --wheel
	$(PIP) install --force-reinstall $(DIST_DIR)/$(PACKAGE)-*.whl

test:
	GEOJAX_TEST_X64=1 $(PYTHON) -m pytest $(PYTESTOPTS)

test-float32:
	GEOJAX_TEST_X64=0 $(PYTHON) -m pytest $(PYTESTOPTS)

test-matrix:
	$(PYTHON) -m tox run

website:
	$(PYTHON) -m sphinx $(SPHINXOPTS) -W --keep-going -b html $(DOCS_DIR) $(SITE_DIR)
	cp $(DOCS_DIR)/_static/brand/geojax-gj-mark.png $(SITE_DIR)/_static/geojax-gj-mark.png
	cp $(DOCS_DIR)/_static/brand/geojax-gj-mark-dark.png $(SITE_DIR)/_static/geojax-gj-mark-dark.png
	cp $(DOCS_DIR)/_static/brand/geojax-gj-favicon.png $(SITE_DIR)/_static/geojax-gj-favicon.png
	$(PYTHON) $(DOCS_DIR)/audit_html.py $(SITE_DIR)
	rm -rf $(JUPYTER_EXECUTE_DIR) $(JUPYTER_CACHE_DIR)

serve: website
	$(PYTHON) -m http.server 8000 --directory $(SITE_DIR)

release-check: test-matrix website
	rm -rf build $(DIST_DIR) *.egg-info
	$(PYTHON) -m build --sdist --wheel --outdir $(RELEASE_DIST_DIR)
	$(PYTHON) -m twine check --strict $(RELEASE_DIST_DIR)/*

clean:
	rm -rf build $(DIST_DIR) $(SITE_DIR) $(JUPYTER_EXECUTE_DIR) $(JUPYTER_CACHE_DIR) *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
