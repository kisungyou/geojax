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
DOCS_DIR ?= docs
SITE_DIR ?= site

.PHONY: help install website serve clean

help:
	@echo "GeoJAX targets"
	@echo "  make install   Build a wheel and install it in the current environment"
	@echo "  make website   Execute tutorials and build the documentation website"
	@echo "  make serve     Serve the built website at http://127.0.0.1:8000"
	@echo "  make clean     Remove build artifacts"
	@echo ""
	@echo "Variables"
	@echo "  PYTHON=$(PYTHON)"

install:
	$(PIP) install --upgrade build
	rm -rf build $(DIST_DIR) *.egg-info
	$(PYTHON) -m build --wheel
	$(PIP) install --force-reinstall $(DIST_DIR)/$(PACKAGE)-*.whl

website:
	$(PYTHON) -m sphinx -W --keep-going -b html $(DOCS_DIR) $(SITE_DIR)
	cp $(DOCS_DIR)/_static/brand/geojax-gj-mark.png $(SITE_DIR)/_static/geojax-gj-mark.png
	cp $(DOCS_DIR)/_static/brand/geojax-gj-mark-dark.png $(SITE_DIR)/_static/geojax-gj-mark-dark.png
	cp $(DOCS_DIR)/_static/brand/geojax-gj-favicon.png $(SITE_DIR)/_static/geojax-gj-favicon.png
	$(PYTHON) $(DOCS_DIR)/audit_html.py $(SITE_DIR)

serve: website
	$(PYTHON) -m http.server 8000 --directory $(SITE_DIR)

clean:
	rm -rf build $(DIST_DIR) $(SITE_DIR) .jupyter_cache *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
