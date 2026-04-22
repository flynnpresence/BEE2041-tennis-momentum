# BEE2041 Tennis Momentum Pipeline
# Replicates the full analysis from raw data to outputs
# Usage: make all
PYTHON = /usr/local/bin/python3
.PHONY: all download clean features model blog_data
all: download clean features model blog_data
download:
	$(PYTHON) scripts/download.py
clean: download
	$(PYTHON) scripts/clean.py
features: clean
	$(PYTHON) scripts/features.py
model: features
	$(PYTHON) scripts/model.py
blog_data: model
	$(PYTHON) scripts/build_blog_data.py
