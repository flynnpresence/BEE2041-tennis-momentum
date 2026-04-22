# BEE2041 Tennis Momentum Pipeline
# Replicates the full analysis from raw data to outputs
# Usage: make all
PYTHON = /usr/local/bin/python3
.PHONY: all download clean features model
all: download clean features model
download:
	$(PYTHON) scripts/download.py
clean: download
	$(PYTHON) scripts/clean.py
features: clean
	$(PYTHON) scripts/features.py
model: features
	$(PYTHON) scripts/model.py
