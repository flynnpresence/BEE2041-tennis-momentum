# ══════════════════════════════════════════════════════════════════════
# BEE2041 Tennis Momentum Pipeline
# Usage:
#   make all   — builds entire project, skips unchanged steps
#   make reset — wipes all generated files for a clean rebuild
# ══════════════════════════════════════════════════════════════════════

PYTHON = python3

.PHONY: all reset

# ─── Master target ────────────────────────────────────────────────────
all: blog.html

# ─── Step 1: Download raw data ────────────────────────────────────────
data/raw/charting-m-matches.csv: scripts/download.py
	$(PYTHON) scripts/download.py

# ─── Step 2: Clean and merge ──────────────────────────────────────────
data/processed/atp_cleaned_points.csv: scripts/clean.py \
    data/raw/charting-m-matches.csv
	$(PYTHON) scripts/clean.py

# ─── Step 3: Feature engineering ─────────────────────────────────────
data/processed/processed_features.csv: scripts/features.py \
    data/processed/atp_cleaned_points.csv
	$(PYTHON) scripts/features.py

# ─── Step 4: Modelling and outputs ───────────────────────────────────
outputs/ate_results.csv: scripts/model.py \
    data/processed/processed_features.csv
	$(PYTHON) scripts/model.py

# ─── Step 5: Blog data generation ────────────────────────────────────
blog_data.js: scripts/build_blog_data.py outputs/ate_results.csv
	$(PYTHON) scripts/build_blog_data.py

# ─── Step 6: Quarto render ───────────────────────────────────────────
blog.html: blog.qmd blog_data.js
	quarto render blog.qmd --to html

# ─── Reset ───────────────────────────────────────────────────────────
reset:
	rm -f data/processed/*.csv outputs/*.html outputs/*.csv \
	      outputs/*.png blog_data.js blog.html
	rm -rf data/raw/*
	touch data/raw/.gitkeep data/processed/.gitkeep
	@echo "Pipeline reset — run make all to rebuild"
