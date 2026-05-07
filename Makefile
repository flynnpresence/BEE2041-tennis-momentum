# ══════════════════════════════════════════════════════════════════════
# BEE2041 Tennis Momentum Pipeline
# Usage:
#   make all   - builds entire project, skips unchanged steps
#   make reset - wipes all generated files for a clean rebuild
#   make help  - shows available targets
# ══════════════════════════════════════════════════════════════════════

PYTHON = python3

.PHONY: all reset help

# ─── Help ─────────────────────────────────────────────────────────────
help:
	@echo "Targets:"
	@echo "  make all    - build entire project, skipping unchanged steps"
	@echo "  make reset  - wipe all generated files for a clean rebuild"
	@echo "  make help   - show this message"

# ─── Master target ────────────────────────────────────────────────────
all: blog.html

# ─── Step 1: Download raw data ────────────────────────────────────────
# download.py fetches multiple raw files; stamp tracks overall completion
.download.stamp: scripts/download.py
	$(PYTHON) scripts/download.py
	touch .download.stamp

# ─── Step 2: Clean and merge ──────────────────────────────────────────
# clean.py writes both atp_cleaned_points.csv and wta_cleaned_points.csv;
# stamp tracks overall completion rather than a single output file
.clean.stamp: scripts/clean.py .download.stamp
	$(PYTHON) scripts/clean.py
	touch .clean.stamp

# ─── Step 3: Feature engineering ─────────────────────────────────────
data/processed/processed_features.csv: scripts/features.py .clean.stamp
	$(PYTHON) scripts/features.py

# ─── Step 4: Modelling and outputs ───────────────────────────────────
outputs/ate_results.csv: scripts/model.py \
    data/processed/processed_features.csv
	$(PYTHON) scripts/model.py

# ─── Step 5: Blog data generation ────────────────────────────────────
blog_data.js: scripts/build_blog_data.py outputs/ate_results.csv
	$(PYTHON) scripts/build_blog_data.py

# ─── Step 6: Quarto render ───────────────────────────────────────────
blog.html: blog.qmd blog_data.js blog.js styles.css
	quarto render blog.qmd --to html

# ─── Reset ───────────────────────────────────────────────────────────
reset:
	rm -f data/processed/*.csv outputs/*.html outputs/*.csv \
	      outputs/*.png blog_data.js blog.html \
	      .download.stamp .clean.stamp
	rm -rf data/raw/* blog_files/
	touch data/raw/.gitkeep data/processed/.gitkeep
	@echo "Pipeline reset: run make all to rebuild"
