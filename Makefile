# One-command shortcuts for the workflow (proposal §5).
#
# These document the order of operations as much as they save typing: the
# same steps can always be run by hand with `python -m admet.cli ...`.
#
# Override the interpreter if you are not using the project venv:
#     make test PYTHON=python3

PYTHON ?= .venv/Scripts/python.exe
ADMET  := $(PYTHON) -m admet.cli

.DEFAULT_GOAL := help

.PHONY: help install data test main sweep clintox aggregate plots all clean clean-results

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and its dependencies in editable mode
	$(PYTHON) -m pip install -e ".[dev]"

data:  ## Download and cache both datasets (also happens on first run)
	$(PYTHON) -c "from admet.data.loaders import load_dataset; \
		[print(load_dataset(n).summary()) for n in ('tox21', 'clintox')]"

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

main:  ## Main experiment: both models x both splits, all seeds (proposal §3.3)
	$(ADMET) run configs/tox21_rf_random.yaml
	$(ADMET) run configs/tox21_rf_scaffold.yaml
	$(ADMET) run configs/tox21_gnn_random.yaml
	$(ADMET) run configs/tox21_gnn_scaffold.yaml

sweep:  ## Additional experiment: the learning-curve grid
	$(ADMET) sweep configs/sweep_learning_curve.yaml

clintox:  ## Second dataset: the same grid on ClinTox (proposal §3.1)
	$(ADMET) run configs/clintox_rf_random.yaml    --train-fractions 0.1 0.25 0.5 1.0
	$(ADMET) run configs/clintox_rf_scaffold.yaml  --train-fractions 0.1 0.25 0.5 1.0
	$(ADMET) run configs/clintox_gnn_random.yaml   --train-fractions 0.1 0.25 0.5 1.0
	$(ADMET) run configs/clintox_gnn_scaffold.yaml --train-fractions 0.1 0.25 0.5 1.0

aggregate:  ## Summarize results/ into runs.csv, summary.csv, gap.csv and per_task.csv
	$(ADMET) aggregate

plots:  ## Redraw every figure from the aggregated tables
	$(ADMET) plots

all: test main sweep clintox aggregate plots  ## Everything, in order

clean:  ## Remove caches and build artefacts (keeps results and figures)
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info

clean-results:  ## Delete every run result and figure. Destructive.
	rm -f results/*.json results/*.csv figures/*.png
