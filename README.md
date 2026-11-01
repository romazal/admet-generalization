# ADMET Generalization

This project is a computational implementation scaffold for studying whether graph neural networks outperform fingerprint-based models for ADMET property prediction.

## Project goal

The research question is: when does a graph neural network (GNN) outperform a fingerprint-based model for ADMET property prediction, and does it benefit from being trained on molecules structurally different from the training set?

The project compares:
- Random forest baselines using molecular fingerprints
- GNN-based models using graph representations
- Optional pretraining and transfer strategies
- Generalization under random and scaffold-based data splits

## Repository structure

```text
admet-generalization/
├── pyproject.toml
├── Makefile
├── README.md
├── AGENT_LOG.md
├── .gitignore
├── configs/
│   ├── base.yaml
│   ├── tox21_rf_random.yaml
│   ├── tox21_rf_scaffold.yaml
│   ├── tox21_gnn_random.yaml
│   ├── tox21_gnn_scaffold.yaml
│   └── sweep_learning_curve.yaml
├── src/
│   └── admet/
│       ├── __init__.py
│       ├── cli.py
│       ├── training.py
│       ├── evaluation.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │   └── splits.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── fingerprints.py
│       │   └── graphs.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── rf.py
│       │   └── gnn.py
│       └── analysis/
│           ├── __init__.py
│           ├── aggregate.py
│           └── plots.py
├── tests/
│   ├── test_splits.py
│   ├── test_features.py
│   └── test_evaluation.py
├── figures/
├── results/
└── .venv/
```

## Setup

Requirements:
- Python >= 3.12 (project metadata currently targets 3.14-compatible tooling)
- pip / virtual environment

Install for local development:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

## Planned workflow

1. Load the Tox21 dataset and optionally clean missing labels.
2. Build train / validation / test splits with random and scaffold strategies.
3. Train baseline and GNN models.
4. Evaluate using ROC-AUC and PR-AUC.
5. Compare generalization trends under varying training-set size ratios.
6. Save summary metrics and plots to results/ and figures/.

## Evaluation protocol

Main experiment:
- Train models under random and scaffold splits.
- Evaluate on held-out validation or test sets.
- Compare performance between fingerprint and graph-based models.

Additional generalization experiment:
- Train on 10%, 25%, 50%, and 100% of the dataset.
- Measure how model quality changes with data availability.

Metrics:
- ROC-AUC (primary metric)
- PR-AUC (secondary metric)
- Report means and standard deviations across repeated runs.

## Current status

This repository is a project scaffold aligned with the specification in the assignment document. The required folder structure and task organization are in place, while the actual model implementation, data pipeline, and experiments remain to be completed in subsequent development work.

## Notes

This project is designed as a reproducible research and implementation template. The structure is intentionally modular to support clear separation between data loading, feature engineering, model definitions, training, evaluation, and analysis.
