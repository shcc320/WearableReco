# Wearable Sports Activity Recognition: Reproducibility Code and Paper Results

This repository provides the code and the compact set of result files supporting the submitted paper on wearable sports-activity recognition. It covers two completed experiments:

1. **Baseline experiment**: LOSO recognition, sensor-configuration analysis, cross-dataset transfer, and deployment ranking.
2. **Extended experiment**: range audit, internal validation, bidirectional transfer, participant-level uncertainty, class-level diagnostics, five-class sensitivity, and MCDA ranking.

The experiments use the six shared activities in PAMAP2 and MHEALTH: lying, sitting, standing, walking, running, and cycling.

## Repository layout

```text
config/                         Baseline and extended experiment settings
scripts/                        Data download and experiment entry points
src/sports_activity_dss/        Data preparation, evaluation, models, transfer, and MCDA code
results/                        Selected baseline results used for the paper
extended_results/               Selected extended results used for the paper
JASE_Final_Table_Uniform/       Submission source, tables, and figures
run_full.bat / run_full.sh      Baseline experiment launcher
run_extended.bat / run_extended.sh
                                Extended experiment launcher
```

## Environment setup

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data preparation

Download PAMAP2 and MHEALTH from their official UCI sources by running:

```bash
python scripts/download_datasets.py
```

The downloader stores the source data under `data/raw/` and the pipelines cache extracted features under `data/processed/`.

## Run the baseline experiment

Windows:

```bat
run_full.bat
```

Linux/macOS:

```bash
bash run_full.sh
```

The baseline pipeline writes its direct outputs to `results/`.

## Run the extended experiment

Windows:

```bat
run_extended.bat
```

Linux/macOS:

```bash
bash run_extended.sh
```

The extended pipeline writes its direct outputs to `extended_results/`.


