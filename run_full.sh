#!/usr/bin/env bash
set -euo pipefail
python scripts/download_datasets.py
python scripts/run_pipeline.py --mode full --device auto "$@"
