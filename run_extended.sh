#!/usr/bin/env bash
set -euo pipefail
PY=python
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi
"$PY" scripts/run_extended_experiments.py --mode full --device auto
printf '\nExtended experiments completed.\nRun: %s scripts/package_extended_results.py\n' "$PY"
