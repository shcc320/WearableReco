@echo off
setlocal
python scripts\download_datasets.py
if errorlevel 1 exit /b 1
python scripts\run_pipeline.py --mode full --device auto %*
endlocal
