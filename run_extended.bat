@echo off
setlocal
if exist .venv\Scripts\python.exe (
  set PY=.venv\Scripts\python.exe
) else (
  set PY=python
)
%PY% scripts\run_extended_experiments.py --mode full --device auto
if errorlevel 1 exit /b %errorlevel%
echo.
echo Extended experiments completed.
echo Run: %PY% scripts\package_extended_results.py
