@echo off
title Thesis Deepfake Generator Companion Studio
cd /d "%~dp0"
echo =====================================================================
echo  Launching Thesis Deepfake Generator Companion Studio...
echo =====================================================================

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

%PYTHON_EXE% run_generator.py %*
pause
