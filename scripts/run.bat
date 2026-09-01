@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

rem xhs-rag launcher for Windows
rem Usage: run.bat doctor | login | check [--offline]

set "ROOT=%~dp0.."
set "PYTHONPATH=%ROOT%\src"

rem Prefer the dedicated venv; fall back to whatever python is on PATH.
set "PY=C:\Users\Administrator\.workbuddy\binaries\python\envs\xhs-rag\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" -m xhs_rag.cli %*

endlocal
