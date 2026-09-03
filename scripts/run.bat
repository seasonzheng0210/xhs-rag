@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

rem xhs-rag launcher for Windows
rem Usage: run.bat doctor | login | check [--offline] | sync | serve | ask "..."
rem
rem 按优先级找 python：
rem   1. 环境变量 XHS_PY（显式指定，如 set XHS_PY=C:\path\to\python.exe）
rem   2. 仓库内 .venv\Scripts\python.exe （pip install -e ".venv" 的默认位置）
rem   3. 仓库内 venv\Scripts\python.exe
rem   4. PATH 上的 python（需已 pip install -e . 或 PYTHONPATH 生效）

set "ROOT=%~dp0.."
set "PYTHONPATH=%ROOT%\src"

set "PY=%XHS_PY%"
if not defined PY if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"
if not defined PY if exist "%ROOT%\venv\Scripts\python.exe" set "PY=%ROOT%\venv\Scripts\python.exe"
if not defined PY set "PY=python"

"%PY%" -m xhs_rag.cli %*

endlocal
