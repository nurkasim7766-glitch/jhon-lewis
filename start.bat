@echo off
setlocal EnableDelayedExpansion
title facepipe
cd /d "%~dp0"

echo.
echo  ===============================================
echo   facepipe - AI video clipping
echo  ===============================================
echo.

REM --- locate Python 3.10+ ---------------------------------------------------
set "PY="
where py >nul 2>&1 && (py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1 && set "PY=py -3")
if not defined PY (
  where python >nul 2>&1 && (python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1 && set "PY=python")
)
if not defined PY (
  echo  Python 3.10 or newer was not found.
  echo  Install it from https://www.python.org/downloads/windows/ and tick "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)

REM --- create the virtual environment on first run ---------------------------
if not exist ".venv\Scripts\python.exe" (
  echo  First run: creating virtual environment ...
  %PY% -m venv .venv || (echo  Could not create .venv & pause & exit /b 1)
  echo  Installing dependencies (this takes a few minutes the first time) ...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -e ".[dev]" || (echo  Dependency installation failed. & pause & exit /b 1)
  echo  Done.
)

REM --- make sure the package is importable (handles git pulls that add deps) --
".venv\Scripts\python.exe" -c "import facepipe.web.app" >nul 2>&1 || ".venv\Scripts\python.exe" -m pip install -e ".[dev]"

REM --- start the server; it opens the browser itself ---------------------------
if not defined FACEPIPE_PORT set "FACEPIPE_PORT=8765"
echo  Starting facepipe on http://127.0.0.1:%FACEPIPE_PORT%/  (close this window to stop)
echo.
".venv\Scripts\python.exe" -m facepipe.web --port %FACEPIPE_PORT% %*
echo.
echo  facepipe stopped.
pause
