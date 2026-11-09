@echo off
REM ============================================================================
REM  Yoyo Clip (facepipe) - Windows launcher
REM  Double-click this file. First run creates .venv and installs dependencies;
REM  later runs start the server in a few seconds and open your browser.
REM ============================================================================
setlocal
title Yoyo Clip
cd /d "%~dp0"

REM UTF-8 everywhere so JSON/ffmpeg output never trips over the console code page
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not defined FACEPIPE_PORT set "FACEPIPE_PORT=8765"

echo.
echo  ===============================================
echo   Yoyo Clip - AI video clipping
echo  ===============================================
echo.

REM ---------------------------------------------------------------------------
REM 1. Find Python 3.10+  (py launcher first, then python on PATH)
REM ---------------------------------------------------------------------------
set "PY="
where py >nul 2>&1
if errorlevel 1 goto :try_python
py -3 -c "import sys; assert sys.hexversion >= 0x030A0000" >nul 2>&1
if errorlevel 1 goto :try_python
set "PY=py -3"
goto :have_python

:try_python
where python >nul 2>&1
if errorlevel 1 goto :no_python
REM The Microsoft Store "python.exe" stub exits non-zero and opens the Store;
REM a real interpreter runs the version check.
python -c "import sys; assert sys.hexversion >= 0x030A0000" >nul 2>&1
if errorlevel 1 goto :no_python
set "PY=python"
goto :have_python

:no_python
echo  Python 3.10 or newer was not found on this computer.
echo.
echo  1. Install Python 3.12 from https://www.python.org/downloads/windows/
echo  2. In the installer tick  "Add python.exe to PATH"
echo  3. Run start.bat again.
echo.
pause
exit /b 1

:have_python
for /f "tokens=2" %%v in ('%PY% --version 2^>^&1') do set "PYVER=%%v"
echo  Using Python %PYVER%  (%PY%)

REM ---------------------------------------------------------------------------
REM 2. Virtual environment + dependencies (first run only)
REM ---------------------------------------------------------------------------
if exist ".venv\Scripts\python.exe" goto :have_venv

echo  First run: creating virtual environment in .venv ...
%PY% -m venv .venv
if errorlevel 1 goto :venv_failed
echo  Installing dependencies - this takes a few minutes the first time ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e ".[dev,transcribe]"
if errorlevel 1 goto :install_failed
echo  Installation complete.

:have_venv
REM Re-install if the package or a dependency is missing (e.g. after a git pull)
".venv\Scripts\python.exe" -c "import facepipe.web.app" >nul 2>&1
if not errorlevel 1 goto :run
echo  Updating dependencies ...
".venv\Scripts\python.exe" -m pip install -e ".[dev,transcribe]"
if errorlevel 1 goto :install_failed

:run
REM ---------------------------------------------------------------------------
REM 3. Start the server. It opens http://127.0.0.1:%FACEPIPE_PORT%/ in the default browser.
REM ---------------------------------------------------------------------------
echo.
echo  Starting Yoyo Clip on http://127.0.0.1:%FACEPIPE_PORT%/
echo  Keep this window open while you use the app. Close it or press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" -m facepipe.web --port %FACEPIPE_PORT% %*
set "RC=%errorlevel%"
echo.
if "%RC%"=="0" goto :stopped
echo  Yoyo Clip stopped with an error, exit code %RC%. See the messages above.
echo  Common cause: port %FACEPIPE_PORT% is already in use. Run:  set FACEPIPE_PORT=9000  and then start.bat
pause
exit /b %RC%

:stopped
echo  Yoyo Clip stopped.
pause
exit /b 0

:venv_failed
echo.
echo  Could not create the virtual environment.
echo  If this folder path is very long, move the project to a short path such as C:\yoyoclip
pause
exit /b 1

:install_failed
echo.
echo  Dependency installation failed. Check your internet connection and the messages above,
echo  then run start.bat again - it will retry the installation.
pause
exit /b 1
