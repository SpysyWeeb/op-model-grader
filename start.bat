@echo off
rem op-model-grader launcher (Windows).
rem First run creates a local .venv and installs the tool; after that it just starts.
rem No arguments -> opens the desktop UI. Any arguments are passed to the CLI instead.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  set "PY=python"
)
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo Python 3 was not found. Install it from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH" in the installer^), then run this again.
  pause
  exit /b 1
)

.venv\Scripts\python.exe -c "import opgrader" >nul 2>nul
if errorlevel 1 (
  echo First run: setting up, this can take a minute...
  if not exist .venv\Scripts\python.exe %PY% -m venv .venv
  if errorlevel 1 ( echo Could not create the virtual environment. & pause & exit /b 1 )
  .venv\Scripts\python.exe -m pip install --quiet --upgrade pip
  .venv\Scripts\python.exe -m pip install --quiet -e .
  if errorlevel 1 ( echo Install failed - see messages above. & pause & exit /b 1 )
)

if "%~1"=="" (
  .venv\Scripts\python.exe -m opgrader --ui
) else (
  .venv\Scripts\python.exe -m opgrader %*
)
if errorlevel 1 pause
