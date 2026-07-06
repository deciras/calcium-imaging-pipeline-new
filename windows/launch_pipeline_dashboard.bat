@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

if not defined CONDA_BIN (
  for %%I in (
    "%USERPROFILE%\anaconda3\Scripts\conda.exe"
    "%USERPROFILE%\miniconda3\Scripts\conda.exe"
    "C:\ProgramData\anaconda3\Scripts\conda.exe"
    "C:\ProgramData\miniconda3\Scripts\conda.exe"
  ) do (
    if exist %%~I (
      set "CONDA_BIN=%%~I"
      goto :conda_found
    )
  )
)

:conda_found
if not defined CONDA_BIN set "CONDA_BIN=conda"
if not defined DASHBOARD_ENV set "DASHBOARD_ENV=dashboard_gui"

where "%CONDA_BIN%" >nul 2>nul
if errorlevel 1 (
  if not exist "%CONDA_BIN%" (
    echo Cannot find conda: %CONDA_BIN%
    echo Set CONDA_BIN=C:\path\to\conda.exe if needed.
    pause
    exit /b 2
  )
)

cd /d "%REPO_ROOT%"
echo Launching calcium pipeline dashboard...
echo Repo root: %REPO_ROOT%
echo Conda: %CONDA_BIN%
echo Environment: %DASHBOARD_ENV%

"%CONDA_BIN%" run -n "%DASHBOARD_ENV%" --no-capture-output python current\pipeline_dashboard_gui.py %*
if errorlevel 1 (
  echo.
  echo Dashboard launch failed.
  pause
  exit /b 1
)

endlocal
