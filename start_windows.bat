@echo off
rem Double-click this file to start the app on Windows.
rem You can also pass a command, e.g.  start_windows.bat notebook
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 run.py %*
) else (
    python run.py %*
)

if errorlevel 1 (
    echo.
    echo Something went wrong. If Python is missing, install Python 3.13 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation.
    pause
)
