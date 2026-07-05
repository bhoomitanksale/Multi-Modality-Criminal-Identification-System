@echo off
title CIS v2 - Full System Launcher
color 0B
cls

echo =========================================================================
echo  CIS v2 - STARTING FULL SYSTEM (BACKEND + FRONTEND)
echo =========================================================================
echo.

:: 1. Start the Web API in a new minimized window
echo [1/2] Launching Interpol Web API...
start "CIS v2 - Web API" /min python web_api/app.py

:: Give the API a second to initialize
timeout /t 2 /nobreak >nul

:: 2. Start the main application
echo [2/2] Launching Main Application...
python cis_v2.py

echo.
echo System closed.
pause
