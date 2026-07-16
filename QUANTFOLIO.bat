@echo off
rem Quantfolio launcher - double-click me
cd /d "%~dp0"
start "Quantfolio" /b pythonw gui.py
if errorlevel 1 (
    python gui.py
    pause
)
