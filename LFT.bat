@echo off
rem LFT (Le Fort) launcher - double-click me
cd /d "%~dp0"
start "LFT" /b pythonw gui.py
if errorlevel 1 (
    python gui.py
    pause
)
