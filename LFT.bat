@echo off
rem LFT (Le Fort) launcher - double-click me
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "LFT" /b ".venv\Scripts\pythonw.exe" gui.py
    exit /b 0
)
where pythonw >nul 2>nul
if not errorlevel 1 (
    start "LFT" /b pythonw gui.py
    exit /b 0
)
python gui.py
if errorlevel 1 pause
