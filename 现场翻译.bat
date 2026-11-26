@echo off
rem Live translation from the default microphone
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py --source mic %*
pause
