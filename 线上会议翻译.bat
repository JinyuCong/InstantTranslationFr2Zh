@echo off
rem Translate whatever the computer is playing (Teams / Zoom / Tencent Meeting / browser)
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py --source loopback %*
pause
