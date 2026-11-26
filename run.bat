@echo off
rem Usage: run.bat [main.py options], e.g.  run.bat --source loopback --debug
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py %*
