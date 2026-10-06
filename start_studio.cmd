@echo off
rem Start the Dub Studio web app (API + React UI) on http://127.0.0.1:8000
rem The Python environment lives on C: (%USERPROFILE%\.venvs\video-dubber); falls back to .venv here.
set "PY=%USERPROFILE%\.venvs\video-dubber\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0.venv\Scripts\python.exe"
cd /d "%~dp0"
"%PY%" serve.py %*
