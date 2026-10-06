@echo off
rem Command-line dubbing:  dub.cmd "https://www.youtube.com/watch?v=..." [--max-minutes 2] [...]
set "PY=%USERPROFILE%\.venvs\video-dubber\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0.venv\Scripts\python.exe"
cd /d "%~dp0"
"%PY%" dub.py %*
