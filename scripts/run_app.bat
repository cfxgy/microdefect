@echo off
REM MicroDefect launcher (run from repository root or scripts folder)
setlocal
set PORT=%~1
if %PORT%==" set PORT=8501
cd /d %~dp0..
if not exist .venv\Scripts\python.exe (
  echo [ERROR] .venv not found. Create it and install requirements first.
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run src\ui\app.py --server.address 127.0.0.1 --server.port %PORT%
endlocal
