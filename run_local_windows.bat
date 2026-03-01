@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set PY=py) else (set PY=python)

set PORT=8506
echo Installing requirements (user-level)...
%PY% -m pip install --user -r requirements.txt
echo.
echo Running app at http://localhost:%PORT% ...
%PY% -m streamlit run app.py --server.port %PORT% --server.headless false
