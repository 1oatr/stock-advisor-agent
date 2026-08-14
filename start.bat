@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title Stock Advisor - Web UI

echo.
echo  ============================================
echo    Stock Advisor - Web UI Launcher
echo  ============================================
echo.

rem ---- Step 1: check Python ----
where python >nul 2>nul
if errorlevel 1 (
    echo  [ERROR] Python not found. Install Python 3.10+ first, check "Add to PATH".
    echo  Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

rem ---- Step 2: check dependencies, install if missing ----
echo  [1/3] Checking dependencies...
python -c "import importlib.util,sys; mods=['flask','akshare','pandas','numpy','pandas_ta','torch','gymnasium','stable_baselines3','openai','yaml','click','rich','matplotlib']; miss=[m for m in mods if importlib.util.find_spec(m) is None]; sys.exit(1 if miss else 0)"
if errorlevel 1 (
    echo  [2/3] Installing dependencies... This may take a few minutes.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo  [ERROR] Dependency install failed. Check your network and retry.
        pause
        exit /b 1
    )
) else (
    echo  [2/3] Dependencies ready.
)

rem ---- Step 3: start server and open browser ----
echo  [3/3] Starting server...
start "" /min cmd /c "timeout /t 2 /nobreak >nul & start "" http://127.0.0.1:5000"
set PYTHONIOENCODING=utf-8
python -m src.webui.app

if errorlevel 1 (
    echo.
    echo  ============================================
    echo  Program exited with error code %ERRORLEVEL%
    echo  ============================================
    pause
)
endlocal
