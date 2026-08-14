@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
title Stock Advisor

python -m src.main %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================
    echo  Program exited with error code: %ERRORLEVEL%
    echo  Please screenshot the error above
    echo ============================================
    pause
)
