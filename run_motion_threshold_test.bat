@echo off
setlocal
cd /d "%~dp0"
echo ====================================================================
echo   Starting SWSTP Real-Time Motion Threshold ^& ROI Calibrator
echo ====================================================================
echo.

if exist "swstp_edge\test_motion_thresholds.py" (
    set SCRIPT_PATH=swstp_edge\test_motion_thresholds.py
) else (
    set SCRIPT_PATH=test_motion_thresholds.py
)

REM 1. Try Python 3.12 launcher (where all full packages are installed)
py -3.12 %SCRIPT_PATH% %*
if %ERRORLEVEL% EQU 0 goto :done

REM 2. Fall back to standard py launcher
py %SCRIPT_PATH% %*
if %ERRORLEVEL% EQU 0 goto :done

REM 3. Fall back to system python
python %SCRIPT_PATH% %*

:done
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Process exited with code %ERRORLEVEL%.
    pause
)

