@echo off
setlocal enabledelayedexpansion
title Fresta Finance - Full Analysis Pipeline

echo.
echo ================================================
echo   FRESTA FINANCE - Full Analysis Pipeline
echo   Order 1-3 : fresta_finance.py  (~15-30 min)
echo   Order 4   : fresta_tree.py     (~2-4 hours)
echo   Order 5   : fresta_political.py (~1-2 hours)
echo ================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found.

:: Check LM Studio
echo [CHECK] Verifying LM Studio is running...
curl -s http://127.0.0.1:1234/v1/models >nul 2>&1
if not errorlevel 1 (
    echo [OK] LM Studio is running.
    goto lm_ok
)

echo.
echo [ERROR] LM Studio is not running or model not loaded!
echo.
echo   Please:
echo   1. Open LM Studio
echo   2. Go to the Developer tab (left sidebar)
echo   3. Click "Start Server" (should show Status: Running)
echo   4. Make sure meta-llama-3-8b-instruct is loaded
echo   5. Run this .bat again
echo.
pause
exit /b 1

:lm_ok
echo.

:: Install dependencies
echo [SETUP] Installing Python dependencies...
pip install pandas numpy yfinance requests --quiet 2>nul
echo [OK] Dependencies ready.
echo.

:: ------------------------------------------------
:: STEP 1 - Financial Analysis
:: ------------------------------------------------
echo [STEP 1/3] Financial Analysis
echo.

if exist "output\sp500_entropy_ranked.csv" (
    echo [SKIP] sp500_entropy_ranked.csv already exists.
    set /p RUN1="Run fresta_finance.py again for fresh data? (y/N): "
    if /i "!RUN1!"=="y" goto run_finance
    echo [OK] Using existing financial data.
    goto step2
)

:run_finance
echo [RUN] python fresta_finance.py
echo       This will take 15-30 minutes...
echo.
python fresta_finance.py
if errorlevel 1 (
    echo [ERROR] fresta_finance.py failed! Check output\run.log
    pause
    exit /b 1
)
echo [OK] Financial analysis complete!

:step2
echo.
:: ------------------------------------------------
:: STEP 2 - Tree Analysis
:: ------------------------------------------------
echo [STEP 2/3] Supply Chain Tree Analysis
echo.

if exist "output\sp500_tree_analysis.csv" (
    echo [SKIP] sp500_tree_analysis.csv already exists.
    set /p RUN2="Run fresta_tree.py again? (y/N): "
    if /i "!RUN2!"=="y" goto run_tree
    echo [OK] Using existing tree data.
    goto step3
)

:run_tree
echo [RUN] python fresta_tree.py
echo       This will take 2-4 hours. Safe to resume - results are cached!
echo.
python fresta_tree.py
if errorlevel 1 (
    echo [ERROR] fresta_tree.py failed! Check output\tree.log
    pause
    exit /b 1
)
echo [OK] Tree analysis complete!

:step3
echo.
:: ------------------------------------------------
:: STEP 3 - Political Analysis
:: ------------------------------------------------
echo [STEP 3/3] Political-Economic Risk Analysis
echo.
echo [RUN] python fresta_political.py
python fresta_political.py
if errorlevel 1 (
    echo [ERROR] fresta_political.py failed! Check output\political.log
    pause
    exit /b 1
)

:: ------------------------------------------------
:: DONE
:: ------------------------------------------------
echo.
echo ================================================
echo   ANALYSIS COMPLETE!
echo   - sp500_entropy_ranked.csv
echo   - sp500_tree_analysis.csv
echo   - sp500_unified_report.html  (open this!)
echo ================================================
echo.

if exist "output\sp500_unified_report.html" (
    echo [OPEN] Opening report in browser...
    start "" "output\sp500_unified_report.html"
) else (
    echo [WARN] Report not found - check errors above.
)

pause
