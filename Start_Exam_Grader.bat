@echo off
setlocal
title Automated Exam Answer Sheet Checker

cd /d "%~dp0"

set "VENV_PY=%~dp0exam_grader_mvp\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] Python virtual environment not found at:
    echo   "%VENV_PY%"
    echo Please create exam_grader_mvp\.venv first.
    pause
    exit /b 1
)

echo ============================================================
echo   AI Exam Answer Sheet Grader - One-Click Launcher
echo ============================================================
echo.

REM Check if server is already running on http://127.0.0.1:8000
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/config' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %errorlevel%==0 (
    echo [OK] Server is already running at http://127.0.0.1:8000
    echo Opening application in your browser...
    start "" "http://127.0.0.1:8000"
    timeout /t 2 /nobreak >nul
    exit /b 0
)

echo [Starting] Launching FastAPI server on http://127.0.0.1:8000 ...
echo [Info] Your browser will open automatically as soon as the server is ready.
echo [Info] Keep this window open while grading papers. Press Ctrl+C to stop.
echo.

REM Wait in background until /api/config responds, then open browser
start "" /b powershell -NoProfile -Command "for ($i = 0; $i -lt 45; $i++) { try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/config' -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -eq 200) { Start-Process 'http://127.0.0.1:8000'; break } } catch {}; Start-Sleep -Seconds 1 }"

"%VENV_PY%" -m uvicorn app:app --app-dir "%~dp0exam_grader_mvp" --host 127.0.0.1 --port 8000

if errorlevel 1 (
    echo.
    echo [ERROR] Server stopped with an error. See details above.
    pause
)
