@echo off
title AI Exam Grader Server
echo ========================================================
echo   Starting AI Exam Grader & Opening Browser...
echo ========================================================
start "" "http://127.0.0.1:8000"
"exam_grader_mvp\.venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000 --app-dir exam_grader_mvp
pause
