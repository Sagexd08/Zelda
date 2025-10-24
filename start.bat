@echo off
REM Facial Authentication System - Full Stack Startup Script (Windows)
REM This script starts both the backend API and frontend development server

echo ==========================================
echo Facial Authentication System
echo Full Stack Web Application
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed
    pause
    exit /b 1
)

echo [1/6] Checking dependencies...

REM Create virtual environment if needed
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install -q -r requirements.txt

echo [DONE] Python dependencies ready

REM Install Node.js dependencies
echo [2/6] Setting up frontend...
cd frontend

if not exist "node_modules" (
    echo Installing Node.js dependencies...
    call npm install
) else (
    echo [DONE] Node modules already installed
)

cd ..

REM Initialize database
echo [3/6] Initializing database...
python -c "from app.core.database import init_db; init_db()" 2>nul
echo [DONE] Database ready

REM Check weights
echo [4/6] Checking model weights...
if not exist "weights\fusion_mlp.pth" (
    echo Warning: Model weights not found
    echo The system will download weights on first use
) else (
    echo [DONE] Model weights found
)

REM Start backend
echo [5/6] Starting backend API server...
echo Backend will be available at: http://localhost:8000
start "Backend Server" cmd /k "venv\Scripts\activate.bat && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait for backend to start
timeout /t 3 /nobreak >nul

echo [DONE] Backend server started

REM Start frontend
echo [6/6] Starting frontend development server...
echo Frontend will be available at: http://localhost:3000
start "Frontend Server" cmd /k "cd frontend && npm run dev"

REM Wait for frontend to start
timeout /t 3 /nobreak >nul

echo.
echo ==========================================
echo [DONE] Full Stack Application Running!
echo ==========================================
echo.
echo Access Points:
echo   Frontend:  http://localhost:3000
echo   Backend:   http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo   Health:    http://localhost:8000/health
echo.
echo Close the terminal windows to stop the servers
echo.
pause


