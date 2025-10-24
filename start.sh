#!/bin/bash

# Facial Authentication System - Full Stack Startup Script
# This script starts both the backend API and frontend development server

set -e

echo "=========================================="
echo "Facial Authentication System"
echo "Full Stack Web Application"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    exit 1
fi

echo -e "${BLUE}[1/6] Checking dependencies...${NC}"

# Install Python dependencies if needed
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || {
    echo -e "${RED}Failed to activate virtual environment${NC}"
    exit 1
}

echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -q -r requirements.txt

echo -e "${GREEN}✓ Python dependencies ready${NC}"

# Install Node.js dependencies if needed
echo -e "${BLUE}[2/6] Setting up frontend...${NC}"
cd frontend

if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
    npm install
else
    echo -e "${GREEN}✓ Node modules already installed${NC}"
fi

cd ..

# Initialize database
echo -e "${BLUE}[3/6] Initializing database...${NC}"
python3 -c "from app.core.database import init_db; init_db()" 2>/dev/null || echo -e "${YELLOW}Database already initialized${NC}"
echo -e "${GREEN}✓ Database ready${NC}"

# Check if weights exist
echo -e "${BLUE}[4/6] Checking model weights...${NC}"
if [ ! -f "weights/fusion_mlp.pth" ]; then
    echo -e "${YELLOW}Warning: Model weights not found in weights/fusion_mlp.pth${NC}"
    echo -e "${YELLOW}The system will download weights on first use${NC}"
else
    echo -e "${GREEN}✓ Model weights found${NC}"
fi

# Start backend
echo -e "${BLUE}[5/6] Starting backend API server...${NC}"
echo -e "${YELLOW}Backend will be available at: http://localhost:8000${NC}"
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Backend failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Backend server started (PID: $BACKEND_PID)${NC}"

# Start frontend
echo -e "${BLUE}[6/6] Starting frontend development server...${NC}"
echo -e "${YELLOW}Frontend will be available at: http://localhost:3000${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 3

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Frontend failed to start${NC}"
    kill $BACKEND_PID
    exit 1
fi

echo -e "${GREEN}✓ Frontend server started (PID: $FRONTEND_PID)${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Full Stack Application Running!${NC}"
echo "=========================================="
echo ""
echo -e "${BLUE}Access Points:${NC}"
echo -e "  Frontend:  ${GREEN}http://localhost:3000${NC}"
echo -e "  Backend:   ${GREEN}http://localhost:8000${NC}"
echo -e "  API Docs:  ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  Health:    ${GREEN}http://localhost:8000/health${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✓ Servers stopped${NC}"
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup INT TERM

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID


