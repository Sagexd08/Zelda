#!/bin/bash
# Development setup script

set -e

echo "=== Setting up Facial Authentication System ==="

# Check Python version
echo "Checking Python version..."
python --version

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install dev dependencies
echo "Installing development dependencies..."
pip install pytest pytest-cov black flake8 mypy pre-commit

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Copy environment file
if [ ! -f ".env" ]; then
    echo "Copying environment template..."
    cp env.example .env
    echo "Please edit .env with your configuration"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p logs weights data

# Initialize database
echo "Initializing database..."
python -c "from app.core.database import init_db; init_db()"

echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "1. Edit .env with your configuration"
echo "2. Download model weights (see weights/README.md)"
echo "3. Run 'python -m uvicorn app.main:app --reload' to start development server"

