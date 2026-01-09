#!/bin/bash
# HardTalks Startup Script

echo "🚀 Starting HardTalks Chat Assistant"
echo "======================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists, if not create one
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Validate setup
echo "🔍 Validating setup..."
python3 test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup validation passed!"
    echo "🎯 Starting server..."
    echo ""
    echo "Access your chat bot at: http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Start the server
    python3 main.py
else
    echo ""
    echo "❌ Setup validation failed. Please check the errors above."
    exit 1
fi