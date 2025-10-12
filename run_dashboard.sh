#!/bin/bash

# Deep Learning Market Microstructure Analyzer Dashboard Launcher

echo "🚀 Starting Deep Learning Market Microstructure Analyzer Dashboard..."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing dashboard dependencies..."
    pip install -r dashboard/requirements.txt
fi

# Set the working directory
cd "$(dirname "$0")"

# Launch the dashboard
echo "🌐 Launching dashboard at http://localhost:8501"
echo "📊 Dashboard will open automatically in your browser"

streamlit run dashboard/main_dashboard.py --server.port 8501 --server.address localhost