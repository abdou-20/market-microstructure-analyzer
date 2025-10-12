#!/usr/bin/env python3
"""
Dashboard Launcher for Deep Learning Market Microstructure Analyzer

This script sets up and launches the comprehensive visualization dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path
import webbrowser
import time

def install_requirements():
    """Install required packages for the dashboard."""
    requirements_file = Path(__file__).parent / "dashboard" / "requirements.txt"
    
    if requirements_file.exists():
        print("📦 Installing dashboard dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    dashboard_file = Path(__file__).parent / "dashboard" / "main_dashboard.py"
    
    if not dashboard_file.exists():
        print(f"❌ Dashboard file not found: {dashboard_file}")
        return False
    
    print("🚀 Starting Deep Learning Market Microstructure Analyzer Dashboard...")
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("📊 Opening dashboard in your browser...")
    
    # Start the dashboard
    try:
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_file),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False
    
    return True

def main():
    """Main launcher function."""
    print("=" * 70)
    print("🚀 DEEP LEARNING MARKET MICROSTRUCTURE ANALYZER")
    print("📊 Comprehensive Visualization Dashboard")
    print("=" * 70)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Please run this script from the project root directory")
        return False
    
    # Install dependencies
    if not install_requirements():
        print("❌ Failed to install dependencies")
        return False
    
    print("\n🎯 Dashboard Features:")
    print("  • 🏠 Project Overview & Status")
    print("  • 📊 Model Performance Analysis")  
    print("  • 🎯 Directional Accuracy Metrics")
    print("  • 💹 Market Data Visualization")
    print("  • ⚡ Real-time Monitoring")
    print("  • 🔄 Training Metrics & History")
    print("  • 💰 Backtesting Results")
    print("  • 🛠️ System Health Monitoring")
    
    print("\n" + "=" * 70)
    
    # Launch dashboard
    return launch_dashboard()

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n❌ Dashboard launch failed")
        sys.exit(1)
    else:
        print("\n✅ Dashboard session completed")