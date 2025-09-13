#!/usr/bin/env python3
"""
Environment-aware launcher that automatically uses the correct Python interpreter.
"""

import sys
import os
import subprocess
from pathlib import Path

def find_venv_python():
    """Find the virtual environment Python executable."""
    current_dir = Path.cwd()
    
    # Common virtual environment paths
    venv_paths = [
        current_dir / ".venv" / "bin" / "python",
        current_dir / "venv" / "bin" / "python",
        current_dir / ".venv" / "Scripts" / "python.exe",  # Windows
        current_dir / "venv" / "Scripts" / "python.exe",   # Windows
    ]
    
    for venv_path in venv_paths:
        if venv_path.exists():
            return str(venv_path)
    
    return None

def check_packages(python_cmd):
    """Check if required packages are available."""
    try:
        result = subprocess.run([
            python_cmd, "-c", 
            "import cv2, numpy, matplotlib, json; print('OK')"
        ], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def main():
    """Main launcher function."""
    print("🔍 Checking Python environment...")
    
    # First, try to find virtual environment Python
    venv_python = find_venv_python()
    
    if venv_python and check_packages(venv_python):
        print(f"✅ Using virtual environment: {venv_python}")
        python_cmd = venv_python
    else:
        # Try current Python interpreter
        if check_packages(sys.executable):
            print(f"✅ Using current Python: {sys.executable}")
            python_cmd = sys.executable
        else:
            print("❌ Error: Required packages not found!")
            print("\nPlease install required packages:")
            print("pip install opencv-python numpy matplotlib pillow scikit-image scipy scikit-learn")
            print("\nOr set up the virtual environment:")
            print("python -m venv .venv")
            print("source .venv/bin/activate  # On macOS/Linux")
            print("pip install opencv-python numpy matplotlib pillow scikit-image scipy scikit-learn")
            return
    
    # Run the curve identifier with the correct Python
    print("🚀 Starting curve identification tool...")
    try:
        subprocess.run([python_cmd, "curve_identifier.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running tool: {e}")
    except KeyboardInterrupt:
        print("\n👋 Tool cancelled by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()