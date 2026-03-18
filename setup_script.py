#!/usr/bin/env python3
"""
Setup script for the Impedance Spectroscopy Analyzer
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"OK: Python version: {sys.version}")
        return True


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("OK: All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("ERROR: Failed to install packages!")
        print("Try manually: pip install -r requirements.txt")
        return False


def check_files():
    """Check if all required files are present."""
    required_files = [
        "circuit_models.py",
        "optimization_algorithms_clean.py", 
        "impedance_fitter.py",
        "file_manager.py",
        "run_analysis.py",
        "main.py",
        "requirements.txt"
    ]
    
    print("Checking required files...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"OK: {file}")
        else:
            print(f"ERROR: {file} - MISSING!")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nERROR: Missing files: {missing_files}")
        return False
    else:
        print("OK: All required files present!")
        return True


def create_test_structure():
    """Create test directory structure."""
    print("Creating test directories...")
    
    test_dirs = [
        "test_data",
        "results",
        "examples"
    ]
    
    for dir_name in test_dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"OK: Created: {dir_name}/")


def run_test():
    """Run a basic test to check if everything works."""
    print("Running basic functionality test...")
    
    try:
        # Test imports
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.optimize import differential_evolution
        
        # Test our modules
        from circuit_models import CircuitModels
        from optimization_algorithms_clean import OptimizationAlgorithms
        from impedance_fitter import ImpedanceFitter
        from file_manager import FileManager
        
        print("OK: All imports successful!")
        
        # Test basic functionality
        print("Testing circuit models...")
        piecewise_model = CircuitModels('piecewise')
        unified_model = CircuitModels('unified')
        print("OK: Circuit models initialized!")
        
        print("Testing optimization algorithms...")
        optimizer = OptimizationAlgorithms(piecewise_model.bounds, piecewise_model.lb, piecewise_model.ub)
        print("OK: Optimization algorithms initialized!")
        
        print("Testing file manager...")
        file_manager = FileManager()
        dirs = file_manager.create_directory_structure("test_setup")
        print("OK: File manager working!")
        
        print("OK: Basic functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"ERROR: Test FAILED: {str(e)}")
        return False


def main():
    """Main setup function."""
    print("IMPEDANCE SPECTROSCOPY ANALYZER SETUP")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check required files
    if not check_files():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create directories
    create_test_structure()
    
    # Run test
    if not run_test():
        return False
    
    print("\nSETUP COMPLETE!")
    print("=" * 50)
    print("Ready to analyze impedance data!")
    print("\nQuick Start:")
    print("   python run_analysis.py                    # Interactive mode")
    print("   python run_analysis.py data.csv           # Single file")
    print("   python run_analysis.py /path/to/data/     # Directory")
    print("\nOutput will be saved in organized folders with:")
    print("   - Individual results (CSV + PNG)")
    print("   - Model comparisons")
    print("   - Batch summaries")
    print("   - Method performance statistics")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\nERROR: Setup failed! Please check the errors above.")
        sys.exit(1)
