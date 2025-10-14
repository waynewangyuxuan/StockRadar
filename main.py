#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the StockRadar application.
This script imports and runs the backend.
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

if __name__ == "__main__":
    # Import and run the backend launcher
    from backend.app import main
    
    # Run the backend
    main() 