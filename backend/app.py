#!/usr/bin/env python3
"""
StockRadar Backend Launcher

This script serves as the main entry point for the StockRadar backend.
It allows running the system in different modes:
1. API mode - Starts the RESTful API server
2. CLI mode - Runs the traditional command-line interface
"""

import argparse
import logging
import sys
import os
import socket

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="StockRadar Trading System")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["api", "cli"],
        default="api",
        help="Mode to run the backend (api or cli)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (required for CLI mode)"
    )
    parser.add_argument(
        "--trading-mode",
        type=str,
        choices=["live", "backtest"],
        help="Trading mode (required for CLI mode, overrides config file)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for API server (only used in API mode)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for API server (only used in API mode)"
    )
    return parser.parse_args()

def run_api_mode(host, port):
    """Run in API mode - start the RESTful API server."""
    # Import API server
    from api import app, initialize_registries
    
    # Initialize components
    initialize_registries()
    
    # Try to start the server, if port is in use, try alternative ports
    max_port_attempts = 10
    original_port = port
    
    for attempt in range(max_port_attempts):
        try:
            print(f"Starting API server on {host}:{port}")
            # Use threaded=True for better concurrency
            app.run(host=host, port=port, debug=True, use_reloader=False, threaded=True)
            break  # If successful, break the loop
        except socket.error as e:
            if 'Address already in use' in str(e) and attempt < max_port_attempts - 1:
                port += 1
                print(f"Port {port-1} is already in use, trying port {port}...")
            else:
                print(f"Failed to start server: {e}")
                if attempt == max_port_attempts - 1:
                    print(f"Could not find an available port after {max_port_attempts} attempts.")
                    print(f"Please specify a different port with --port option.")
                    sys.exit(1)
                raise

def run_cli_mode(config_path, trading_mode):
    """Run in CLI mode - traditional command-line interface."""
    # Import CLI runner
    from run import main as run_main
    
    # Set command line arguments for the CLI runner
    sys.argv = [sys.argv[0]]
    
    if config_path:
        sys.argv.extend(["--config", config_path])
    else:
        print("Error: Config file is required for CLI mode")
        sys.exit(1)
        
    if trading_mode:
        sys.argv.extend(["--mode", trading_mode])
        
    # Run the CLI mode
    run_main()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    try:
        if args.mode == "api":
            # Run in API mode
            run_api_mode(args.host, args.port)
        else:
            # Run in CLI mode
            run_cli_mode(args.config, args.trading_mode)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 