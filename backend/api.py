#!/usr/bin/env python3
"""
StockRadar API - RESTful API for the StockRadar trading system.

This module provides RESTful endpoints to interact with the StockRadar
trading system. It allows users to:
- Manage strategies
- Control trading sessions
- Configure system parameters
- Access market data and portfolio information
"""

import os
import json
import logging
import threading
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

from core.config import ConfigManager
from core.runner import TradingRunner
from backend.strategy_engine.strategy_registry import StrategyRegistry
from core.factor_registry import FactorRegistry
from data_fetcher.yfinance_provider import YahooFinanceProvider
from data_processor.processor import DataProcessor

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Shared state for the trading runner
trading_state = {
    'runner': None,
    'is_running': False,
    'mode': None,
    'config': None,
    'lock': threading.Lock()
}

# Initialize components
strategy_registry = StrategyRegistry()
factor_registry = FactorRegistry()

# Configure logging
logger = logging.getLogger(__name__)

#################################################
# Root Endpoint
#################################################

@app.route('/', methods=['GET'])
def root():
    """Root endpoint that provides API information and available endpoints."""
    # Check if client accepts HTML
    if 'text/html' in request.headers.get('Accept', ''):
        # Return HTML interface
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>StockRadar API</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                    color: #2c3e50;
                }
                h2 {
                    margin-top: 30px;
                    color: #3498db;
                }
                .endpoint {
                    background: #f8f9fa;
                    padding: 10px 15px;
                    border-radius: 4px;
                    margin-bottom: 10px;
                    border-left: 4px solid #3498db;
                }
                .method {
                    font-weight: bold;
                    color: #e74c3c;
                }
                .path {
                    font-family: monospace;
                    font-weight: bold;
                }
                .description {
                    margin-left: 10px;
                    color: #555;
                }
                .status {
                    background: #d4edda;
                    color: #155724;
                    padding: 5px 10px;
                    border-radius: 4px;
                    display: inline-block;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <h1>StockRadar API</h1>
            <p>RESTful API for the StockRadar trading system. Version 1.0.0</p>
            
            <div class="status">Status: Running</div>
            
            <h2>Strategies Endpoints</h2>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/api/strategies</span>
                <span class="description">List all available strategies</span>
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/api/strategies/{id}</span>
                <span class="description">Get details about a specific strategy</span>
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/api/strategies/{id}/enable</span>
                <span class="description">Enable a strategy for trading</span>
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/api/strategies/{id}/disable</span>
                <span class="description">Disable a strategy</span>
            </div>
            
            <h2>Trading Endpoints</h2>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/api/trading/start</span>
                <span class="description">Start trading with specified mode and configuration</span>
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/api/trading/stop</span>
                <span class="description">Stop the current trading session</span>
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/api/trading/status</span>
                <span class="description">Get the current status of the trading system</span>
            </div>
            
            <h2>Configuration Endpoints</h2>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/api/config</span>
                <span class="description">Get the current configuration</span>
            </div>
            <div class="endpoint">
                <span class="method">PUT</span>
                <span class="path">/api/config</span>
                <span class="description">Update the configuration</span>
            </div>
            
            <h2>Data Endpoints</h2>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/api/data/market</span>
                <span class="description">Get market data for specified symbols (use ?symbols=AAPL,MSFT&start_date=2023-01-01&end_date=2023-12-31)</span>
            </div>
            
            <h2>Portfolio Endpoints</h2>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/api/portfolio</span>
                <span class="description">Get the current portfolio status</span>
            </div>
            
            <p style="margin-top: 40px; font-size: 0.9em; color: #777;">
                For programmatic access, this endpoint also returns JSON when accessed with the appropriate Accept header.
            </p>
        </body>
        </html>
        """
        return html
    else:
        # Return JSON for programmatic access
        api_info = {
            "name": "StockRadar API",
            "version": "1.0.0",
            "description": "RESTful API for the StockRadar trading system",
            "endpoints": {
                "strategies": {
                    "GET /api/strategies": "List all available strategies",
                    "GET /api/strategies/{id}": "Get details about a specific strategy",
                    "POST /api/strategies/{id}/enable": "Enable a strategy for trading",
                    "POST /api/strategies/{id}/disable": "Disable a strategy"
                },
                "trading": {
                    "POST /api/trading/start": "Start trading with specified mode and configuration",
                    "POST /api/trading/stop": "Stop the current trading session",
                    "GET /api/trading/status": "Get the current status of the trading system"
                },
                "configuration": {
                    "GET /api/config": "Get the current configuration",
                    "PUT /api/config": "Update the configuration"
                },
                "data": {
                    "GET /api/data/market": "Get market data for specified symbols (use ?symbols=AAPL,MSFT&start_date=2023-01-01&end_date=2023-12-31)"
                },
                "portfolio": {
                    "GET /api/portfolio": "Get the current portfolio status"
                }
            },
            "status": "running"
        }
        return jsonify(api_info)

#################################################
# Strategy Management Endpoints
#################################################

@app.route('/api/strategies', methods=['GET'])
def list_strategies():
    """List all available strategies."""
    try:
        # Make sure registry is initialized
        if not strategy_registry.get_all():
            # Try to initialize registries if empty
            initialize_registries()
            
        strategies = []
        all_strategies = strategy_registry.get_all()
        logger.info(f"Available strategies: {list(all_strategies.keys())}")
        
        for name, strategy_class in all_strategies.items():
            strategies.append({
                'id': name,
                'name': strategy_class.__name__,
                'description': strategy_class.__doc__ or "No description available",
                'parameters': {} # Default parameters would go here
            })
        return jsonify(strategies)
    except Exception as e:
        logger.exception("Error listing strategies")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/<strategy_id>', methods=['GET'])
def get_strategy(strategy_id):
    """Get details about a specific strategy."""
    try:
        # Make sure registry is initialized
        if not strategy_registry.get_all():
            initialize_registries()
            
        # Log available strategies for debugging
        logger.info(f"Looking for strategy: {strategy_id}")
        logger.info(f"Available strategies: {list(strategy_registry.get_all().keys())}")
        
        strategy_class = strategy_registry.get(strategy_id)
        if not strategy_class:
            # Try case-insensitive match
            for name, cls in strategy_registry.get_all().items():
                if name.lower() == strategy_id.lower():
                    strategy_class = cls
                    strategy_id = name  # Use the correct case
                    break
                    
        if not strategy_class:
            return jsonify({'error': 'Strategy not found', 'available': list(strategy_registry.get_all().keys())}), 404
            
        strategy_info = {
            'id': strategy_id,
            'name': strategy_class.__name__,
            'description': strategy_class.__doc__ or "No description available",
            'parameters': {} # Default parameters would go here
        }
        return jsonify(strategy_info)
    except Exception as e:
        logger.exception(f"Error getting strategy {strategy_id}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/<strategy_id>/enable', methods=['POST'])
def enable_strategy(strategy_id):
    """Enable a strategy for trading."""
    try:
        # Make sure registry is initialized
        if not strategy_registry.get_all():
            initialize_registries()
            
        # Check if strategy exists
        strategy_class = strategy_registry.get(strategy_id)
        # Try case-insensitive match
        if not strategy_class:
            for name, cls in strategy_registry.get_all().items():
                if name.lower() == strategy_id.lower():
                    strategy_class = cls
                    strategy_id = name
                    break
                    
        if not strategy_class:
            return jsonify({'error': 'Strategy not found', 'available': list(strategy_registry.get_all().keys())}), 404
            
        # Get parameters from request
        data = request.json or {}
        parameters = data.get('parameters', {})
        
        # Validate parameters
        # TODO: Implement parameter validation
        
        # Update config
        with trading_state['lock']:
            # Initialize config if it doesn't exist
            if not trading_state['config']:
                trading_state['config'] = {
                    'general': {
                        'mode': 'backtest',
                        'paper_trading': True
                    },
                    'strategies': []
                }
                
            # Update strategy config
            if 'strategies' not in trading_state['config']:
                trading_state['config']['strategies'] = []
                
            # Remove strategy if already exists
            trading_state['config']['strategies'] = [
                s for s in trading_state['config']['strategies'] 
                if s.get('name') != strategy_id
            ]
            
            # Add strategy with new parameters
            trading_state['config']['strategies'].append({
                'name': strategy_id,
                'enabled': True,
                'parameters': parameters
            })
            
        return jsonify({
            'status': 'success', 
            'message': f'Strategy {strategy_id} enabled',
            'config': trading_state['config']  # Return current config for debugging
        })
    except Exception as e:
        logger.exception(f"Error enabling strategy {strategy_id}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/<strategy_id>/disable', methods=['POST'])
def disable_strategy(strategy_id):
    """Disable a strategy for trading."""
    try:
        with trading_state['lock']:
            if not trading_state['config']:
                return jsonify({'error': 'No active configuration'}), 400
                
            # Update strategy config if it exists
            if 'strategies' in trading_state['config']:
                for strategy in trading_state['config']['strategies']:
                    if strategy.get('name') == strategy_id:
                        strategy['enabled'] = False
                        
        return jsonify({'status': 'success', 'message': f'Strategy {strategy_id} disabled'})
    except Exception as e:
        logger.exception(f"Error disabling strategy {strategy_id}")
        return jsonify({'error': str(e)}), 500

#################################################
# Trading Control Endpoints
#################################################

@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    """Start trading with specified mode and configuration."""
    try:
        # Get parameters from request
        data = request.json or {}
        mode = data.get('mode', 'backtest')  # Default to backtest
        config_path = data.get('config_path')
        config_data = data.get('config')
        
        # Validate mode
        if mode not in ['live', 'backtest']:
            return jsonify({'error': 'Invalid mode. Must be "live" or "backtest"'}), 400
            
        # Check if already running
        with trading_state['lock']:
            if trading_state['is_running']:
                return jsonify({'error': 'Trading is already running'}), 400
        
        # Load configuration
        if config_path:
            # Load from file
            if not os.path.exists(config_path):
                return jsonify({'error': f'Config file not found: {config_path}'}), 404
                
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                return jsonify({'error': f'Failed to load config file: {str(e)}'}), 400
        elif config_data:
            # Use provided config
            config = config_data
        else:
            return jsonify({'error': 'No configuration provided'}), 400
            
        # Initialize components
        data_provider = YahooFinanceProvider(config.get('data', {}))
        data_processor = DataProcessor(config.get('data_processor', {}))
        
        # Set mode
        config['general'] = config.get('general', {})
        config['general']['mode'] = mode
        
        # Create runner
        runner = TradingRunner(
            config=config,
            strategy_registry=strategy_registry,
            factor_registry=factor_registry,
            data_provider=data_provider,
            data_processor=data_processor
        )
        
        # Start trading based on mode
        with trading_state['lock']:
            trading_state['runner'] = runner
            trading_state['config'] = config
            trading_state['mode'] = mode
            trading_state['is_running'] = True
        
        if mode == 'live':
            # Start in a separate thread
            threading.Thread(target=runner.run_live, daemon=True).start()
            message = 'Live trading started'
        else:
            # Run backtest (blocking)
            results = runner.run_backtest()
            message = 'Backtest completed'
            
            # Update state
            with trading_state['lock']:
                trading_state['is_running'] = False
            
        return jsonify({'status': 'success', 'message': message})
    except Exception as e:
        logger.exception("Error starting trading")
        # Update state in case of error
        with trading_state['lock']:
            trading_state['is_running'] = False
            
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    """Stop the current trading session."""
    try:
        with trading_state['lock']:
            if not trading_state['is_running']:
                return jsonify({'message': 'No active trading session'}), 200
                
            if trading_state['runner']:
                if trading_state['mode'] == 'live':
                    trading_state['runner'].stop_live()
                
                trading_state['is_running'] = False
                message = f"{trading_state['mode'].capitalize()} trading stopped"
                
                # Clear runner
                trading_state['runner'] = None
                
                return jsonify({'status': 'success', 'message': message})
            else:
                return jsonify({'error': 'No active runner found'}), 500
    except Exception as e:
        logger.exception("Error stopping trading")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/status', methods=['GET'])
def get_trading_status():
    """Get the current status of the trading system."""
    try:
        with trading_state['lock']:
            status = {
                'is_running': trading_state['is_running'],
                'mode': trading_state['mode'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Add additional info if available
            if trading_state['runner'] and trading_state['is_running']:
                if hasattr(trading_state['runner'], 'portfolio'):
                    status['portfolio'] = {
                        'cash': trading_state['runner'].portfolio.get('cash', 0),
                        'positions_count': len(trading_state['runner'].portfolio.get('positions', {})),
                        'trades_count': len(trading_state['runner'].portfolio.get('trades', []))
                    }
                    
        return jsonify(status)
    except Exception as e:
        logger.exception("Error getting trading status")
        return jsonify({'error': str(e)}), 500

#################################################
# Configuration Endpoints
#################################################

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get the current configuration."""
    try:
        with trading_state['lock']:
            if not trading_state['config']:
                return jsonify({'error': 'No active configuration'}), 404
                
            return jsonify(trading_state['config'])
    except Exception as e:
        logger.exception("Error getting configuration")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['PUT'])
def update_config():
    """Update the configuration."""
    try:
        # Get parameters from request
        data = request.json
        if not data:
            return jsonify({'error': 'No configuration provided'}), 400
            
        with trading_state['lock']:
            if trading_state['is_running']:
                return jsonify({'error': 'Cannot update configuration while trading is running'}), 400
                
            # Update config
            trading_state['config'] = data
            
        return jsonify({'status': 'success', 'message': 'Configuration updated'})
    except Exception as e:
        logger.exception("Error updating configuration")
        return jsonify({'error': str(e)}), 500

#################################################
# Data Management Endpoints
#################################################

@app.route('/api/data/market', methods=['GET'])
def get_market_data():
    """Get market data for specified symbols."""
    try:
        # Get parameters from request
        symbols = request.args.get('symbols', '').split(',')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if not symbols or symbols[0] == '':
            return jsonify({'error': 'No symbols provided'}), 400
            
        # Create data provider
        with trading_state['lock']:
            config = trading_state['config'] or {}
            
        data_provider = YahooFinanceProvider(config.get('data', {}))
        
        # Fetch data
        data = data_provider.fetch_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Convert DataFrame to JSON
        data_json = data.reset_index().to_json(orient='records', date_format='iso')
        
        return Response(data_json, mimetype='application/json')
    except Exception as e:
        logger.exception("Error fetching market data")
        return jsonify({'error': str(e)}), 500

#################################################
# Portfolio Endpoints
#################################################

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Get the current portfolio status."""
    try:
        with trading_state['lock']:
            if not trading_state['runner'] or not hasattr(trading_state['runner'], 'portfolio'):
                return jsonify({'error': 'No active portfolio'}), 404
                
            portfolio = trading_state['runner'].portfolio
            
            # Format portfolio for JSON response
            portfolio_json = {
                'cash': portfolio.get('cash', 0),
                'positions': [
                    {
                        'symbol': symbol,
                        'shares': shares,
                        'value': shares * 0  # TODO: Get current price
                    } for symbol, shares in portfolio.get('positions', {}).items()
                ],
                'trades': portfolio.get('trades', [])
            }
            
            return jsonify(portfolio_json)
    except Exception as e:
        logger.exception("Error getting portfolio")
        return jsonify({'error': str(e)}), 500

#################################################
# Static files
#################################################

@app.route('/favicon.ico')
def favicon():
    """Serve favicon or return empty response to prevent 404."""
    try:
        return send_from_directory(os.path.join(app.root_path, 'static'),
                                   'favicon.ico', mimetype='image/vnd.microsoft.icon')
    except:
        # Return an empty response with 204 status (No Content)
        return '', 204

#################################################
# Debug Endpoints
#################################################

@app.route('/api/debug/registries', methods=['GET'])
def debug_registries():
    """Get debug information about the registries."""
    try:
        # Re-initialize registries if requested
        if request.args.get('reinitialize') == 'true':
            initialize_registries()
            
        # Build debug info
        debug_info = {
            'strategies': {
                'count': len(strategy_registry.get_all()),
                'items': [
                    {
                        'id': name,
                        'class': cls.__name__,
                        'description': cls.__doc__ or "No description" 
                    } for name, cls in strategy_registry.get_all().items()
                ]
            },
            'factors': {
                'count': len(factor_registry.get_all()),
                'items': [
                    {
                        'id': name,
                        'class': cls.__name__
                    } for name, cls in factor_registry.get_all().items()
                ]
            },
            'trading_state': {
                'is_running': trading_state['is_running'],
                'mode': trading_state['mode'],
                'has_config': trading_state['config'] is not None,
                'has_runner': trading_state['runner'] is not None
            },
            'import_paths': sys.path,
            'python_version': sys.version
        }
        
        return jsonify(debug_info)
    except Exception as e:
        logger.exception("Error in debug endpoint")
        return jsonify({'error': str(e)}), 500

#################################################
# Helper Functions
#################################################

def initialize_registries():
    """Initialize strategy and factor registries."""
    logger.info("Initializing registries...")
    
    try:
        # Import register functions
        from run import register_strategies, register_factors
        
        # Register strategies
        logger.info("Registering strategies...")
        register_strategies(strategy_registry)
        logger.info(f"Registered strategies: {list(strategy_registry.get_all().keys())}")
        
        # Register factors
        logger.info("Registering factors...")
        register_factors(factor_registry)
        logger.info(f"Registered factors: {list(factor_registry.get_all().keys())}")
        
        # Log success
        logger.info("Registry initialization completed successfully")
    except Exception as e:
        logger.exception(f"Error initializing registries: {str(e)}")
        
        # Try alternative approach
        logger.info("Trying alternative strategy registration...")
        try:
            # Direct registration of known strategies
            from plugins.strategies.mean_reversion import MeanReversionStrategy
            strategy_registry.register("mean_reversion", MeanReversionStrategy)
            
            # Add more strategies as needed
            # from plugins.strategies.ma_crossover import MACrossoverStrategy
            # strategy_registry.register("ma_crossover", MACrossoverStrategy)
            
            logger.info(f"Directly registered strategies: {list(strategy_registry.get_all().keys())}")
        except Exception as e2:
            logger.exception(f"Alternative registration failed: {str(e2)}")

#################################################
# Main Function
#################################################

def main(host='0.0.0.0', port=5000, debug=True):
    """Run the Flask API server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        debug: Whether to run in debug mode
    """
    # Initialize registries
    initialize_registries()
    
    # Configure server settings
    print(f"Starting StockRadar API server on {host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Run Flask app (use_reloader=False to avoid duplicate processes in debug mode)
        app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"Error starting API server: {e}")
        # Don't exit here, let the caller handle it

if __name__ == '__main__':
    import socket
    import sys
    
    # Try to start the server, if port is in use, try alternative ports
    port = 5000
    max_port_attempts = 10
    
    for attempt in range(max_port_attempts):
        try:
            main(port=port)
            break  # If successful, break the loop
        except socket.error as e:
            if 'Address already in use' in str(e) and attempt < max_port_attempts - 1:
                port += 1
                print(f"Port {port-1} is already in use, trying port {port}...")
            else:
                print(f"Failed to start server: {e}")
                if attempt == max_port_attempts - 1:
                    print(f"Could not find an available port after {max_port_attempts} attempts.")
                    print(f"Please specify a different port when calling the script.")
                    sys.exit(1)
                raise 