#!/bin/bash
# Script to start the StockRadar API server

# Default values
PORT=5000
HOST="0.0.0.0"
MODE="api"
CONFIG=""
TRADING_MODE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --host=*)
      HOST="${1#*=}"
      shift
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
    --config=*)
      CONFIG="${1#*=}"
      shift
      ;;
    --trading-mode=*)
      TRADING_MODE="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--port=PORT] [--host=HOST] [--mode=api|cli] [--config=CONFIG_PATH] [--trading-mode=live|backtest]"
      exit 1
      ;;
  esac
done

# Construct command
CMD="python main.py --mode=$MODE --port=$PORT --host=$HOST"

if [ ! -z "$CONFIG" ]; then
  CMD="$CMD --config=$CONFIG"
fi

if [ ! -z "$TRADING_MODE" ]; then
  CMD="$CMD --trading-mode=$TRADING_MODE"
fi

# Print info
echo "Starting StockRadar in $MODE mode"
echo "Command: $CMD"

# Execute
exec $CMD 