import argparse
import os
import sys
import logging
from datetime import datetime

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.signal_generator import SignalGenerator
from config.config_loader import ConfigLoader

def setup_logger(log_file=None):
    """Configure logger"""
    logger = logging.getLogger('weekly_signal')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    
    # Set log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    if log_file:
        file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run weekly signal generation task')
    parser.add_argument('--config', required=True,
                      help='Configuration file path')
    parser.add_argument('--date', required=True,
                      help='Signal generation date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger()
    logger.info("Starting weekly signal generation task")
    
    # Create output directory
    output_dir = os.path.join('output', 'signals', args.date)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run strategy
    config = ConfigLoader.load(args.config)
    generator = SignalGenerator(config)
    generator.generate_signals(args.date, output_dir)
    
    logger.info("Weekly signal generation task completed")

if __name__ == '__main__':
    main() 