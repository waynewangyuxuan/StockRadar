import os
import yaml
from pathlib import Path

def get_project_root():
    """Get the absolute path to the project root directory."""
    # This assumes the script is in backend/storage_integration/config_loader.py
    current_file = Path(__file__)
    return current_file.parent.parent.parent

def load_storage_config(config_path=None):
    """
    Load storage configuration from file.
    
    Args:
        config_path (str, optional): Path to the config file. 
                                    If None, uses default location.
    
    Returns:
        dict: The storage configuration dictionary
    """
    if config_path is None:
        root_dir = get_project_root()
        config_path = os.path.join(root_dir, 'config', 'storage_config.yaml')
    
    # Create default config if file doesn't exist
    if not os.path.exists(config_path):
        create_default_config(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def create_default_config(config_path):
    """
    Create a default configuration file if one doesn't exist.
    
    Args:
        config_path (str): Path where the config file should be created
    """
    default_config = {
        'default_backend': 'local',
        'backends': {
            'local': {
                'base_path': 'data/storage'
            }
        },
        'data_types': {
            'market_data': {
                'retention_days': 90,
                'format': 'parquet',
                'compression': 'snappy'
            },
            'backtest_results': {
                'retention_days': 365,
                'format': 'parquet',
                'compression': 'snappy'
            },
            'models': {
                'retention_days': 180,
                'format': 'parquet',
                'compression': 'snappy'
            }
        }
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False) 