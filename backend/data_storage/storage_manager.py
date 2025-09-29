"""
Unified Storage Manager for the StockRadar system
Provides a centralized interface for data storage and retrieval operations

This consolidated implementation replaces:
- backend/core/storage_integration.py 
- backend/storage_integration/storage_manager.py
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import numpy as np

# Import storage backends
from .base import DataStorageBase
from .local_storage import LocalStorage
from .redis_cache import RedisCache
from .timescaledb_storage import TimescaleDBStorage
from .version_control import VersionControl

# Import for config loading
import yaml

logger = logging.getLogger(__name__)

def get_project_root():
    """Get the absolute path to the project root directory."""
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
        logger.warning(f"Storage config not found at {config_path}")
        return {
            'storage': {
                'backend': 'local',
                'local': {
                    'base_path': 'data/storage'
                }
            },
            'data_types': {
                'market_data': {
                    'path': 'market_data/',
                    'format': 'parquet',
                    'retention': {
                        'days': 90,
                        'count': 100
                    }
                },
                'backtest_results': {
                    'path': 'backtest_results/',
                    'format': 'parquet',
                    'retention': {
                        'days': 365,
                        'count': 50
                    }
                }
            }
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

class StorageManager:
    """
    Unified Storage Manager that provides a central interface for all data storage operations
    
    Manages:
    - Multiple backend storage systems (local, redis, timescale, S3, GCS)
    - Data versioning and retrieval
    - Data retention policies
    - Metadata handling
    
    This class consolidates functionality from multiple pre-existing storage managers.
    """
    
    def __init__(self, config: Union[Dict[str, Any], str, None] = None):
        """
        Initialize the storage manager with configuration.
        
        Args:
            config: Either a configuration dictionary, a path to a config file,
                   or None to use the default config file
        """
        # Load configuration
        if config is None:
            self.config = load_storage_config()
        elif isinstance(config, str):
            self.config = load_storage_config(config)
        else:
            self.config = config
            
        # Extract storage config
        self.storage_config = self.config.get('storage', {})
        if not self.storage_config:
            # Fall back to pre-existing format
            self.storage_config = self.config
            
        # Initialize storage backends
        self.storage_backends = {}
        self.primary_backend = None
        self.version_control = None
        self._initialize_backends()
        
    def _initialize_backends(self):
        """Initialize storage backends based on configuration."""
        backend_type = self.storage_config.get('backend', 'local')
        
        # Initialize LocalStorage
        if backend_type == 'local' or 'local' in self.storage_config:
            local_config = self.storage_config.get('local', {})
            base_path = local_config.get('base_path', 'data/storage')
            # Convert to absolute path if relative
            if not os.path.isabs(base_path):
                base_path = os.path.join(get_project_root(), base_path)
                
            try:
                local_storage = LocalStorage({'storage_path': base_path})
                self.storage_backends['local'] = local_storage
                logger.info(f"LocalStorage initialized with base_path={base_path}")
                
                # Set as default if no primary backend specified or if it's the chosen backend
                if self.primary_backend is None or backend_type == 'local':
                    self.primary_backend = local_storage
            except Exception as e:
                logger.error(f"Failed to initialize LocalStorage: {str(e)}")
        
        # Initialize RedisCache if available and enabled
        if 'redis' in self.storage_config and self.storage_config.get('redis', {}).get('enabled', False):
            try:
                redis_config = self.storage_config.get('redis', {})
                redis_cache = RedisCache(redis_config)
                self.storage_backends['redis'] = redis_cache
                logger.info(f"RedisCache initialized")
                
                # Set as primary if specified
                if backend_type == 'redis':
                    self.primary_backend = redis_cache
            except Exception as e:
                logger.error(f"Failed to initialize RedisCache: {str(e)}")
        
        # Initialize TimescaleDBStorage if available and enabled
        if 'timescaledb' in self.storage_config and self.storage_config.get('timescaledb', {}).get('enabled', False):
            try:
                ts_config = self.storage_config.get('timescaledb', {})
                ts_storage = TimescaleDBStorage(ts_config)
                self.storage_backends['timescaledb'] = ts_storage
                logger.info(f"TimescaleDBStorage initialized")
                
                # Set as primary if specified
                if backend_type == 'timescaledb':
                    self.primary_backend = ts_storage
            except Exception as e:
                logger.error(f"Failed to initialize TimescaleDBStorage: {str(e)}")
        
        # Initialize VersionControl with primary backend
        if self.primary_backend:
            try:
                self.version_control = VersionControl(self.primary_backend)
                logger.info(f"VersionControl initialized with {backend_type} backend")
            except Exception as e:
                logger.error(f"Failed to initialize VersionControl: {str(e)}")
                
    def _get_data_type_config(self, data_type):
        """Get configuration for a specific data type."""
        data_types = self.config.get('data_types', {})
        if data_type not in data_types:
            # Default config if data type not explicitly defined
            return {
                'format': 'parquet',
                'retention': {
                    'days': 90,
                    'count': 100
                }
            }
        return data_types[data_type]
        
    def _prepare_data_for_storage(self, data: pd.DataFrame, dataset_type: str = "market_data") -> pd.DataFrame:
        """
        Prepare data for storage by ensuring it has required columns.
        
        Args:
            data: DataFrame to prepare
            dataset_type: Type of dataset (market_data, portfolio, etc.)
            
        Returns:
            DataFrame ready for storage
        """
        # First check if we have a MultiIndex
        if isinstance(data.index, pd.MultiIndex):
            # If we have a MultiIndex, reset it to get the index levels as columns
            data_for_storage = data.reset_index()
            logger.info(f"Reset MultiIndex to convert index levels to columns for storage")
        else:
            # Create a copy to avoid modifying the original
            data_for_storage = data.copy()
        
        if dataset_type == "market_data":
            required_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
            
            # Check if data meets requirements for market data
            missing_columns = [col for col in required_columns if col not in data_for_storage.columns]
            
            if missing_columns:
                # If data doesn't have required columns, don't try to validate as market data
                # Instead, enhance the dataframe with the missing required columns
                
                # Add index levels as columns if they match required columns and are missing
                if hasattr(data.index, 'names'):
                    for idx_name in data.index.names:
                        if idx_name in missing_columns and idx_name is not None:
                            # Extract index level to column
                            if isinstance(data.index, pd.MultiIndex):
                                # For MultiIndex, get the level position
                                idx_pos = data.index.names.index(idx_name)
                                data_for_storage[idx_name] = data.index.get_level_values(idx_pos)
                            else:
                                # For regular Index
                                data_for_storage[idx_name] = data.index.values
                
                # For other missing columns, add dummy values
                for col in missing_columns:
                    if col not in data_for_storage.columns:
                        if col == 'ticker':
                            data_for_storage[col] = 'UNKNOWN'
                        elif col == 'date':
                            data_for_storage[col] = pd.Timestamp.now()
                        elif col in ['open', 'high', 'low', 'close', 'volume']:
                            data_for_storage[col] = 0.0
                
                logger.info(f"Added missing columns {missing_columns} to data for storage")
            
        return data_for_storage
    
    def store_market_data(self, data: pd.DataFrame, metadata: Dict[str, Any] = None) -> str:
        """
        Store market data with versioning.
        
        Args:
            data: Market data DataFrame
            metadata: Optional metadata about the data
            
        Returns:
            Version ID of the stored data
        """
        if self.version_control is None:
            logger.warning("VersionControl not initialized, using fallback")
            if not self.primary_backend:
                # Create a fallback local storage
                base_path = os.path.join(get_project_root(), 'data/storage')
                self.primary_backend = LocalStorage({'storage_path': base_path})
                self.version_control = VersionControl(self.primary_backend)
        
        # Prepare metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # Prepare the data for storage
        prepared_data = self._prepare_data_for_storage(data, "market_data")
        
        try:
            # Store data with version control
            version_id, success = self.version_control.create_version(
                prepared_data, 
                "market_data", 
                metadata
            )
            
            if success:
                logger.info(f"Stored market data with version_id={version_id}")
            else:
                logger.error("Failed to store market data")
                
            return version_id
        except Exception as e:
            logger.error(f"Failed to store market data: {str(e)}")
            # Still return a None version_id so the code doesn't break
            return None
    
    def store_factor_data(self, factor_data: Dict[str, pd.DataFrame], metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Store factor data with versioning.
        
        Args:
            factor_data: Dictionary mapping factor names to DataFrames
            metadata: Optional metadata about the data
            
        Returns:
            Dictionary mapping factor names to version IDs
        """
        version_ids = {}
        
        for factor_name, data in factor_data.items():
            # Prepare factor-specific metadata
            factor_metadata = metadata.copy() if metadata else {}
            factor_metadata["factor_name"] = factor_name
            factor_metadata["timestamp"] = datetime.now().isoformat()
            
            # Store factor data
            version_id = self.store_market_data(data, factor_metadata)
            version_ids[factor_name] = version_id
            
        return version_ids
    
    def store_backtest_results(self, results, strategy_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Store backtest results with versioning.
        
        Args:
            results: Either a dictionary of backtest results or an EvaluationResult object
            strategy_name: Name of the strategy
            metadata: Optional metadata about the results
            
        Returns:
            Version ID of the stored results
        """
        # Prepare metadata
        result_metadata = metadata.copy() if metadata else {}
        result_metadata["strategy_name"] = strategy_name
        result_metadata["timestamp"] = datetime.now().isoformat()
        
        # Handle different result types (dict or EvaluationResult)
        if hasattr(results, 'equity_curve') and hasattr(results, 'metrics') and hasattr(results, 'trades'):
            # It's an EvaluationResult object
            equity_df = results.equity_curve
            metrics_df = pd.DataFrame([results.metrics])
            
            # Ensure trades are in a proper DataFrame format
            if results.trades:
                # Convert all values to basic Python types
                sanitized_trades = []
                for trade in results.trades:
                    sanitized_trade = {}
                    for k, v in trade.items():
                        if isinstance(v, (pd.Timestamp, datetime, date)):
                            sanitized_trade[k] = v.isoformat()
                        elif isinstance(v, (np.integer, np.floating)):
                            sanitized_trade[k] = float(v)
                        else:
                            sanitized_trade[k] = v
                    sanitized_trades.append(sanitized_trade)
                trades_df = pd.DataFrame(sanitized_trades)
            else:
                # Create an empty DataFrame with expected columns
                trades_df = pd.DataFrame(columns=[
                    'ticker', 'entry_date', 'exit_date', 'position', 
                    'entry_price', 'exit_price', 'pnl', 'return_pct'
                ])
        else:
            # Assume it's a dictionary
            # Convert Dict results to DataFrame for storage
            # Equity curve is already a DataFrame
            equity_df = results.get("equity_curve", pd.DataFrame())
            
            # Store metrics as DataFrame
            metrics_df = pd.DataFrame([results.get("metrics", {})])
            
            # Store trades as DataFrame
            trades_list = results.get("trades", [])
            if trades_list:
                trades_df = pd.DataFrame(trades_list)
            else:
                trades_df = pd.DataFrame(columns=[
                    'ticker', 'entry_date', 'exit_date', 'position', 
                    'entry_price', 'exit_price', 'pnl', 'return_pct'
                ])
        
        # Convert any date objects in metadata to string
        result_metadata = self._convert_dates_to_str(result_metadata)
        
        # Handle empty DataFrames by creating minimal dummy data
        current_time = pd.Timestamp.now()
        
        # Ensure equity_df is not empty
        if equity_df.empty:
            logger.info(f"Creating dummy equity curve data for {strategy_name}")
            equity_df = pd.DataFrame({
                'date': [current_time],
                'equity': [100.0],
                'strategy': [strategy_name]
            })
        
        # Ensure metrics_df is not empty
        if metrics_df.empty:
            logger.info(f"Creating dummy metrics data for {strategy_name}")
            metrics_df = pd.DataFrame({
                'date': [current_time],
                'strategy': [strategy_name],
                'empty_result': [True]
            })
        
        # Ensure trades_df is not empty
        if trades_df.empty:
            logger.info(f"Creating dummy trades data for {strategy_name}")
            trades_df = pd.DataFrame({
                'date': [current_time],
                'ticker': ['NONE'],
                'strategy': [strategy_name],
                'empty_result': [True]
            })
        
        try:
            # Store each component using the generic method
            equity_id = self.store_generic_data(equity_df, "equity_curve", {**result_metadata, "component": "equity_curve"})
            metrics_id = self.store_generic_data(metrics_df, "metrics", {**result_metadata, "component": "metrics"})
            trades_id = self.store_generic_data(trades_df, "trades", {**result_metadata, "component": "trades"})
            
            # Store reference to all components
            components = {
                "equity_curve": equity_id,
                "metrics": metrics_id,
                "trades": trades_id
            }
            
            # Create index entry for the complete backtest
            index_metadata = {**result_metadata, "components": components}
            if self.version_control:
                # Prepare a simple DataFrame for the index
                index_df = pd.DataFrame({'strategy': [strategy_name]})
                
                # Add a date column to make the storage system happy
                index_df['date'] = pd.Timestamp.now()
                
                index_id, success = self.version_control.create_version(
                    index_df, 
                    f"backtest_{strategy_name}",
                    index_metadata
                )
                if success:
                    logger.info(f"Stored backtest results for {strategy_name} with version_id={index_id}")
                    return index_id
                else:
                    logger.error(f"Failed to store backtest index for {strategy_name}")
            else:
                logger.error("Failed to store backtest index, VersionControl not initialized")
            
            return None
        except Exception as e:
            logger.error(f"Error storing backtest results: {str(e)}")
            return None
    
    def _convert_dates_to_str(self, data):
        """Convert date objects in a dictionary to strings."""
        if isinstance(data, dict):
            return {k: self._convert_dates_to_str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_dates_to_str(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, date):
            return data.isoformat()
        elif isinstance(data, timedelta):
            return str(data)
        else:
            return data
    
    def get_market_data(self, version_id: str = None, latest: bool = True) -> pd.DataFrame:
        """
        Retrieve market data by version ID or get latest.
        
        Args:
            version_id: Optional version ID to retrieve
            latest: Whether to get the latest version (if version_id not specified)
            
        Returns:
            DataFrame containing the requested market data
        """
        if self.version_control is None:
            logger.error("VersionControl not initialized")
            return pd.DataFrame()
        
        try:    
            if version_id:
                # Get specific version
                data, metadata = self.version_control.get_version("market_data", version_id)
            elif latest:
                # Get latest version
                versions = self.version_control.list_versions("market_data")
                if not versions:
                    logger.warning("No market data versions found")
                    return pd.DataFrame()
                    
                latest_version = versions[0]  # Already sorted newest first
                data, metadata = self.version_control.get_version("market_data", latest_version['version_id'])
            else:
                logger.error("Either version_id or latest=True must be specified")
                return pd.DataFrame()
                
            return data
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return pd.DataFrame()
    
    def get_backtest_results(self, strategy_name: str, version_id: str = None) -> Dict[str, Any]:
        """
        Retrieve backtest results for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            version_id: Optional version ID (if not specified, get latest)
            
        Returns:
            Dictionary containing the backtest results
        """
        if self.version_control is None:
            logger.error("VersionControl not initialized")
            return {}
        
        try:
            # Get backtest index
            if version_id:
                index_df, index_metadata = self.version_control.get_version(f"backtest_{strategy_name}", version_id)
            else:
                # Get latest version
                versions = self.version_control.list_versions(f"backtest_{strategy_name}")
                if not versions:
                    logger.warning(f"No backtest results found for strategy {strategy_name}")
                    return {}
                    
                latest_version = versions[0]  # Already sorted newest first
                index_df, index_metadata = self.version_control.get_version(f"backtest_{strategy_name}", latest_version['version_id'])
            
            # Extract component IDs
            components = index_metadata.get("components", {})
            
            # Retrieve each component
            results = {}
            
            if "equity_curve" in components:
                equity_df, _ = self.version_control.get_version("market_data", components["equity_curve"])
                results["equity_curve"] = equity_df
                
            if "metrics" in components:
                metrics_df, _ = self.version_control.get_version("market_data", components["metrics"])
                # Convert metrics DataFrame to dict
                results["metrics"] = metrics_df.iloc[0].to_dict() if not metrics_df.empty else {}
                
            if "trades" in components:
                trades_df, _ = self.version_control.get_version("market_data", components["trades"])
                # Convert trades DataFrame to list of dicts
                results["trades"] = trades_df.to_dict(orient="records")
                
            return results
        except Exception as e:
            logger.error(f"Error getting backtest results: {str(e)}")
            return {}
    
    def list_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
            List of dataset names
        """
        if self.primary_backend:
            return self.primary_backend.list_datasets()
        return []
    
    def get_versions(self, dataset_name: str) -> List[Dict]:
        """
        Get available versions for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of version metadata
        """
        if self.version_control:
            return self.version_control.list_versions(dataset_name)
        return []
    
    # Legacy compatibility methods
    
    def save(self, data, data_type, identifier, metadata=None):
        """Legacy method for saving data (compatibility with old API)"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        # Add identifier to metadata
        meta = metadata or {}
        meta['identifier'] = identifier
        
        # Use new versioning system
        return self.store_market_data(data, meta)
    
    def load(self, data_type, identifier=None, date_range=None, latest=True):
        """Legacy method for loading data (compatibility with old API)"""
        # Map to new version-based system
        return self.get_market_data(None, latest)
    
    def list_data(self, data_type, identifier=None):
        """Legacy method for listing data (compatibility with old API)"""
        return [v['version_id'] for v in self.get_versions(data_type)]
    
    def store_generic_data(self, data: pd.DataFrame, dataset_type: str, metadata: Dict[str, Any] = None) -> str:
        """
        Store generic DataFrame with versioning without enforcing market data schema.
        
        Args:
            data: DataFrame to store
            dataset_type: Type of dataset (e.g., "portfolio_data", "trade_data", "metrics")
            metadata: Optional metadata about the data
            
        Returns:
            Version ID of the stored data
        """
        if self.version_control is None:
            logger.warning("VersionControl not initialized, using fallback")
            if not self.primary_backend:
                # Create a fallback local storage
                base_path = os.path.join(get_project_root(), 'data/storage')
                self.primary_backend = LocalStorage({'storage_path': base_path})
                self.version_control = VersionControl(self.primary_backend)
        
        # Prepare metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # Convert any date objects in metadata to string
        metadata = self._convert_dates_to_str(metadata)
        
        # Prepare the data for storage, ensuring proper column format
        prepared_data = self._prepare_data_for_storage(data, dataset_type)
        
        try:
            # Store data with version control
            version_id, success = self.version_control.create_version(
                prepared_data, 
                dataset_type, 
                metadata
            )
            
            if success:
                logger.info(f"Stored {dataset_type} with version_id={version_id}")
            else:
                logger.error(f"Failed to store {dataset_type}")
                
            return version_id
        except Exception as e:
            logger.error(f"Failed to store {dataset_type}: {str(e)}")
            return None 