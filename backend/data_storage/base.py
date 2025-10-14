"""Base class for data storage implementations."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd

class DataStorageBase(ABC):
    """Base class for data storage implementations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the storage.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def save_data(self, data, dataset_name, version=None):
        """Save market data to storage.
        
        Args:
            data (pd.DataFrame): Market data to save
            dataset_name (str): Name of the dataset
            version (str, optional): Version identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_data(self, dataset_name, tickers=None, start_date=None, end_date=None, version=None):
        """Load market data from storage.
        
        Args:
            dataset_name (str): Name of the dataset
            tickers (list, optional): List of tickers to load
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            version (str, optional): Version identifier
            
        Returns:
            pd.DataFrame: Loaded market data
        """
        pass
    
    @abstractmethod
    def delete_data(self, dataset_name, version=None):
        """Delete market data from storage.
        
        Args:
            dataset_name (str): Name of the dataset
            version (str, optional): Version identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def list_datasets(self):
        """List all available datasets.
        
        Returns:
            list: List of dataset names
        """
        pass
    
    @abstractmethod
    def get_versions(self, dataset_name):
        """Get available versions for a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            list: List of version identifiers
        """
        pass
    
    def validate_data(self, data, dataset_type="market_data"):
        """Validate data structure based on the dataset type.
        
        Args:
            data (pd.DataFrame): Data to validate
            dataset_type (str): Type of dataset to validate against
            
        Returns:
            bool: True if valid, False otherwise
            
        Raises:
            ValueError: If data is invalid
        """
        if data is None or data.empty:
            raise ValueError("Data cannot be empty")
        
        # Different validation rules for different dataset types
        # Only apply market data validation for raw market data and processed data
        if dataset_type == "market_data" or dataset_type == "raw_market_data" or dataset_type == "processed_data":
            self._validate_market_data(data)
        # For other types (like equity_curve, metrics, trades), no additional validation needed
            
        return True
    
    def _validate_market_data(self, data):
        """Validate market data specific structure.
        
        Args:
            data (pd.DataFrame): Market data to validate
            
        Raises:
            ValueError: If market data is invalid
        """
        required_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        
        # If data has a MultiIndex, check if required columns are in the index before flagging as missing
        if isinstance(data.index, pd.MultiIndex):
            index_names = list(data.index.names)
            for col in ['ticker', 'date']:
                if col in index_names and col not in data.columns:
                    # This column is in the index, so it's not really missing
                    required_columns.remove(col)
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for nulls in key columns
        for col in ['ticker', 'date']:
            if col in data.columns and data[col].isnull().any():
                raise ValueError(f"Column '{col}' contains null values") 