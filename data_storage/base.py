"""Base class for data storage implementations."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

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
    
    def validate_data(self, data):
        """Validate market data structure.
        
        Args:
            data (pd.DataFrame): Market data to validate
            
        Returns:
            bool: True if valid, False otherwise
            
        Raises:
            ValueError: If data is invalid
        """
        if data is None or data.empty:
            raise ValueError("Data cannot be empty")
            
        required_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return True 