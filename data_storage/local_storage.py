"""Local file-based storage implementation."""

import os
import pandas as pd
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from .base import DataStorageBase

class LocalStorage(DataStorageBase):
    """Local file-based storage using Parquet format.
    
    This implementation stores data in Parquet files organized by dataset and version.
    Directory structure:
    storage_root/
        dataset_name/
            data.parquet (main data file)
            versions/
                version_id.parquet (versioned data files)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize local storage.
        
        Args:
            config: Dictionary containing storage configuration
                   Required keys:
                   - storage_path: Base directory for data storage
        """
        super().__init__()
        self.config = config or {}
        self.storage_path = self.config.get('storage_path', 'storage')
        os.makedirs(self.storage_path, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _get_dataset_path(self, dataset_name: str) -> str:
        """Get the path for a dataset directory."""
        return os.path.join(self.storage_path, dataset_name)
        
    def _get_version_path(self, dataset_name: str, version: str) -> str:
        """Get the path for a versioned data file."""
        version_dir = os.path.join(self._get_dataset_path(dataset_name), 'versions')
        os.makedirs(version_dir, exist_ok=True)
        return os.path.join(version_dir, f"{version}.parquet")
        
    def _get_main_data_path(self, dataset_name: str) -> str:
        """Get the path for the main data file."""
        dataset_dir = self._get_dataset_path(dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        return os.path.join(dataset_dir, "data.parquet")
    
    def save_data(self, 
                 data: pd.DataFrame,
                 dataset_name: str,
                 version: Optional[str] = None) -> bool:
        """Save market data to local storage.
        
        Args:
            data: DataFrame containing market data
            dataset_name: Name of the dataset
            version: Optional version identifier
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Validate data first
            self.validate_data(data)
            
            # Save data
            if version:
                path = self._get_version_path(dataset_name, version)
            else:
                path = self._get_main_data_path(dataset_name)
            
            data.to_parquet(path)
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            return False
    
    def load_data(self,
                 dataset_name: str,
                 tickers: Optional[List[str]] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 version: Optional[str] = None) -> pd.DataFrame:
        """Load market data from local storage.
        
        Args:
            dataset_name: Name of the dataset
            tickers: Optional list of tickers to load
            start_date: Optional start date for filtering (YYYY-MM-DD)
            end_date: Optional end date for filtering (YYYY-MM-DD)
            version: Optional version to load
            
        Returns:
            DataFrame containing the requested market data
        """
        try:
            # Get data path
            if version:
                path = self._get_version_path(dataset_name, version)
            else:
                path = self._get_main_data_path(dataset_name)
            
            # Check if file exists
            if not os.path.exists(path):
                return pd.DataFrame()
            
            # Load data
            data = pd.read_parquet(path)
            
            # Apply filters
            if tickers:
                data = data[data['ticker'].isin(tickers)]
            
            if start_date:
                data = data[data['date'] >= pd.Timestamp(start_date)]
            
            if end_date:
                data = data[data['date'] <= pd.Timestamp(end_date)]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def delete_data(self,
                   dataset_name: str,
                   version: Optional[str] = None) -> bool:
        """Delete market data from local storage.
        
        Args:
            dataset_name: Name of the dataset
            version: Optional version to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if version:
                # Delete specific version
                path = self._get_version_path(dataset_name, version)
                if os.path.exists(path):
                    os.remove(path)
            else:
                # Delete entire dataset
                path = self._get_dataset_path(dataset_name)
                if os.path.exists(path):
                    import shutil
                    shutil.rmtree(path)
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting data: {e}")
            return False
    
    def list_datasets(self) -> List[str]:
        """List all available datasets.
        
        Returns:
            List of dataset names
        """
        try:
            return [d for d in os.listdir(self.storage_path)
                   if os.path.isdir(os.path.join(self.storage_path, d))]
        except Exception as e:
            self.logger.error(f"Error listing datasets: {e}")
            return []
    
    def get_versions(self, dataset_name: str) -> List[str]:
        """Get available versions for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of version identifiers
        """
        try:
            version_dir = os.path.join(self._get_dataset_path(dataset_name), 'versions')
            if not os.path.exists(version_dir):
                return []
            
            return [os.path.splitext(f)[0] for f in os.listdir(version_dir)
                   if f.endswith('.parquet')]
        except Exception as e:
            self.logger.error(f"Error getting versions: {e}")
            return []
