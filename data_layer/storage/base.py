from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

class DataStorageBase(ABC):
    """Base class for all storage implementations"""
    
    def __init__(self):
        """Initialize storage"""
        self.connection = None
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to storage"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to storage"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if storage is available"""
        pass
    
    @abstractmethod
    def save_market_data(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save market data to storage
        
        Args:
            data: Market data to save
            metadata: Optional metadata about the data
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get market data from storage
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            fields: Optional list of fields to retrieve
            
        Returns:
            Dict containing the market data and metadata
        """
        pass
    
    @abstractmethod
    def get_latest_data(self, symbols: List[str], fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get latest market data from storage
        
        Args:
            symbols: List of stock symbols
            fields: Optional list of fields to retrieve
            
        Returns:
            Dict containing the latest market data and metadata
        """
        pass
    
    @abstractmethod
    def delete_market_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        """Delete market data from storage
        
        Args:
            symbols: Optional list of stock symbols to delete data for
            start_date: Optional start date for deletion range
            end_date: Optional end date for deletion range
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        pass 