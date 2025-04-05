from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
from datetime import datetime

class DataProcessor(ABC):
    """Base class for all data processors"""
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        enabled: bool = True
    ):
        """Initialize data processor
        
        Args:
            name: Name of the processor
            description: Optional description of what the processor does
            enabled: Whether the processor is enabled
        """
        self.name = name
        self.description = description or f"Data processor: {name}"
        self.enabled = enabled
        self.metrics = {
            'processed_rows': 0,
            'errors': 0,
            'processing_time': 0.0
        }
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the input data
        
        Args:
            data: Input DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        pass
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate input data
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics
        
        Returns:
            Dict containing processing metrics
        """
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset processing metrics"""
        self.metrics = {
            'processed_rows': 0,
            'errors': 0,
            'processing_time': 0.0
        }
    
    def __str__(self) -> str:
        """String representation of the processor"""
        return f"{self.name} ({'enabled' if self.enabled else 'disabled'})" 