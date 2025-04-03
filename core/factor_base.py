from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class FactorBase(ABC):
    """Base factor class that defines the basic interface for factor calculation"""
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the factor
        
        Args:
            name: Factor name
            params: Factor parameters
        """
        self.name = name
        self.params = params or {}
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor values
        
        Args:
            data: Input data containing necessary market data
            
        Returns:
            pd.Series: Factor value series
        """
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate if input data meets factor calculation requirements
        
        Args:
            data: Input data
            
        Returns:
            bool: Whether the data is valid
        """
        pass
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Data preprocessing
        
        Args:
            data: Input data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        return data
    
    def postprocess(self, factor: pd.Series) -> pd.Series:
        """
        Factor post-processing (e.g., standardization, outlier removal)
        
        Args:
            factor: Original factor values
            
        Returns:
            pd.Series: Processed factor values
        """
        return factor
    
    def get_required_fields(self) -> list:
        """
        Get the list of fields required for factor calculation
        
        Returns:
            list: List of required fields
        """
        return []
    
    def get_factor_info(self) -> Dict[str, Any]:
        """
        Get factor information
        
        Returns:
            Dict[str, Any]: Factor information dictionary
        """
        return {
            'name': self.name,
            'params': self.params,
            'required_fields': self.get_required_fields()
        } 