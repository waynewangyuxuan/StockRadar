"""Base strategy class defining the basic interface for strategy execution"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import os

class StrategyBase(ABC):
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """Initialize strategy
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.factors: List[Any] = []  # List of factors used by the strategy
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals
        
        Args:
            data: Input data containing factor values
            
        Returns:
            pd.DataFrame: Signal DataFrame containing signal values and related metadata
        """
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate if input data meets strategy requirements
        
        Args:
            data: Input data
            
        Returns:
            bool: Whether the data is valid
        """
        pass
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Data preprocessing
        
        Args:
            data: Input data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        return data
    
    def postprocess(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Signal post-processing
        
        Args:
            signals: Generated signals
            
        Returns:
            pd.DataFrame: Post-processed signals
        """
        return signals
    
    def add_factor(self, factor: Any) -> None:
        """Add factor to strategy
        
        Args:
            factor: Factor instance
        """
        self.factors.append(factor)
    
    def get_required_fields(self) -> List[str]:
        """Get required data fields
        
        Returns:
            List[str]: List of required fields
        """
        fields = []
        for factor in self.factors:
            fields.extend(factor.get_required_fields())
        return list(set(fields))
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information
        
        Returns:
            Dict[str, Any]: Strategy information dictionary
        """
        return {
            'name': self.name,
            'params': self.params,
            'factors': [f.get_factor_info() for f in self.factors]
        }
    
    def save_signals(self, signals: pd.DataFrame, output_path: str) -> None:
        """Save signals to file
        
        Args:
            signals: Signal DataFrame
            output_path: Output file path
        """
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, f"{self.name}_signals.csv")
        signals.to_csv(file_path, index=False) 