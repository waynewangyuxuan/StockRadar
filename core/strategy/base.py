from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
            parameters: Dictionary of strategy parameters
        """
        self.name = name
        self.parameters = parameters
        self.signals = None
        self.positions = None
        
    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals based on the strategy logic.
        
        Args:
            data: DataFrame containing price and technical indicators
            
        Returns:
            DataFrame with calculated signals
        """
        pass
    
    @abstractmethod
    def generate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Generate position recommendations based on signals.
        
        Args:
            signals: DataFrame containing trading signals
            
        Returns:
            DataFrame with position recommendations
        """
        pass
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            bool: True if parameters are valid, False otherwise
        """
        return True
    
    def get_required_indicators(self) -> List[str]:
        """
        Get list of required technical indicators for the strategy.
        
        Returns:
            List of indicator names
        """
        return []
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """
        Update strategy parameters.
        
        Args:
            new_parameters: New parameter values
        """
        self.parameters.update(new_parameters)
        if not self.validate_parameters():
            raise ValueError("Invalid parameters provided")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and current parameters.
        
        Returns:
            Dictionary containing strategy information
        """
        return {
            "name": self.name,
            "parameters": self.parameters,
            "required_indicators": self.get_required_indicators()
        } 