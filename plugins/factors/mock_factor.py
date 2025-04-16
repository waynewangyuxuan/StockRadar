from typing import Dict, Any
import pandas as pd
from core.factor_base import FactorBase

class MockFactor(FactorBase):
    """Mock factor for testing purposes."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the mock factor."""
        super().__init__(config)
        self.short_window = config.get('short_window', 5)
        self.long_window = config.get('long_window', 20)

    def _get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the factor."""
        return {
            'name': 'mock_factor',
            'description': 'Mock factor for testing',
            'parameters': {
                'short_window': 'Short window length',
                'long_window': 'Long window length'
            }
        }

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mock factor values."""
        # Add a simple mock column for testing
        data['mock_factor'] = data['close'].rolling(window=self.short_window).mean()
        return data 