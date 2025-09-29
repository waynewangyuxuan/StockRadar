from typing import Dict, Type, Optional
import logging

class FactorRegistry:
    """Registry for managing trading factors."""
    
    def __init__(self):
        """Initialize the factor registry."""
        self._factors: Dict[str, Type] = {}
        self.logger = logging.getLogger(__name__)
        
    def register(self, name: str, factor_class: Type) -> None:
        """Register a factor class.
        
        Args:
            name: Name of the factor
            factor_class: Factor class to register
        """
        if name in self._factors:
            self.logger.warning(f"Factor {name} already registered, overwriting")
        self._factors[name] = factor_class
        self.logger.info(f"Registered factor: {name}")
        
    def get(self, name: str) -> Optional[Type]:
        """Get a factor class by name.
        
        Args:
            name: Name of the factor
            
        Returns:
            Factor class if found, None otherwise
        """
        return self._factors.get(name)
        
    def list_factors(self) -> list[str]:
        """Get list of registered factor names.
        
        Returns:
            List of factor names
        """
        return list(self._factors.keys())
        
    def create_factor(self, name: str, config: dict) -> Optional[object]:
        """Create a factor instance.
        
        Args:
            name: Name of the factor
            config: Factor configuration
            
        Returns:
            Factor instance if found, None otherwise
        """
        factor_class = self.get(name)
        if factor_class is None:
            self.logger.error(f"Factor {name} not found")
            return None
            
        try:
            return factor_class(config)
        except Exception as e:
            self.logger.error(f"Failed to create factor {name}: {str(e)}")
            return None
            
    def get_all(self) -> Dict[str, Type]:
        """Get all registered factors.

        Returns:
            Dictionary of factor name to factor class mappings
        """
        return self._factors.copy()

    def clear(self) -> None:
        """Clear all registered factors."""
        self._factors.clear()

# Global registry instance
registry = FactorRegistry() 