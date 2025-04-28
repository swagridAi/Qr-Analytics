"""
Signal Generator Registry

This module provides a central registry for signal generators, allowing dynamic
loading and creation of generators by name. It implements the registry pattern
to decouple generator definition from usage.

Features:
    - Register signal generators by name
    - Create generator instances by name
    - List available generators
    - Get information about registered generators
"""

# Standard library imports
import logging
from typing import Dict, List, Type, Any

# Local imports
from .signal_generator import SignalGenerator

# Configure logger
logger = logging.getLogger("quant_research.analytics")


class SignalGeneratorRegistry:
    """
    Registry for signal generators.
    
    This class provides a central registry for signal generators, allowing
    dynamic loading and creation of generators by name. It implements the
    registry pattern to decouple generator definition from usage.
    
    Usage:
        # Register a generator
        SignalGeneratorRegistry.register('volatility', VolatilitySignalGenerator)
        
        # Create a generator instance by name
        volatility_generator = SignalGeneratorRegistry.create('volatility', window=21)
    """
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, generator_class: Type[SignalGenerator]) -> None:
        """
        Register a signal generator class.
        
        Args:
            name: Name to register the generator under
            generator_class: SignalGenerator class
            
        Returns:
            None
            
        Raises:
            TypeError: If the class does not inherit from SignalGenerator
        """
        if not issubclass(generator_class, SignalGenerator):
            raise TypeError(f"Class {generator_class.__name__} must inherit from SignalGenerator")
            
        cls._registry[name] = generator_class
        logger.debug(f"Registered signal generator: {name}")
    
    @classmethod
    def create(cls, name: str, **kwargs) -> SignalGenerator:
        """
        Create a signal generator instance by name.
        
        Args:
            name: Name of the generator to create
            **kwargs: Parameters to pass to the generator constructor
            
        Returns:
            SignalGenerator instance
            
        Raises:
            ValueError: If the generator name is not registered
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown signal generator: {name}. Available generators: {', '.join(cls._registry.keys())}")
            
        generator_class = cls._registry[name]
        return generator_class(**kwargs)
    
    @classmethod
    def list_generators(cls) -> List[str]:
        """
        List all registered generator names.
        
        Returns:
            List of registered generator names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_info(cls, name: str = None) -> Dict[str, Any]:
        """
        Get information about registered generators.
        
        Args:
            name: Optional name of specific generator to get info for
            
        Returns:
            Dictionary with generator information
            
        Raises:
            ValueError: If the specified generator name is not registered
        """
        if name is not None:
            if name not in cls._registry:
                raise ValueError(f"Unknown signal generator: {name}")
                
            generator_class = cls._registry[name]
            return {
                "name": name,
                "class": generator_class.__name__,
                "module": generator_class.__module__,
                "description": generator_class.__doc__.split('\n')[0] if generator_class.__doc__ else "No description"
            }
        else:
            return {
                name: {
                    "class": generator_class.__name__,
                    "module": generator_class.__module__,
                    "description": generator_class.__doc__.split('\n')[0] if generator_class.__doc__ else "No description"
                }
                for name, generator_class in cls._registry.items()
            }