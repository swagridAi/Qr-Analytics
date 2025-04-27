# src/quant_research/providers/provider_factory.py
import importlib
import inspect
import pkgutil
import logging
from typing import Dict, Type, Any, Optional, List, TypeVar

from ..core.config import ProviderConfig, ProviderType
from .base import BaseProvider


logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseProvider)


class ProviderRegistry:
    """Registry for provider classes"""
    
    # Map of provider ID to provider class
    _providers: Dict[str, Type[BaseProvider]] = {}
    
    # Map of provider type to list of provider IDs
    _provider_types: Dict[ProviderType, List[str]] = {}
    
    @classmethod
    def register(cls, provider_id: str, provider_class: Type[BaseProvider], 
                provider_type: ProviderType = ProviderType.CUSTOM) -> None:
        """
        Register a provider class with an identifier
        
        Args:
            provider_id: Unique identifier for the provider
            provider_class: Provider class that extends BaseProvider
            provider_type: Type category for the provider
        """
        if not issubclass(provider_class, BaseProvider):
            raise TypeError(f"Provider class must extend BaseProvider: {provider_class}")
        
        cls._providers[provider_id] = provider_class
        
        # Register with provider type
        if provider_type not in cls._provider_types:
            cls._provider_types[provider_type] = []
        
        if provider_id not in cls._provider_types[provider_type]:
            cls._provider_types[provider_type].append(provider_id)
        
        logger.info(f"Registered provider '{provider_id}' of type '{provider_type}'")
    
    @classmethod
    def unregister(cls, provider_id: str) -> None:
        """Unregister a provider by ID"""
        if provider_id in cls._providers:
            provider_class = cls._providers[provider_id]
            del cls._providers[provider_id]
            
            # Remove from provider types
            for provider_type, providers in cls._provider_types.items():
                if provider_id in providers:
                    providers.remove(provider_id)
            
            logger.info(f"Unregistered provider '{provider_id}'")
    
    @classmethod
    def get_provider_class(cls, provider_id: str) -> Optional[Type[BaseProvider]]:
        """Get a provider class by ID"""
        return cls._providers.get(provider_id)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider IDs"""
        return list(cls._providers.keys())
    
    @classmethod
    def list_providers_by_type(cls, provider_type: ProviderType) -> List[str]:
        """List all provider IDs of a specific type"""
        return cls._provider_types.get(provider_type, [])
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (mainly for testing)"""
        cls._providers.clear()
        cls._provider_types.clear()


class ProviderFactory:
    """Factory for creating provider instances"""
    
    @classmethod
    def create(cls, provider_id: str, config: ProviderConfig) -> BaseProvider:
        """
        Create a provider instance from configuration
        
        Args:
            provider_id: ID of the registered provider
            config: Configuration for the provider
            
        Returns:
            An instance of the provider
            
        Raises:
            ValueError: If the provider ID is not registered
            TypeError: If the configuration type doesn't match the provider
        """
        provider_class = ProviderRegistry.get_provider_class(provider_id)
        
        if not provider_class:
            raise ValueError(f"Provider '{provider_id}' not registered")
        
        # Create the provider instance
        return provider_class(config)
    
    @classmethod
    def create_from_config(cls, config: ProviderConfig) -> BaseProvider:
        """
        Create a provider from its configuration, inferring the provider ID
        
        Args:
            config: Provider configuration with name and type
            
        Returns:
            An instance of the provider
        """
        # Try to find a provider matching the name or type
        provider_id = None
        
        # First try exact name match
        if config.name in ProviderRegistry.list_providers():
            provider_id = config.name
        
        # Then try by type
        if not provider_id and config.type:
            type_providers = ProviderRegistry.list_providers_by_type(config.type)
            if type_providers:
                provider_id = type_providers[0]
        
        if not provider_id:
            raise ValueError(f"Could not find provider for config: {config.name} (type: {config.type})")
        
        return cls.create(provider_id, config)
    
    @classmethod
    def discover_providers(cls) -> int:
        """
        Auto-discover and register all providers in the package
        
        Returns:
            Number of providers discovered
        """
        import quant_research.providers as providers_pkg
        
        count = 0
        for _, module_name, is_pkg in pkgutil.iter_modules(providers_pkg.__path__):
            if not is_pkg and module_name != "base" and not module_name.startswith("_"):
                try:
                    module = importlib.import_module(f"quant_research.providers.{module_name}")
                    
                    # Look for provider classes in the module
                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        
                        if (inspect.isclass(item) and 
                            issubclass(item, BaseProvider) and 
                            item != BaseProvider):
                            
                            # Determine provider type from module name
                            provider_type = ProviderType.CUSTOM
                            for pt in ProviderType:
                                if pt.value in module_name:
                                    provider_type = pt
                                    break
                            
                            # Register using a standard naming convention
                            provider_id = module_name
                            ProviderRegistry.register(provider_id, item, provider_type)
                            count += 1
                
                except Exception as e:
                    logger.error(f"Error discovering providers in {module_name}: {e}")
        
        return count