# src/quant_research/providers/provider_factory.py
import importlib
import inspect
import pkgutil
import logging
from functools import wraps
from typing import Dict, Type, Any, Optional, List, TypeVar, Callable, Set, Union

from ..core.config import ProviderConfig, ProviderType
from .base import BaseProvider


logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseProvider)


class ProviderRegistry:
    """Registry for provider classes with decorator-based registration"""
    
    # Map of provider ID to provider class
    _providers: Dict[str, Type[BaseProvider]] = {}
    
    # Map of provider type to list of provider IDs
    _provider_types: Dict[ProviderType, List[str]] = {}
    
    # Set of auto-discovered modules to avoid re-processing
    _discovered_modules: Set[str] = set()
    
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
        cls._discovered_modules.clear()


def register_provider(
    provider_id: Optional[str] = None, 
    provider_type: ProviderType = ProviderType.CUSTOM
) -> Callable:
    """
    Decorator to register a provider class
    
    Usage:
        @register_provider("my_provider")
        class MyProvider(BaseProvider):
            ...
    
    Args:
        provider_id: Unique identifier for the provider (defaults to class name in snake_case)
        provider_type: Type category for the provider
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[BaseProvider]) -> Type[BaseProvider]:
        # Determine provider_id if not provided
        nonlocal provider_id
        if provider_id is None:
            # Convert class name from CamelCase to snake_case
            name = cls.__name__
            provider_id = ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')
            
            # Remove 'Provider' suffix if present
            if provider_id.endswith('_provider'):
                provider_id = provider_id[:-9]
        
        # Determine provider type if not explicitly set
        actual_provider_type = provider_type
        if provider_type == ProviderType.CUSTOM:
            # Try to infer from class name or module name
            module_name = cls.__module__.split('.')[-1]
            class_name = cls.__name__.lower()
            
            for pt in ProviderType:
                if pt.value in module_name or pt.value in class_name:
                    actual_provider_type = pt
                    break
        
        # Register the provider
        ProviderRegistry.register(provider_id, cls, actual_provider_type)
        
        return cls
    
    # Handle case where decorator is used without parentheses
    if inspect.isclass(provider_id) and issubclass(provider_id, BaseProvider):
        cls = provider_id
        provider_id = None
        return decorator(cls)
    
    return decorator


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
            # Try automatic discovery first
            cls.discover_providers()
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
        
        # If still not found, try discovery
        if not provider_id:
            cls.discover_providers()
            
            # Try again after discovery
            if config.name in ProviderRegistry.list_providers():
                provider_id = config.name
            elif config.type:
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
        
        # Get all modules in the providers package
        for _, module_name, is_pkg in pkgutil.iter_modules(providers_pkg.__path__):
            # Skip packages, base.py, and any modules starting with _
            if is_pkg or module_name == "base" or module_name.startswith("_"):
                continue
                
            # Skip already discovered modules
            if module_name in ProviderRegistry._discovered_modules:
                continue
            
            try:
                # Import the module
                full_module_name = f"quant_research.providers.{module_name}"
                module = importlib.import_module(full_module_name)
                ProviderRegistry._discovered_modules.add(module_name)
                
                # Look for provider classes in the module
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    
                    # Check if it's a provider class and not the base class
                    if (inspect.isclass(item) and 
                        issubclass(item, BaseProvider) and 
                        item != BaseProvider):
                        
                        # Skip if already registered through the decorator
                        is_registered = False
                        for registered_id, registered_class in ProviderRegistry._providers.items():
                            if registered_class == item:
                                is_registered = True
                                break
                        
                        if is_registered:
                            continue
                        
                        # Determine provider type from module name
                        provider_type = ProviderType.CUSTOM
                        for pt in ProviderType:
                            if pt.value in module_name:
                                provider_type = pt
                                break
                        
                        # Register using module name as provider ID
                        provider_id = module_name
                        ProviderRegistry.register(provider_id, item, provider_type)
                        count += 1
                
            except Exception as e:
                logger.error(f"Error discovering providers in {module_name}: {e}")
        
        return count


# For backward compatibility:
def get_provider(provider_id: str, config: ProviderConfig) -> BaseProvider:
    """Get a provider instance (backward compatibility function)"""
    return ProviderFactory.create(provider_id, config)