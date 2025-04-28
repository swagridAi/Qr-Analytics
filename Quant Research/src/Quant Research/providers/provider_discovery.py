# src/quant_research/providers/provider_discovery.py
import inspect
import logging
from typing import Dict, List, Any, Optional, Set, Type, Union, Tuple
from enum import Enum

from ..core.config import ProviderConfig, ProviderType
from .base import BaseProvider
from .provider_factory import ProviderRegistry

logger = logging.getLogger(__name__)


class ProviderCapability(str, Enum):
    """Capabilities that providers can support"""
    HISTORICAL_DATA = "historical_data"
    REAL_TIME_DATA = "real_time_data"
    ORDER_BOOK = "order_book"
    TICKER = "ticker"
    OHLCV = "ohlcv"
    TRADES = "trades"
    SENTIMENT = "sentiment"
    FUNDAMENTALS = "fundamentals"
    STREAMING = "streaming"
    BLOCKCHAIN_METRICS = "blockchain_metrics"


class ProviderDiscovery:
    """
    API for discovering and introspecting available providers.
    
    This class provides methods to:
    - List available providers
    - Get detailed information about providers
    - Filter providers by type, capability, etc.
    - Get configuration requirements
    
    Examples:
        # Get all available providers
        providers = ProviderDiscovery.get_available_providers()
        
        # Find providers with real-time data capability
        realtime_providers = ProviderDiscovery.find_providers_by_capability(
            ProviderCapability.REAL_TIME_DATA
        )
        
        # Get configuration schema for a provider
        config_info = ProviderDiscovery.get_provider_config_schema('crypto_ccxt')
    """
    
    # Cache for provider capabilities (provider_id -> List[ProviderCapability])
    _provider_capabilities: Dict[str, List[ProviderCapability]] = {}
    
    # Cache for provider metadata
    _provider_metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available providers.
        
        Returns:
            Dictionary mapping provider IDs to metadata including:
            - provider_type: Type category of the provider
            - description: Brief description of the provider
            - config_class: Configuration class required by this provider
            - capabilities: List of capabilities this provider supports
        
        Example:
            >>> providers = ProviderDiscovery.get_available_providers()
            >>> for provider_id, info in providers.items():
            ...     print(f"{provider_id}: {info['provider_type']}")
            crypto_ccxt: crypto
            equities_yf: equity
        """
        # Force discovery of providers first
        ProviderRegistry.discover_providers()
        
        # Initialize results
        result = {}
        
        # Process each provider in the registry
        for provider_id in ProviderRegistry.list_providers():
            provider_class = ProviderRegistry.get_provider_class(provider_id)
            if provider_class:
                # Get provider metadata
                metadata = cls._get_provider_metadata(provider_id, provider_class)
                result[provider_id] = metadata
        
        return result
    
    @classmethod
    def get_provider_info(cls, provider_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific provider.
        
        Args:
            provider_id: Identifier of the provider
            
        Returns:
            Dictionary with provider metadata
            
        Raises:
            ValueError: If provider is not found
            
        Example:
            >>> info = ProviderDiscovery.get_provider_info('crypto_ccxt')
            >>> print(f"Type: {info['provider_type']}")
            >>> print(f"Capabilities: {', '.join(info['capabilities'])}")
        """
        provider_class = ProviderRegistry.get_provider_class(provider_id)
        if not provider_class:
            raise ValueError(f"Provider '{provider_id}' not found")
        
        # Get the metadata for this provider
        metadata = cls._get_provider_metadata(provider_id, provider_class)
        
        # Add extra detailed information
        metadata['methods'] = cls._get_provider_methods(provider_class)
        metadata['config_schema'] = cls.get_provider_config_schema(provider_id)
        
        return metadata
    
    @classmethod
    def find_providers_by_type(cls, provider_type: ProviderType) -> List[str]:
        """
        Find providers of a specific type.
        
        Args:
            provider_type: Type of provider to find
            
        Returns:
            List of provider IDs matching the type
            
        Example:
            >>> crypto_providers = ProviderDiscovery.find_providers_by_type(ProviderType.CRYPTO)
            >>> print(crypto_providers)
            ['crypto_ccxt', 'crypto_binance']
        """
        return ProviderRegistry.list_providers_by_type(provider_type)
    
    @classmethod
    def find_providers_by_capability(cls, capability: ProviderCapability) -> List[str]:
        """
        Find providers that support a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of provider IDs supporting the capability
            
        Example:
            >>> streaming_providers = ProviderDiscovery.find_providers_by_capability(
            ...     ProviderCapability.STREAMING
            ... )
            >>> print(streaming_providers)
            ['crypto_ccxt', 'crypto_websocket']
        """
        # Get all available providers
        providers = cls.get_available_providers()
        
        # Filter by capability
        return [
            provider_id for provider_id, info in providers.items()
            if capability in info.get('capabilities', [])
        ]
    
    @classmethod
    def get_provider_config_schema(cls, provider_id: str) -> Dict[str, Any]:
        """
        Get configuration schema information for a provider.
        
        Args:
            provider_id: Identifier of the provider
            
        Returns:
            Dictionary with configuration schema information
            
        Raises:
            ValueError: If provider is not found
            
        Example:
            >>> schema = ProviderDiscovery.get_provider_config_schema('crypto_ccxt')
            >>> print(f"Required fields: {schema['required_fields']}")
            >>> for field, info in schema['properties'].items():
            ...     print(f"{field}: {info['description']}")
        """
        provider_class = ProviderRegistry.get_provider_class(provider_id)
        if not provider_class:
            raise ValueError(f"Provider '{provider_id}' not found")
        
        # Try to determine the config class through type annotations
        config_class = cls._get_config_class_for_provider(provider_class)
        
        if not config_class:
            # Default to generic ProviderConfig
            from ..core.config import ProviderConfig
            config_class = ProviderConfig
        
        # Extract schema from the configuration class
        schema = {}
        
        # Get fields from Config class
        if hasattr(config_class, '__fields__'):
            schema['properties'] = {}
            schema['required_fields'] = []
            
            for field_name, field in config_class.__fields__.items():
                field_info = {
                    'type': str(field.type_),
                    'description': field.field_info.description or '',
                    'default': None if field.default is ... else field.default,
                    'required': field.required
                }
                
                schema['properties'][field_name] = field_info
                
                if field.required:
                    schema['required_fields'].append(field_name)
        
        # Add example configuration if available
        schema['example'] = cls._get_example_config(provider_class, config_class)
        
        return schema
    
    @classmethod
    def create_provider_example(cls, provider_id: str) -> str:
        """
        Generate example code for creating and using a provider.
        
        Args:
            provider_id: Identifier of the provider
            
        Returns:
            String containing example Python code
            
        Raises:
            ValueError: If provider is not found
            
        Example:
            >>> example = ProviderDiscovery.create_provider_example('crypto_ccxt')
            >>> print(example)
            ```python
            from quant_research.providers import ProviderFactory
            from quant_research.core.config import ProviderConfig
            
            # Create configuration
            config = CCXTProviderConfig(
                name="crypto_ccxt",
                exchange="binance",
                symbols=["BTC/USDT", "ETH/USDT"],
                timeframe="1h"
            )
            
            # Create provider
            provider = ProviderFactory.create('crypto_ccxt', config)
            
            # Usage example
            async def fetch_data():
                await provider.connect()
                async for data_point in provider.fetch_data():
                    print(data_point)
                await provider.disconnect()
            ```
        """
        provider_class = ProviderRegistry.get_provider_class(provider_id)
        if not provider_class:
            raise ValueError(f"Provider '{provider_id}' not found")
        
        # Get provider info
        info = cls.get_provider_info(provider_id)
        
        # Generate example code
        config_class = info.get('config_class', 'ProviderConfig')
        config_example = info.get('config_schema', {}).get('example', {})
        
        # Convert config example dict to code
        config_code = []
        for key, value in config_example.items():
            if isinstance(value, str):
                config_code.append(f'    {key}="{value}"')
            elif isinstance(value, list):
                if all(isinstance(item, str) for item in value):
                    items_str = '", "'.join(value)
                    config_code.append(f'    {key}=["{items_str}"]')
                else:
                    config_code.append(f'    {key}={value}')
            else:
                config_code.append(f'    {key}={value}')
        
        config_code_str = ',\n'.join(config_code)
        
        # Sample usage code
        example = f'''```python
from quant_research.providers import ProviderFactory
from quant_research.core.config import {config_class.__name__ if isinstance(config_class, type) else 'ProviderConfig'}

# Create configuration
config = {config_class.__name__ if isinstance(config_class, type) else 'ProviderConfig'}(
{config_code_str}
)

# Create provider
provider = ProviderFactory.create('{provider_id}', config)

# Usage example
async def fetch_data():
    await provider.connect()
    async for data_point in provider.fetch_data():
        print(data_point)
    await provider.disconnect()
```'''
        
        return example
    
    @classmethod
    def get_available_capabilities(cls) -> List[str]:
        """
        Get list of all available capability types.
        
        Returns:
            List of capability names
            
        Example:
            >>> capabilities = ProviderDiscovery.get_available_capabilities()
            >>> print(capabilities)
            ['historical_data', 'real_time_data', 'order_book', ...]
        """
        return [capability.value for capability in ProviderCapability]
    
    @classmethod
    def get_available_provider_types(cls) -> List[str]:
        """
        Get list of all available provider types.
        
        Returns:
            List of provider type names
            
        Example:
            >>> types = ProviderDiscovery.get_available_provider_types()
            >>> print(types)
            ['crypto', 'equity', 'blockchain', 'sentiment', 'custom']
        """
        return [provider_type.value for provider_type in ProviderType]
    
    #
    # Private helper methods
    #
    
    @classmethod
    def _get_provider_metadata(cls, provider_id: str, provider_class: Type[BaseProvider]) -> Dict[str, Any]:
        """Get metadata for a provider class"""
        # Check cache first
        if provider_id in cls._provider_metadata:
            return cls._provider_metadata[provider_id].copy()
        
        # Determine provider type
        provider_type = cls._determine_provider_type(provider_class)
        
        # Get docstring
        doc = inspect.getdoc(provider_class) or ""
        description = doc.split('\n\n')[0] if doc else ""
        
        # Get config class
        config_class = cls._get_config_class_for_provider(provider_class)
        
        # Get capabilities
        capabilities = cls._determine_provider_capabilities(provider_id, provider_class)
        
        # Create metadata
        metadata = {
            'provider_id': provider_id,
            'provider_type': provider_type.value if isinstance(provider_type, ProviderType) else provider_type,
            'description': description,
            'config_class': config_class,
            'capabilities': capabilities,
            'class_name': provider_class.__name__,
            'module': provider_class.__module__
        }
        
        # Cache the result
        cls._provider_metadata[provider_id] = metadata
        
        return metadata.copy()
    
    @classmethod
    def _determine_provider_type(cls, provider_class: Type[BaseProvider]) -> Union[ProviderType, str]:
        """Determine the provider type for a provider class"""
        # Look for type in class attributes
        for provider_type in ProviderType:
            # Check module or class name
            if provider_type.value in provider_class.__module__ or provider_type.value in provider_class.__name__.lower():
                return provider_type
        
        # Try to create an instance to check its type
        try:
            # Most providers expect a config in __init__, find all registered configs
            for provider_id, registered_class in ProviderRegistry._providers.items():
                if registered_class == provider_class:
                    # Found this provider in registry, look up its type
                    for provider_type, providers in ProviderRegistry._provider_types.items():
                        if provider_id in providers:
                            return provider_type
        except:
            pass
        
        # Default
        return ProviderType.CUSTOM
    
    @classmethod
    def _get_config_class_for_provider(cls, provider_class: Type[BaseProvider]) -> Optional[Type]:
        """Get the config class for a provider through type hints"""
        # Check class __orig_bases__ for Generic parameters
        config_class = None
        
        if hasattr(provider_class, '__orig_bases__'):
            for base in provider_class.__orig_bases__:
                if hasattr(base, '__origin__') and base.__origin__ == BaseProvider:
                    if hasattr(base, '__args__') and base.__args__:
                        config_class = base.__args__[0]
                        break
        
        # Try to infer from init signature
        if not config_class:
            try:
                sig = inspect.signature(provider_class.__init__)
                config_param = next(
                    (param for param in sig.parameters.values() 
                     if param.name == 'config'), 
                    None
                )
                if config_param and config_param.annotation != inspect.Parameter.empty:
                    config_class = config_param.annotation
            except Exception:
                pass
        
        return config_class
    
    @classmethod
    def _determine_provider_capabilities(
        cls, provider_id: str, provider_class: Type[BaseProvider]
    ) -> List[ProviderCapability]:
        """Determine the capabilities of a provider class"""
        # Check cache first
        if provider_id in cls._provider_capabilities:
            return cls._provider_capabilities[provider_id].copy()
        
        capabilities = []
        
        # Check method names for capabilities
        methods = cls._get_provider_methods(provider_class)
        for method in methods:
            method_name = method.lower()
            
            # Map method names to capabilities
            if 'fetch_data' in method_name or 'get_data' in method_name:
                capabilities.append(ProviderCapability.HISTORICAL_DATA)
            
            if 'stream' in method_name or 'subscribe' in method_name:
                capabilities.append(ProviderCapability.STREAMING)
            
            if 'ticker' in method_name:
                capabilities.append(ProviderCapability.TICKER)
            
            if 'ohlcv' in method_name or 'candle' in method_name:
                capabilities.append(ProviderCapability.OHLCV)
            
            if 'order_book' in method_name or 'depth' in method_name:
                capabilities.append(ProviderCapability.ORDER_BOOK)
            
            if 'trade' in method_name:
                capabilities.append(ProviderCapability.TRADES)
            
            if 'sentiment' in method_name:
                capabilities.append(ProviderCapability.SENTIMENT)
            
            if 'fundamental' in method_name:
                capabilities.append(ProviderCapability.FUNDAMENTALS)
            
            if 'blockchain' in method_name or 'onchain' in method_name:
                capabilities.append(ProviderCapability.BLOCKCHAIN_METRICS)
        
        # Check class name and module for capabilities
        class_name = provider_class.__name__.lower()
        module_name = provider_class.__module__.lower()
        
        if 'realtime' in class_name or 'websocket' in class_name or 'stream' in class_name:
            capabilities.append(ProviderCapability.REAL_TIME_DATA)
        elif 'realtime' in module_name or 'websocket' in module_name or 'stream' in module_name:
            capabilities.append(ProviderCapability.REAL_TIME_DATA)
        
        # Check docstring for capabilities
        doc = inspect.getdoc(provider_class) or ""
        
        for capability in ProviderCapability:
            if capability.value in doc.lower():
                capabilities.append(capability)
        
        # Ensure uniqueness
        capabilities = list(set(capabilities))
        
        # Cache the result
        cls._provider_capabilities[provider_id] = capabilities
        
        return capabilities.copy()
    
    @classmethod
    def _get_provider_methods(cls, provider_class: Type[BaseProvider]) -> List[str]:
        """Get public methods of a provider class"""
        methods = []
        
        for name, method in inspect.getmembers(provider_class, predicate=inspect.isfunction):
            # Skip private methods
            if name.startswith('_'):
                continue
            
            # Skip methods from BaseProvider (already known)
            if hasattr(BaseProvider, name):
                if method.__qualname__ == f"BaseProvider.{name}":
                    continue
            
            methods.append(name)
        
        return methods
    
    @classmethod
    def _get_example_config(cls, provider_class: Type[BaseProvider], config_class: Type) -> Dict[str, Any]:
        """Generate an example configuration for a provider"""
        example = {
            'name': provider_class.__name__.lower().replace('provider', ''),
            'type': cls._determine_provider_type(provider_class),
        }
        
        # Add provider-specific fields based on config class
        if hasattr(config_class, '__fields__'):
            for field_name, field in config_class.__fields__.items():
                # Skip base fields already set
                if field_name in ('name', 'type', 'connection'):
                    continue
                
                # Add required fields with sample values
                if field.required:
                    # Generate sample value based on field type
                    if 'symbol' in field_name.lower() or field_name.lower() == 'symbols':
                        if 'List' in str(field.type_):
                            if 'crypto' in str(example.get('type', '')).lower():
                                example[field_name] = ['BTC/USDT', 'ETH/USDT']
                            else:
                                example[field_name] = ['AAPL', 'MSFT']
                        else:
                            if 'crypto' in str(example.get('type', '')).lower():
                                example[field_name] = 'BTC/USDT'
                            else:
                                example[field_name] = 'AAPL'
                    
                    elif 'key' in field_name.lower() or 'secret' in field_name.lower():
                        example[field_name] = '*** your key here ***'
                    
                    elif 'url' in field_name.lower() or 'endpoint' in field_name.lower():
                        example[field_name] = 'https://api.example.com'
                    
                    elif 'exchange' in field_name.lower():
                        example[field_name] = 'binance'
                    
                    elif 'timeframe' in field_name.lower() or 'interval' in field_name.lower():
                        example[field_name] = '1h'
                    
                    elif 'network' in field_name.lower():
                        example[field_name] = 'ethereum'
                    
                    elif field_name.lower() == 'keywords':
                        example[field_name] = ['bitcoin', 'crypto']
        
        return example


# Add discovery methods to ProviderFactory for convenience
def extend_provider_factory():
    """Add discovery methods to ProviderFactory for convenience"""
    from .provider_factory import ProviderFactory
    
    # Add methods if they don't exist
    if not hasattr(ProviderFactory, 'get_available_providers'):
        ProviderFactory.get_available_providers = ProviderDiscovery.get_available_providers
    
    if not hasattr(ProviderFactory, 'get_provider_info'):
        ProviderFactory.get_provider_info = ProviderDiscovery.get_provider_info
    
    if not hasattr(ProviderFactory, 'find_providers_by_type'):
        ProviderFactory.find_providers_by_type = ProviderDiscovery.find_providers_by_type
    
    if not hasattr(ProviderFactory, 'find_providers_by_capability'):
        ProviderFactory.find_providers_by_capability = ProviderDiscovery.find_providers_by_capability


# Initialize by extending ProviderFactory
extend_provider_factory()