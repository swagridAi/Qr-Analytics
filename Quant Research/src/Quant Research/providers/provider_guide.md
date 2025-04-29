# Providers Module Guide: Data Ingestion in the Quant Research Pipeline

This guide explains how to effectively use the providers module for data ingestion in the Quant Research pipeline.

## What Is the Providers Module?

The providers module is the edge layer of our hexagonal architecture, responsible for retrieving data from external sources and standardizing it for use throughout the platform. It follows a plugin pattern where all data sources implement a common interface.

Key features include:
- Connection pooling and management
- Automatic retries with exponential backoff
- Rate limiting
- Standardized error handling
- Provider discovery and configuration
- Telemetry and monitoring

## Provider Architecture

```
providers/
├── __init__.py                  # Package exports
├── base.py                      # BaseProvider interface
├── provider_factory.py          # Factory and registration
├── connection_manager.py        # Connection management
├── crypto_ccxt.py               # Crypto exchange data
├── equities_yf.py               # Stock market data
├── onchain.py                   # Blockchain data
└── sentiment_twitter.py         # Social sentiment
```

Each provider implements the `BaseProvider` interface with standardized methods like `connect()`, `disconnect()`, and `fetch_data()`.

## How to Use Providers in the Pipeline

### 1. Select and Configure a Provider

Each provider has its own configuration class that extends `ProviderConfig`:

```python
from quant_research.providers import ProviderFactory
from quant_research.providers.crypto_ccxt import CCXTProviderConfig

# Create configuration
config = CCXTProviderConfig(
    name="binance_data",
    exchange="binance",
    symbols=["BTC/USDT", "ETH/USDT"],
    timeframe="1h"
)
```

### 2. Create a Provider Instance

Use the `ProviderFactory` to create a provider instance:

```python
# Create provider through factory
provider = ProviderFactory.create("crypto_ccxt", config)

```

### 3. Connect and Fetch Data

```python
async def fetch_crypto_data():
    try:
        # Connect to the data source
        await provider.connect()
        
        # Check connection status
        if await provider.is_connected():
            # Fetch data
            async for data_point in provider.fetch_data(
                symbols=["BTC/USDT"],
                timeframe="1h",
                since=datetime(2023, 1, 1),
                limit=1000
            ):
                # Process each data point (PriceBar object)
                yield data_point
    finally:
        # Always disconnect when done
        await provider.disconnect()
```

### 4. Use with Context Manager

For automatic resource management, use the provider as an async context manager:

```python
async def fetch_data_with_context():
    async with provider as p:
        async for data_point in p.fetch_data():
            yield data_point
    # Provider automatically disconnects
```

## Pipeline Integration Examples

### ETL Pipeline Example

```python
async def crypto_etl_pipeline():
    """ETL pipeline for cryptocurrency data"""
    # Create provider
    config = CCXTProviderConfig(
        name="crypto_ccxt",
        exchange="binance",
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1h"
    )
    provider = ProviderFactory.create("crypto_ccxt", config)
    
    try:
        # Connect to data source
        await provider.connect()
        
        # Fetch data
        data_points = []
        async for data_point in provider.fetch_data():
            data_points.append(data_point)
        
        # Transform to DataFrame
        import pandas as pd
        df = pd.DataFrame([p.dict() for p in data_points])
        
        # Save to parquet in DuckDB
        from quant_research.core.storage import save_to_duckdb
        save_to_duckdb(df, "crypto_data")
        
    finally:
        await provider.disconnect()
```

### Multi-Provider Pipeline

```python
async def multi_source_pipeline():
    """Collect data from multiple providers"""
    # Configure providers
    crypto_config = CCXTProviderConfig(...)
    equity_config = YahooFinanceProviderConfig(...)
    sentiment_config = TwitterSentimentConfig(...)
    
    # Create provider instances
    crypto_provider = ProviderFactory.create("crypto_ccxt", crypto_config)
    equity_provider = ProviderFactory.create("equities_yf", equity_config)
    sentiment_provider = ProviderFactory.create("sentiment_twitter", sentiment_config)
    
    # Fetch data from each source
    async with crypto_provider as cp, equity_provider as ep, sentiment_provider as sp:
        # Collect crypto data
        crypto_data = []
        async for data in cp.fetch_data():
            crypto_data.append(data)
            
        # Collect equity data
        equity_data = []
        async for data in ep.fetch_data():
            equity_data.append(data)
            
        # Collect sentiment data
        sentiment_data = []
        async for data in sp.fetch_data():
            sentiment_data.append(data)
    
    # Process and merge the collected data
    # ...
```

## Error Handling

The provider module defines several error types that you should handle:

```python
from quant_research.core.errors import (
    ConnectionError, DataFetchError, 
    RateLimitError, AuthenticationError
)

async def fetch_with_error_handling():
    try:
        await provider.connect()
        async for data in provider.fetch_data():
            yield data
    except ConnectionError as e:
        # Handle connection issues
        logger.error(f"Connection failed: {e}")
    except RateLimitError as e:
        # Handle rate limiting
        logger.warning(f"Rate limit hit: {e}, retry after {e.retry_after}s")
        await asyncio.sleep(e.retry_after)
    except AuthenticationError as e:
        # Handle auth issues
        logger.error(f"Authentication failed: {e}")
    except DataFetchError as e:
        # Handle data retrieval issues
        logger.error(f"Failed to fetch data: {e}")
    finally:
        await provider.disconnect()
```

## Provider Metadata and Discovery

You can discover and inspect available providers:

```python
# Get all available providers
from quant_research.providers.provider_discovery import ProviderDiscovery

# List all providers
providers = ProviderDiscovery.get_available_providers()
for provider_id, info in providers.items():
    print(f"{provider_id}: {info['provider_type']}")

# Find providers by capability
from quant_research.providers.provider_discovery import ProviderCapability
realtime_providers = ProviderDiscovery.find_providers_by_capability(
    ProviderCapability.REAL_TIME_DATA
)
```

## Best Practices

1. **Use the Factory Pattern**: Always create providers through `ProviderFactory` for proper initialization and discovery.

2. **Properly Handle Connections**: Always use `connect()` and `disconnect()` or the async context manager pattern.

3. **Use Appropriate Error Handling**: Handle specific exception types from providers.

4. **Configure Rate Limiting**: Set appropriate rate limits to avoid being banned by data sources.

5. **Use Environment Variables for Credentials**: Store API keys in environment variables referenced by the config.

6. **Validate Configurations**: Use the validation framework to check configurations before use.

7. **Monitor Telemetry**: Track provider performance using the telemetry system.

8. **Batch Processing**: For large data sets, process data in batches rather than collecting all at once.

## Configuration via Files

For reproducible pipelines, store provider configurations in YAML/JSON:

```python
from quant_research.providers.config.loader import load_config
from quant_research.providers import ProviderFactory

# Load configuration from file
config = load_config("configs/crypto_provider.yaml")

# Create provider
provider = ProviderFactory.create_from_config(config)
```

Example YAML configuration:

```yaml
name: binance_btc
type: crypto
exchange: binance
symbols:
  - BTC/USDT
  - ETH/USDT
timeframe: 1h
connection:
  timeout: 30
  pool_size: 5
  keep_alive: true
```

## Extending with New Providers

To add a new data source:

1. Create a new provider class implementing `BaseProvider`
2. Create a configuration class extending `ProviderConfig`
3. Register the provider with the `@register_provider` decorator

Example:

```python
from quant_research.providers.base import BaseProvider
from quant_research.core.config import ProviderConfig, ProviderType
from quant_research.providers.provider_factory import register_provider

class MyProviderConfig(ProviderConfig):
    # Configuration fields
    api_url: str
    dataset: str
    
    # Override defaults
    name: str = "my_provider"
    type: ProviderType = ProviderType.CUSTOM

@register_provider("my_provider")
class MyProvider(BaseProvider[MyProviderConfig]):
    # Implement required methods
    async def connect(self) -> None:
        # ...
    
    async def is_connected(self) -> bool:
        # ...
    
    async def disconnect(self) -> None:
        # ...
    
    async def fetch_data(self, **params) -> AsyncIterator[Dict[str, Any]]:
        # ...
    
    async def get_metadata(self) -> Dict[str, Any]:
        # ...
```

## Conclusion

The providers module offers a flexible, resilient way to ingest data from multiple sources. By using the standardized interface and factory pattern, pipelines can be built that work consistently across different data providers while remaining extensible for new data sources.

Remember to handle connections properly, manage resources efficiently, and use appropriate error handling for production reliability.