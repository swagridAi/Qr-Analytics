# src/quant_research/providers/__init__.py
"""
Providers package for data source adapters and standardized data retrieval.

This package implements the edge layer of the hexagonal architecture,
connecting to external data services and emitting standardized data records.
"""

__version__ = "0.1.0"

# Export base classes
from .base import BaseProvider
from .provider_factory import ProviderRegistry, ProviderFactory
from .connection_manager import ConnectionManager

# Export telemetry system
from .telemetry import (
    get_telemetry_manager, 
    with_telemetry, 
    extract_quota_headers,
    TelemetryManager
)

# Export concrete providers
try:
    from .crypto_ccxt import CCXTProvider, CCXTProviderConfig
except ImportError:
    pass

# Auto-discover and register all providers
ProviderRegistry.discover_providers()