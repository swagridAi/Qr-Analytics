# src/quant_research/providers/config_validation.py
"""
Configuration validation framework for provider configurations.

This module provides:
1. A validation framework for checking provider configurations
2. Provider-specific configuration validators
3. Utilities for running validation suites and formatting results
"""

import asyncio
import ipaddress
import inspect
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import ccxt
from pydantic import Field, root_validator, validator

from ..core.config import ProviderConfig, ProviderType
from .base import BaseProvider

logger = logging.getLogger(__name__)

#
# ===== Core Validation Framework =====
#

@dataclass
class ValidationResult:
    """Result of a configuration validation check"""
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "error"  # "error", "warning", or "info"


class ConfigValidator:
    """Utility class for validating provider configurations"""
    
    @staticmethod
    def validate_config(config: ProviderConfig) -> List[ValidationResult]:
        """
        Validate a provider configuration and return validation results
        
        Args:
            config: Provider configuration to validate
            
        Returns:
            List of validation results
        """
        results = []
        
        # 1. Basic validation through Pydantic (this happens when the config is created)
        # If we got here, the basic validation passed
        
        # 2. Check if required fields are present and non-empty
        results.extend(ConfigValidator._validate_required_fields(config))
        
        # 3. Check environment variables
        results.extend(ConfigValidator._validate_environment_variables(config))
        
        # 4. Check for suspicious or problematic values
        results.extend(ConfigValidator._check_for_problems(config))
        
        return results
    
    @staticmethod
    def validate_config_for_provider(
        config: ProviderConfig,
        provider_class: Type[BaseProvider]
    ) -> List[ValidationResult]:
        """
        Validate a configuration for compatibility with a specific provider class
        
        Args:
            config: Provider configuration to validate
            provider_class: Provider class to validate against
            
        Returns:
            List of validation results
        """
        results = []
        
        # 1. Basic validation
        results.extend(ConfigValidator.validate_config(config))
        
        # 2. Check if config type matches provider's expected type
        results.extend(ConfigValidator._validate_config_type_match(config, provider_class))
        
        # 3. Check if provider requires fields that are missing in config
        results.extend(ConfigValidator._validate_provider_required_fields(config, provider_class))
        
        return results
    
    @staticmethod
    async def validate_connection(
        config: ProviderConfig,
        provider: Union[BaseProvider, Type[BaseProvider]]
    ) -> List[ValidationResult]:
        """
        Validate that a connection can be established with the provider
        
        Args:
            config: Provider configuration
            provider: Provider instance or class
            
        Returns:
            List of validation results
        """
        results = []
        
        # If given a class, instantiate it
        if inspect.isclass(provider):
            try:
                provider_instance = provider(config)
            except Exception as e:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Failed to instantiate provider: {e}",
                    details={"error": str(e), "provider_class": provider.__name__},
                    severity="error"
                ))
                return results
        else:
            provider_instance = provider
        
        # Try to connect
        try:
            await provider_instance.connect()
            
            # Check if connection was successful
            is_connected = await provider_instance.is_connected()
            
            if is_connected:
                results.append(ValidationResult(
                    is_valid=True,
                    message="Successfully connected to provider",
                    severity="info"
                ))
            else:
                results.append(ValidationResult(
                    is_valid=False,
                    message="Connection failed - provider reports it is not connected",
                    severity="error"
                ))
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Connection failed: {e}",
                details={"error": str(e)},
                severity="error"
            ))
        finally:
            # Try to disconnect (cleanup)
            try:
                await provider_instance.disconnect()
            except Exception as e:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Failed to disconnect: {e}",
                    details={"error": str(e)},
                    severity="warning"
                ))
        
        return results
    
    @staticmethod
    def _validate_required_fields(config: ProviderConfig) -> List[ValidationResult]:
        """Check if required fields are present and non-empty"""
        results = []
        
        # Check for empty fields that shouldn't be empty
        empty_fields = []
        
        if not config.name:
            empty_fields.append("name")
        
        if empty_fields:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Required fields cannot be empty: {', '.join(empty_fields)}",
                details={"empty_fields": empty_fields},
                severity="error"
            ))
        
        return results
    
    @staticmethod
    def _validate_environment_variables(config: ProviderConfig) -> List[ValidationResult]:
        """Check if required environment variables are set"""
        results = []
        
        if config.require_auth:
            missing_keys = []
            for key in config.api_keys:
                if not config.get_api_key(key):
                    missing_keys.append(f"{config.env_prefix}{key}")
            
            if missing_keys:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Missing required environment variables: {', '.join(missing_keys)}",
                    details={"missing_env_vars": missing_keys},
                    severity="error"
                ))
        
        return results
    
    @staticmethod
    def _check_for_problems(config: ProviderConfig) -> List[ValidationResult]:
        """Check for suspicious or problematic values"""
        results = []
        
        # Check connection timeout value
        if config.connection.timeout > 120:
            results.append(ValidationResult(
                is_valid=True,  # This is a warning, not an error
                message=f"Connection timeout is set to {config.connection.timeout}s, which is unusually high",
                details={"timeout": config.connection.timeout},
                severity="warning"
            ))
        
        # Check pool size
        if config.connection.pool_size > 20:
            results.append(ValidationResult(
                is_valid=True,  # This is a warning, not an error
                message=f"Connection pool size is set to {config.connection.pool_size}, which may consume excessive resources",
                details={"pool_size": config.connection.pool_size},
                severity="warning"
            ))
        
        # Rate limit checks
        if config.rate_limit and config.rate_limit.get("requests_per_second", 0) > 100:
            results.append(ValidationResult(
                is_valid=True,  # This is a warning, not an error
                message=f"Rate limit of {config.rate_limit.get('requests_per_second')} requests per second is unusually high",
                details={"rate_limit": config.rate_limit},
                severity="warning"
            ))
        
        return results
    
    @staticmethod
    def _validate_config_type_match(
        config: ProviderConfig,
        provider_class: Type[BaseProvider]
    ) -> List[ValidationResult]:
        """Check if the config type matches the provider's expected type"""
        results = []
        
        # Try to get the expected config type from the provider's type annotations
        expected_config_type = None
        
        # Check class __orig_bases__ for Generic parameters
        if hasattr(provider_class, '__orig_bases__'):
            for base in provider_class.__orig_bases__:
                if hasattr(base, '__origin__') and base.__origin__ is BaseProvider:
                    if hasattr(base, '__args__') and base.__args__:
                        expected_config_type = base.__args__[0]
                        break
        
        if expected_config_type and not isinstance(config, expected_config_type):
            results.append(ValidationResult(
                is_valid=False,
                message=f"Configuration type mismatch: Provider {provider_class.__name__} expects {expected_config_type.__name__}, got {config.__class__.__name__}",
                details={
                    "expected_type": expected_config_type.__name__,
                    "actual_type": config.__class__.__name__
                },
                severity="error"
            ))
        
        return results
    
    @staticmethod
    def _validate_provider_required_fields(
        config: ProviderConfig,
        provider_class: Type[BaseProvider]
    ) -> List[ValidationResult]:
        """Check if the provider requires fields that are missing in the config"""
        results = []
        
        # Check expected fields based on provider type and naming convention
        provider_type_field_requirements = {
            ProviderType.CRYPTO: ["symbols"],
            ProviderType.EQUITY: ["symbols"],
            ProviderType.SENTIMENT: ["keywords"],
            ProviderType.BLOCKCHAIN: ["network"],
        }
        
        if hasattr(config, 'type') and config.type in provider_type_field_requirements:
            required_fields = provider_type_field_requirements[config.type]
            for field in required_fields:
                if not hasattr(config, field) or not getattr(config, field):
                    results.append(ValidationResult(
                        is_valid=False,
                        message=f"Missing required field for {provider_class.__name__}: {field}",
                        details={"missing_field": field, "provider_type": config.type.value},
                        severity="error"
                    ))
        
        return results


class ConfigValidationSuite:
    """Suite of configuration validation tests"""
    
    @staticmethod
    async def run_validation_suite(
        config: ProviderConfig,
        provider_class: Type[BaseProvider],
        test_connection: bool = True
    ) -> Dict[str, List[ValidationResult]]:
        """
        Run a comprehensive validation suite on a provider configuration
        
        Args:
            config: Provider configuration to validate
            provider_class: Provider class to validate against
            test_connection: Whether to test the connection
            
        Returns:
            Dictionary of validation results by category
        """
        results = {}
        
        # Basic configuration validation
        results["basic_validation"] = ConfigValidator.validate_config(config)
        
        # Provider-specific validation
        results["provider_compatibility"] = ConfigValidator.validate_config_for_provider(
            config, provider_class
        )
        
        # Connection test (optional)
        if test_connection:
            results["connection_test"] = await ConfigValidator.validate_connection(
                config, provider_class
            )
        
        return results
    
    @staticmethod
    def format_validation_results(
        results: Dict[str, List[ValidationResult]],
        include_details: bool = False
    ) -> str:
        """
        Format validation results as a readable string
        
        Args:
            results: Validation results by category
            include_details: Whether to include detailed information
            
        Returns:
            Formatted validation results
        """
        lines = ["Configuration Validation Results:"]
        
        # Track overall validity
        is_valid = True
        error_count = 0
        warning_count = 0
        
        for category, category_results in results.items():
            # Skip empty categories
            if not category_results:
                continue
            
            lines.append(f"\n=== {category.replace('_', ' ').title()} ===")
            
            for result in category_results:
                # Add to counts
                if result.severity == "error":
                    error_count += 1
                    is_valid = False
                elif result.severity == "warning":
                    warning_count += 1
                
                # Format the message with appropriate prefix
                prefix = {
                    "error": "❌ ERROR:",
                    "warning": "⚠️ WARNING:",
                    "info": "ℹ️ INFO:"
                }.get(result.severity, "•")
                
                lines.append(f"{prefix} {result.message}")
                
                # Add details if requested
                if include_details and result.details:
                    for key, value in result.details.items():
                        lines.append(f"  - {key}: {value}")
        
        # Add summary
        lines.append("\n=== Summary ===")
        status = "✅ VALID" if is_valid else "❌ INVALID"
        lines.append(f"{status}: Found {error_count} errors and {warning_count} warnings")
        
        return "\n".join(lines)
    
    @staticmethod
    async def validate_and_report(
        config: ProviderConfig,
        provider_class: Type[BaseProvider],
        test_connection: bool = True,
        include_details: bool = False
    ) -> Tuple[bool, str]:
        """
        Validate a configuration and generate a report
        
        Args:
            config: Provider configuration to validate
            provider_class: Provider class to validate against
            test_connection: Whether to test the connection
            include_details: Whether to include detailed information
            
        Returns:
            Tuple of (is_valid, report)
        """
        results = await ConfigValidationSuite.run_validation_suite(
            config, provider_class, test_connection
        )
        
        # Check if valid (no errors)
        is_valid = not any(
            result.severity == "error" and not result.is_valid
            for category_results in results.values()
            for result in category_results
        )
        
        report = ConfigValidationSuite.format_validation_results(
            results, include_details
        )
        
        return is_valid, report


#
# ===== Provider-Specific Validation =====
#

class CCXTProviderConfig(ProviderConfig):
    """Configuration for CCXT provider with enhanced validation"""
    
    # Override defaults from base
    name: str = "crypto_ccxt"
    type: ProviderType = ProviderType.CRYPTO
    env_prefix: str = "CCXT_"
    require_auth: bool = True
    
    # CCXT-specific settings
    exchange: str = Field(..., description="Exchange ID (e.g., 'binance', 'coinbase')")
    symbols: List[str] = Field(..., description="List of symbols to track (e.g., 'BTC/USDT')")
    timeframe: str = Field(default="1m", description="Timeframe for OHLCV data")
    
    # Authentication
    api_keys: List[str] = Field(
        default_factory=lambda: ["API_KEY", "API_SECRET"],
        description="Required API key environment variables"
    )
    
    # Rate limiting
    rate_limit: Dict[str, Any] = Field(
        default_factory=lambda: {
            "requests_per_second": 5.0, 
            "burst": 10
        },
        description="Rate limiting configuration"
    )
    
    # Additional settings
    sandbox_mode: bool = Field(
        default=False, 
        description="Whether to use exchange sandbox mode"
    )
    default_limit: int = Field(
        default=100, 
        ge=1, 
        le=1000, 
        description="Default limit for data fetching"
    )
    fetch_ticker: bool = Field(
        default=True, 
        description="Whether to fetch ticker data"
    )
    fetch_ohlcv: bool = Field(
        default=True, 
        description="Whether to fetch OHLCV data"
    )
    fetch_order_book: bool = Field(
        default=False, 
        description="Whether to fetch order book data"
    )
    
    @validator('exchange')
    def validate_exchange(cls, v):
        """Validate exchange ID against CCXT supported exchanges"""
        if v not in ccxt.exchanges:
            supported = ", ".join(sorted(ccxt.exchanges)[:10]) + "..."
            raise ValueError(
                f"Invalid exchange: '{v}'. Not in list of supported CCXT exchanges. "
                f"Examples of supported exchanges: {supported}"
            )
        return v
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """Validate trading pair symbols format"""
        invalid_symbols = []
        
        for symbol in v:
            # Check basic format (e.g., 'BTC/USDT')
            if '/' not in symbol:
                invalid_symbols.append(f"{symbol} (missing '/')")
                continue
            
            base, quote = symbol.split('/')
            
            # Check that base and quote are non-empty
            if not base or not quote:
                invalid_symbols.append(f"{symbol} (empty base or quote)")
                continue
            
            # Check for common cryptocurrency format errors
            if not re.match(r'^[A-Za-z0-9._-]+$', base) or not re.match(r'^[A-Za-z0-9._-]+$', quote):
                invalid_symbols.append(f"{symbol} (invalid characters)")
                
        if invalid_symbols:
            raise ValueError(
                f"Invalid symbol formats: {', '.join(invalid_symbols)}. "
                f"Expected format: 'BTC/USDT'"
            )
        
        return v
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        """Validate timeframe format"""
        # Common timeframes in CCXT
        valid_timeframes = {
            '1m', '3m', '5m', '15m', '30m',  # minutes
            '1h', '2h', '4h', '6h', '8h', '12h',  # hours
            '1d', '3d',  # days
            '1w', '1M'  # week, month
        }
        
        if v not in valid_timeframes:
            raise ValueError(
                f"Invalid timeframe: '{v}'. "
                f"Valid timeframes: {', '.join(sorted(valid_timeframes))}"
            )
        
        return v
    
    @root_validator
    def validate_exchange_capabilities(cls, values):
        """Validate that requested operations are supported by the exchange"""
        exchange_id = values.get('exchange')
        fetch_ohlcv = values.get('fetch_ohlcv', True)
        fetch_order_book = values.get('fetch_order_book', False)
        
        # Skip validation if exchange is not specified
        if not exchange_id:
            return values
        
        try:
            # Check exchange capabilities without creating a full instance
            exchange_class = getattr(ccxt, exchange_id)
            exchange_has = exchange_class.has if hasattr(exchange_class, 'has') else {}
            
            # Validate OHLCV capability
            if fetch_ohlcv and not exchange_has.get('fetchOHLCV', False):
                logger.warning(
                    f"Exchange '{exchange_id}' does not support OHLCV data fetching "
                    f"but fetch_ohlcv is set to True. This operation will fail."
                )
            
            # Validate order book capability
            if fetch_order_book and not exchange_has.get('fetchOrderBook', False):
                logger.warning(
                    f"Exchange '{exchange_id}' does not support order book fetching "
                    f"but fetch_order_book is set to True. This operation will fail."
                )
                
        except AttributeError:
            # This should never happen if exchange validation passed, but just in case
            logger.warning(f"Could not verify capabilities for exchange '{exchange_id}'")
        
        return values
    
    @root_validator
    def validate_symbols_availability(cls, values):
        """Check that symbols exist on the exchange (if possible without making API calls)"""
        # This is a light validation that doesn't require API calls
        # A more thorough validation would happen when the provider connects
        
        exchange_id = values.get('exchange')
        symbols = values.get('symbols', [])
        
        if not exchange_id or not symbols:
            return values
            
        # Just log a reminder that symbols will be validated on connection
        logger.info(
            f"Symbols {symbols} will be validated against exchange '{exchange_id}' "
            f"when the provider connects"
        )
        
        return values


class YahooFinanceConfig(ProviderConfig):
    """Configuration for Yahoo Finance provider with enhanced validation"""
    
    # Override defaults from base
    name: str = "equities_yf"
    type: ProviderType = ProviderType.EQUITY
    env_prefix: str = "YF_"
    
    # YF-specific settings
    symbols: List[str] = Field(..., description="List of symbols to track (e.g., 'AAPL')")
    period: str = Field(default="1d", description="Data interval period")
    start_date: Optional[Union[str, datetime]] = Field(
        default=None, 
        description="Start date for historical data"
    )
    end_date: Optional[Union[str, datetime]] = Field(
        default=None, 
        description="End date for historical data"
    )
    
    # Additional settings
    adjust_prices: bool = Field(
        default=True, 
        description="Whether to adjust prices for splits and dividends"
    )
    include_dividends: bool = Field(
        default=False, 
        description="Whether to include dividend information"
    )
    include_fundamentals: bool = Field(
        default=False, 
        description="Whether to include fundamental data"
    )
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """Validate stock symbols format"""
        invalid_symbols = []
        
        for symbol in v:
            # Check basic format for stock symbols
            if not re.match(r'^[A-Za-z0-9.^-]+$', symbol):
                invalid_symbols.append(f"{symbol} (invalid characters)")
                continue
                
            # Common error: URLs instead of symbols
            if '/' in symbol or ':' in symbol or '.com' in symbol:
                invalid_symbols.append(f"{symbol} (appears to be a URL, not a symbol)")
                continue
                
        if invalid_symbols:
            raise ValueError(
                f"Invalid symbol formats: {', '.join(invalid_symbols)}. "
                f"Expected format: 'AAPL', 'MSFT', etc."
            )
        
        return v
    
    @validator('period')
    def validate_period(cls, v):
        """Validate data period format"""
        valid_periods = {'1d', '5d', '1wk', '1mo', '3mo'}
        
        if v not in valid_periods:
            raise ValueError(
                f"Invalid period: '{v}'. "
                f"Valid periods: {', '.join(sorted(valid_periods))}"
            )
        
        return v
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        """Validate and normalize date formats"""
        if v is None:
            return v
            
        if isinstance(v, datetime):
            return v
            
        if isinstance(v, str):
            try:
                # Try ISO format first
                return datetime.fromisoformat(v)
            except ValueError:
                try:
                    # Try common date formats
                    formats = ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d-%m-%Y']
                    for fmt in formats:
                        try:
                            return datetime.strptime(v, fmt)
                        except ValueError:
                            continue
                    
                    # If we get here, none of the formats worked
                    raise ValueError()
                except ValueError:
                    raise ValueError(
                        f"Invalid date format: '{v}'. "
                        f"Please use ISO format (YYYY-MM-DD) or common date formats"
                    )
        
        raise ValueError(f"Date must be a string or datetime object: {v}")
    
    @root_validator
    def validate_date_range(cls, values):
        """Validate that start_date is before end_date"""
        start_date = values.get('start_date')
        end_date = values.get('end_date')
        
        if start_date and end_date and start_date > end_date:
            raise ValueError(
                f"Start date ({start_date}) must be before end date ({end_date})"
            )
        
        return values


class OnChainConfig(ProviderConfig):
    """Configuration for blockchain data provider with enhanced validation"""
    
    # Override defaults from base
    name: str = "onchain"
    type: ProviderType = ProviderType.BLOCKCHAIN
    env_prefix: str = "ONCHAIN_"
    require_auth: bool = True
    
    # API sources
    api_source: str = Field(
        default="etherscan", 
        description="API source (etherscan, glassnode, etc.)"
    )
    
    # Authentication
    api_keys: List[str] = Field(
        default_factory=lambda: ["API_KEY"],
        description="Required API key environment variables"
    )
    
    # Blockchain settings
    network: str = Field(
        default="ethereum", 
        description="Blockchain network"
    )
    
    # Data types to fetch
    fetch_transactions: bool = Field(
        default=True, 
        description="Whether to fetch transaction data"
    )
    fetch_contracts: bool = Field(
        default=False, 
        description="Whether to fetch contract data"
    )
    fetch_metrics: bool = Field(
        default=False, 
        description="Whether to fetch on-chain metrics"
    )
    
    # Addresses to track
    addresses: List[str] = Field(
        default_factory=list, 
        description="List of blockchain addresses to track"
    )
    
    @validator('api_source')
    def validate_api_source(cls, v):
        """Validate API source"""
        valid_sources = {'etherscan', 'glassnode', 'blockchair', 'blockstream', 'mempool'}
        
        if v not in valid_sources:
            raise ValueError(
                f"Invalid API source: '{v}'. "
                f"Valid sources: {', '.join(sorted(valid_sources))}"
            )
        
        return v
    
    @validator('network')
    def validate_network(cls, v):
        """Validate blockchain network"""
        valid_networks = {
            'ethereum', 'bitcoin', 'solana', 'arbitrum',
            'optimism', 'polygon', 'avalanche', 'bsc'
        }
        
        if v not in valid_networks:
            raise ValueError(
                f"Invalid blockchain network: '{v}'. "
                f"Valid networks: {', '.join(sorted(valid_networks))}"
            )
        
        return v
    
    @validator('addresses')
    def validate_addresses(cls, v, values):
        """Validate blockchain addresses"""
        if not v:
            return v
        
        network = values.get('network', 'ethereum')
        invalid_addresses = []
        
        for addr in v:
            # Basic format validation based on network
            if network == 'ethereum':
                # Ethereum address validation
                if not re.match(r'^0x[a-fA-F0-9]{40}$', addr):
                    invalid_addresses.append(f"{addr} (invalid format for Ethereum)")
            elif network == 'bitcoin':
                # Bitcoin address validation (basic)
                if not (re.match(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$', addr) or 
                       re.match(r'^bc1[ac-hj-np-z02-9]{39,59}$', addr)):
                    invalid_addresses.append(f"{addr} (invalid format for Bitcoin)")
            else:
                # Basic check for other networks
                if not re.match(r'^[a-zA-Z0-9]{30,}$', addr):
                    invalid_addresses.append(f"{addr} (invalid format)")
        
        if invalid_addresses:
            raise ValueError(
                f"Invalid blockchain addresses for {network}: {', '.join(invalid_addresses)}"
            )
        
        return v
    
    @root_validator
    def validate_api_compatibility(cls, values):
        """Validate that requested operations are compatible with the API source"""
        api_source = values.get('api_source', 'etherscan')
        network = values.get('network', 'ethereum')
        fetch_metrics = values.get('fetch_metrics', False)
        
        # Check API source and network compatibility
        ethereum_sources = {'etherscan', 'glassnode'}
        bitcoin_sources = {'blockchair', 'blockstream', 'mempool', 'glassnode'}
        
        if network == 'ethereum' and api_source not in ethereum_sources:
            logger.warning(
                f"API source '{api_source}' may not support Ethereum network. "
                f"Consider using one of: {', '.join(ethereum_sources)}"
            )
        
        if network == 'bitcoin' and api_source not in bitcoin_sources:
            logger.warning(
                f"API source '{api_source}' may not support Bitcoin network. "
                f"Consider using one of: {', '.join(bitcoin_sources)}"
            )
        
        # Check metrics availability
        if fetch_metrics and api_source != 'glassnode':
            logger.warning(
                f"On-chain metrics are primarily available through Glassnode, "
                f"but api_source is set to '{api_source}'. This operation may fail."
            )
        
        return values


class TwitterSentimentConfig(ProviderConfig):
    """Configuration for Twitter sentiment provider with enhanced validation"""
    
    # Override defaults from base
    name: str = "sentiment_twitter"
    type: ProviderType = ProviderType.SENTIMENT
    env_prefix: str = "TWITTER_"
    require_auth: bool = True
    
    # Authentication
    api_keys: List[str] = Field(
        default_factory=lambda: ["API_KEY", "API_SECRET", "ACCESS_TOKEN", "ACCESS_SECRET"],
        description="Required API key environment variables"
    )
    
    # Query settings
    keywords: List[str] = Field(
        default_factory=list, 
        description="Keywords to track"
    )
    hashtags: List[str] = Field(
        default_factory=list, 
        description="Hashtags to track"
    )
    accounts: List[str] = Field(
        default_factory=list, 
        description="Twitter accounts to track"
    )
    
    # Filter settings
    languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Languages to include (ISO 639-1 codes)"
    )
    exclude_retweets: bool = Field(
        default=True, 
        description="Whether to exclude retweets"
    )
    min_followers: int = Field(
        default=0, 
        ge=0, 
        description="Minimum followers for included accounts"
    )
    
    # Sentiment analysis
    sentiment_model: str = Field(
        default="transformers",
        description="Sentiment analysis model to use"
    )
    batch_size: int = Field(
        default=32, 
        ge=1, 
        le=256, 
        description="Batch size for sentiment analysis"
    )
    
    @validator('keywords', 'hashtags')
    def validate_query_terms(cls, v, values, **kwargs):
        """Validate search terms"""
        field_name = kwargs.get('field_name', 'terms')
        
        if not v:
            return v
        
        invalid_terms = []
        for term in v:
            # Check for empty or overly short terms
            if not term or len(term) < 2:
                invalid_terms.append(f"'{term}' (too short)")
                continue
            
            # Check for potentially problematic characters
            if re.search(r'[\'"\\]', term):
                invalid_terms.append(f"'{term}' (contains problematic characters)")
        
        if invalid_terms:
            raise ValueError(
                f"Invalid {field_name}: {', '.join(invalid_terms)}. "
                f"Terms should be at least 2 characters and avoid special characters."
            )
        
        return v
    
    @validator('accounts')
    def validate_accounts(cls, v):
        """Validate Twitter account handles"""
        if not v:
            return v
        
        invalid_accounts = []
        for account in v:
            # Remove @ prefix if present
            account_clean = account[1:] if account.startswith('@') else account
            
            # Check for valid Twitter handle format
            if not re.match(r'^[A-Za-z0-9_]{1,15}$', account_clean):
                invalid_accounts.append(
                    f"'{account}' (invalid format, should be 1-15 alphanumeric or underscore characters)"
                )
        
        if invalid_accounts:
            raise ValueError(
                f"Invalid Twitter handles: {', '.join(invalid_accounts)}. "
                f"Handles should match Twitter's format requirements."
            )
        
        # Normalize handles by adding @ prefix if missing
        return [f"@{a}" if not a.startswith('@') else a for a in v]
    
    @validator('languages')
    def validate_languages(cls, v):
        """Validate language codes"""
        valid_languages = {
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi', 'nl',
            'no', 'sv', 'fi', 'da', 'pl', 'hu', 'fa', 'he', 'ur', 'th'
        }
        
        invalid_langs = [lang for lang in v if lang not in valid_languages]
        
        if invalid_langs:
            raise ValueError(
                f"Invalid language codes: {', '.join(invalid_langs)}. "
                f"Please use ISO 639-1 codes (e.g., 'en' for English)."
            )
        
        return v
    
    @validator('sentiment_model')
    def validate_sentiment_model(cls, v):
        """Validate sentiment analysis model"""
        valid_models = {'transformers', 'vader', 'textblob', 'flair', 'custom'}
        
        if v not in valid_models:
            raise ValueError(
                f"Invalid sentiment model: '{v}'. "
                f"Valid models: {', '.join(sorted(valid_models))}"
            )
        
        return v
    
    @root_validator
    def validate_search_criteria(cls, values):
        """Validate that at least one search criterion is provided"""
        keywords = values.get('keywords', [])
        hashtags = values.get('hashtags', [])
        accounts = values.get('accounts', [])
        
        if not keywords and not hashtags and not accounts:
            raise ValueError(
                "At least one search criterion (keywords, hashtags, or accounts) must be provided"
            )
        
        return values

#
# ===== Convenience Functions =====
#

async def validate_provider_config(config: ProviderConfig, provider_class: Type[BaseProvider],
                                 test_connection: bool = True) -> Tuple[bool, str]:
    """
    Convenience function to validate a provider configuration
    
    Args:
        config: Provider configuration
        provider_class: Provider class
        test_connection: Whether to test the connection
        
    Returns:
        Tuple of (is_valid, report)
    """
    return await ConfigValidationSuite.validate_and_report(
        config, provider_class, test_connection, include_details=True
    )


def get_provider_config_class(provider_type: ProviderType) -> Type[ProviderConfig]:
    """
    Get the appropriate provider config class for a provider type
    
    Args:
        provider_type: Type of provider
        
    Returns:
        Provider configuration class
    """
    config_map = {
        ProviderType.CRYPTO: CCXTProviderConfig,
        ProviderType.EQUITY: YahooFinanceConfig,
        ProviderType.BLOCKCHAIN: OnChainConfig,
        ProviderType.SENTIMENT: TwitterSentimentConfig,
        # For other types, return the base ProviderConfig
    }
    
    return config_map.get(provider_type, ProviderConfig)