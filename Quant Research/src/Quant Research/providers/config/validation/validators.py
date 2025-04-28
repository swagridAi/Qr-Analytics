# src/quant_research/core/config/validation/validators.py
"""
Reusable validators for configuration validation.

This module provides a collection of validator functions that can be used
with the configuration validation framework. Each validator follows a consistent
pattern, accepting a value and returning a ValidationResult.
"""

import re
import ipaddress
from datetime import datetime, timedelta
from typing import Any, List, Optional, Union, Dict, Callable
import logging

from .result import ValidationResult

logger = logging.getLogger(__name__)

#
# Basic Field Validators
#

def validate_provider_name(name: str) -> ValidationResult:
    """Validate provider name format"""
    if not re.match(r'^[a-z][a-z0-9_]*$', name):
        return ValidationResult(
            is_valid=False,
            message="Provider name must start with a lowercase letter and contain only lowercase letters, numbers, and underscores",
            severity="error"
        )
    return ValidationResult(is_valid=True, message="Provider name is valid", severity="info")


def validate_non_empty_string(value: str, field_name: str = "field") -> ValidationResult:
    """Validate that a string is not empty"""
    if not value or not value.strip():
        return ValidationResult(
            is_valid=False,
            message=f"{field_name} cannot be empty",
            severity="error"
        )
    return ValidationResult(is_valid=True, message=f"{field_name} is valid", severity="info")


#
# Symbol Validators
#

def validate_crypto_symbol(symbol: str) -> ValidationResult:
    """Validate cryptocurrency trading pair symbol format"""
    if '/' not in symbol:
        return ValidationResult(
            is_valid=False,
            message=f"Invalid symbol format: {symbol}. Expected format: 'BTC/USDT'",
            severity="error"
        )
    base, quote = symbol.split('/')
    if not base or not quote:
        return ValidationResult(
            is_valid=False,
            message=f"Invalid symbol format: {symbol}. Empty base or quote",
            severity="error"
        )
    return ValidationResult(is_valid=True, message="Symbol format is valid", severity="info")


def validate_equity_symbol(symbol: str) -> ValidationResult:
    """Validate equity symbol format"""
    if not re.match(r'^[A-Za-z0-9.^-]+$', symbol):
        return ValidationResult(
            is_valid=False,
            message=f"Invalid symbol format: {symbol}. Expected format: 'AAPL'",
            severity="error"
        )
    return ValidationResult(is_valid=True, message="Symbol format is valid", severity="info")


def validate_symbol_format(symbol: str, provider_type: str = None) -> ValidationResult:
    """
    Validate symbol format based on provider type
    
    Args:
        symbol: Symbol to validate
        provider_type: Provider type to determine validation rules
        
    Returns:
        ValidationResult with validation outcome
    """
    if provider_type == "crypto":
        return validate_crypto_symbol(symbol)
    elif provider_type == "equity":
        return validate_equity_symbol(symbol)
    else:
        # Try to infer from symbol format
        if '/' in symbol:
            return validate_crypto_symbol(symbol)
        else:
            return validate_equity_symbol(symbol)


def validate_symbols_list(symbols: List[str], provider_type: str = None) -> List[ValidationResult]:
    """
    Validate a list of symbols
    
    Args:
        symbols: List of symbols to validate
        provider_type: Provider type to determine validation rules
        
    Returns:
        List of ValidationResults, one for each symbol
    """
    results = []
    
    if not symbols:
        results.append(ValidationResult(
            is_valid=False,
            message="Symbol list cannot be empty",
            severity="error"
        ))
        return results
    
    # Validate each symbol
    for symbol in symbols:
        results.append(validate_symbol_format(symbol, provider_type))
    
    return results


#
# Timeframe Validators
#

def validate_timeframe(timeframe: str, allowed_values: Optional[List[str]] = None) -> ValidationResult:
    """
    Validate timeframe format
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')
        allowed_values: Optional list of allowed timeframe values
        
    Returns:
        ValidationResult with validation outcome
    """
    # Default allowed timeframes if not specified
    default_timeframes = {
        '1m', '3m', '5m', '15m', '30m',  # minutes
        '1h', '2h', '4h', '6h', '8h', '12h',  # hours
        '1d', '3d',  # days
        '1w', '1M'  # week, month
    }
    
    allowed = allowed_values or default_timeframes
    
    if timeframe not in allowed:
        return ValidationResult(
            is_valid=False,
            message=f"Invalid timeframe: '{timeframe}'. Valid timeframes: {', '.join(sorted(allowed))}",
            severity="error"
        )
    
    return ValidationResult(is_valid=True, message="Timeframe format is valid", severity="info")


#
# Connection Parameter Validators
#

def validate_timeout(timeout: int) -> ValidationResult:
    """
    Validate connection timeout value
    
    Args:
        timeout: Timeout value in seconds
        
    Returns:
        ValidationResult with validation outcome
    """
    if timeout <= 0:
        return ValidationResult(
            is_valid=False,
            message="Timeout must be positive",
            severity="error"
        )
    
    if timeout > 300:
        return ValidationResult(
            is_valid=True,  # Still valid, just a warning
            message=f"Timeout value {timeout}s is unusually high and may lead to resource issues",
            severity="warning"
        )
    
    return ValidationResult(is_valid=True, message="Timeout value is valid", severity="info")


def validate_pool_size(pool_size: int) -> ValidationResult:
    """
    Validate connection pool size
    
    Args:
        pool_size: Connection pool size
        
    Returns:
        ValidationResult with validation outcome
    """
    if pool_size <= 0:
        return ValidationResult(
            is_valid=False,
            message="Pool size must be positive",
            severity="error"
        )
    
    if pool_size > 50:
        return ValidationResult(
            is_valid=True,  # Still valid, just a warning
            message=f"Pool size {pool_size} is unusually high and may consume excessive resources",
            severity="warning"
        )
    
    return ValidationResult(is_valid=True, message="Pool size is valid", severity="info")


def validate_retry_config(max_retries: int, retry_delay: float) -> ValidationResult:
    """
    Validate retry configuration
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries in seconds
        
    Returns:
        ValidationResult with validation outcome
    """
    if max_retries < 0:
        return ValidationResult(
            is_valid=False,
            message="Maximum retries cannot be negative",
            severity="error"
        )
    
    if retry_delay <= 0:
        return ValidationResult(
            is_valid=False,
            message="Retry delay must be positive",
            severity="error"
        )
    
    if retry_delay > 60:
        return ValidationResult(
            is_valid=True,  # Still valid, just a warning
            message=f"Retry delay of {retry_delay}s is unusually high",
            severity="warning"
        )
    
    return ValidationResult(is_valid=True, message="Retry configuration is valid", severity="info")


def validate_connection_config(timeout: int, pool_size: int, keep_alive: bool, 
                             max_retries: int, retry_delay: float) -> List[ValidationResult]:
    """
    Validate complete connection configuration
    
    Args:
        timeout: Connection timeout in seconds
        pool_size: Connection pool size
        keep_alive: Whether to keep connections alive
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries in seconds
        
    Returns:
        List of ValidationResults for each aspect of the configuration
    """
    results = []
    
    # Validate timeout
    results.append(validate_timeout(timeout))
    
    # Validate pool size
    results.append(validate_pool_size(pool_size))
    
    # Validate retry config
    results.append(validate_retry_config(max_retries, retry_delay))
    
    # Specific recommendations for keep_alive
    if not keep_alive and pool_size > 1:
        results.append(ValidationResult(
            is_valid=True,  # Still valid, just a recommendation
            message="Using pool_size > 1 without keep_alive may create unnecessary connections",
            severity="warning"
        ))
    
    return results


#
# URL and API Validators
#

def validate_url(url: str) -> ValidationResult:
    """
    Validate URL format
    
    Args:
        url: URL to validate
        
    Returns:
        ValidationResult with validation outcome
    """
    url_pattern = re.compile(
        r'^(https?:\/\/)?'  # http:// or https://
        r'([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+' # domain
        r'[a-zA-Z]{2,}'  # TLD
        r'(:\d{1,5})?'  # optional port
        r'(\/[-a-zA-Z0-9%_.~#?&=]*)*'  # optional path, query, etc.
        r'$'
    )
    
    if not url_pattern.match(url):
        return ValidationResult(
            is_valid=False,
            message=f"Invalid URL format: {url}",
            severity="error"
        )
    
    # Check for HTTP/HTTPS
    if not url.startswith('http://') and not url.startswith('https://'):
        return ValidationResult(
            is_valid=True,  # Still valid but should have protocol
            message=f"URL {url} does not specify protocol (http:// or https://)",
            severity="warning"
        )
    
    # Prefer HTTPS
    if url.startswith('http://'):
        return ValidationResult(
            is_valid=True,  # Still valid but should use HTTPS
            message=f"URL {url} uses HTTP protocol which is not secure",
            severity="warning"
        )
    
    return ValidationResult(is_valid=True, message="URL format is valid", severity="info")


def validate_api_endpoint(endpoint: str) -> ValidationResult:
    """
    Validate API endpoint format
    
    Args:
        endpoint: API endpoint to validate
        
    Returns:
        ValidationResult with validation outcome
    """
    # If it's a full URL, validate as URL
    if endpoint.startswith('http://') or endpoint.startswith('https://'):
        return validate_url(endpoint)
    
    # Otherwise, validate as a path
    if not endpoint.startswith('/'):
        return ValidationResult(
            is_valid=True,  # Still valid but unusual
            message=f"API endpoint '{endpoint}' doesn't start with '/', which is unusual for endpoints",
            severity="warning"
        ))
    
    # Check for path parameters
    if '{' in endpoint or '}' in endpoint:
        # Check for balanced braces
        if endpoint.count('{') != endpoint.count('}'):
            return ValidationResult(
                is_valid=False,
                message=f"API endpoint '{endpoint}' has unbalanced path parameter braces",
                severity="error"
            )
    
    return ValidationResult(is_valid=True, message="API endpoint format is valid", severity="info")


#
# Authentication Validators
#

def validate_api_key_format(api_key: str, min_length: int = 10) -> ValidationResult:
    """
    Validate API key format
    
    Args:
        api_key: API key to validate
        min_length: Minimum expected length for API keys
        
    Returns:
        ValidationResult with validation outcome
    """
    if len(api_key) < min_length:
        return ValidationResult(
            is_valid=False,
            message=f"API key is too short (less than {min_length} characters)",
            severity="error"
        )
    
    # Check if it's just alphanumeric
    if api_key.isalnum():
        # Most API keys have some special chars or are base64, so this is unusual
        return ValidationResult(
            is_valid=True,
            message=f"API key is purely alphanumeric, which is unusual for secure API keys",
            severity="warning"
        )
    
    return ValidationResult(is_valid=True, message="API key format appears valid", severity="info")


def validate_env_var_name(env_var: str) -> ValidationResult:
    """
    Validate environment variable name format
    
    Args:
        env_var: Environment variable name
        
    Returns:
        ValidationResult with validation outcome
    """
    if not re.match(r'^[A-Z][A-Z0-9_]*$', env_var):
        return ValidationResult(
            is_valid=False,
            message=f"Invalid environment variable name: {env_var}. Should be uppercase with underscores (e.g., API_KEY)",
            severity="error"
        )
    
    return ValidationResult(is_valid=True, message="Environment variable name format is valid", severity="info")


#
# Provider-Specific Validators
#

def validate_ccxt_exchange(exchange: str) -> ValidationResult:
    """
    Validate CCXT exchange name
    
    Args:
        exchange: Exchange name to validate
        
    Returns:
        ValidationResult with validation outcome
    """
    # This would normally import ccxt and check against ccxt.exchanges
    # For simplicity, we'll just check common exchanges
    common_exchanges = {
        'binance', 'coinbase', 'kraken', 'kucoin', 'ftx', 'bitstamp',
        'bitfinex', 'huobi', 'okex', 'bitmex', 'bybit'
    }
    
    if exchange not in common_exchanges:
        return ValidationResult(
            is_valid=True,  # Can't determine for sure without ccxt
            message=f"Exchange '{exchange}' is not in the list of common exchanges. Please verify it exists.",
            severity="warning"
        )
    
    return ValidationResult(is_valid=True, message=f"Exchange '{exchange}' is recognized", severity="info")


def validate_blockchain_network(network: str) -> ValidationResult:
    """
    Validate blockchain network name
    
    Args:
        network: Blockchain network name
        
    Returns:
        ValidationResult with validation outcome
    """
    valid_networks = {
        'ethereum', 'bitcoin', 'solana', 'arbitrum',
        'optimism', 'polygon', 'avalanche', 'bsc'
    }
    
    if network.lower() not in valid_networks:
        return ValidationResult(
            is_valid=False,
            message=f"Invalid blockchain network: '{network}'. Valid networks: {', '.join(sorted(valid_networks))}",
            severity="error"
        )
    
    return ValidationResult(is_valid=True, message=f"Blockchain network '{network}' is valid", severity="info")


#
# Configuration Validators
#

def validate_rate_limit_config(config: Dict[str, Any]) -> ValidationResult:
    """
    Validate rate limiting configuration
    
    Args:
        config: Rate limit configuration dictionary
        
    Returns:
        ValidationResult with validation outcome
    """
    required_fields = ['requests_per_second', 'burst']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        return ValidationResult(
            is_valid=False,
            message=f"Rate limit configuration missing required fields: {', '.join(missing_fields)}",
            severity="error"
        )
    
    # Check requests_per_second
    rps = config.get('requests_per_second', 0)
    if rps <= 0:
        return ValidationResult(
            is_valid=False,
            message="requests_per_second must be positive",
            severity="error"
        )
    
    # Check burst
    burst = config.get('burst', 0)
    if burst <= 0:
        return ValidationResult(
            is_valid=False,
            message="burst must be positive",
            severity="error"
        )
    
    # Check unusually high values
    if rps > 100:
        return ValidationResult(
            is_valid=True,  # Still valid, just a warning
            message=f"requests_per_second of {rps} is unusually high and may trigger API rate limits",
            severity="warning"
        )
    
    return ValidationResult(is_valid=True, message="Rate limit configuration is valid", severity="info")


#
# Validator Registry
#

# Validator registry for easy access
VALIDATORS = {
    # Basic validators
    "provider_name": validate_provider_name,
    "non_empty_string": validate_non_empty_string,
    
    # Symbol validators
    "crypto_symbol": validate_crypto_symbol,
    "equity_symbol": validate_equity_symbol,
    "symbol_format": validate_symbol_format,
    "symbols_list": validate_symbols_list,
    
    # Timeframe validators
    "timeframe": validate_timeframe,
    
    # Connection validators
    "timeout": validate_timeout,
    "pool_size": validate_pool_size,
    "retry_config": validate_retry_config,
    "connection_config": validate_connection_config,
    
    # URL and API validators
    "url": validate_url,
    "api_endpoint": validate_api_endpoint,
    
    # Authentication validators
    "api_key_format": validate_api_key_format,
    "env_var_name": validate_env_var_name,
    
    # Provider-specific validators
    "ccxt_exchange": validate_ccxt_exchange,
    "blockchain_network": validate_blockchain_network,
    
    # Configuration validators
    "rate_limit_config": validate_rate_limit_config,
}


def get_validator(name: str) -> Optional[Callable]:
    """
    Get a validator by name
    
    Args:
        name: Validator name
        
    Returns:
        Validator function or None if not found
    """
    return VALIDATORS.get(name)


def get_all_validators() -> Dict[str, Callable]:
    """
    Get all available validators
    
    Returns:
        Dictionary of validator name to validator function
    """
    return VALIDATORS.copy()