# src/quant_research/core/config/schemas/connection.py
"""
Connection configuration schemas for data providers.

This module defines the structure of connection configurations without validation logic.
Validation rules are defined separately in the validation module.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ConnectionConfig(BaseModel):
    """
    Connection configuration for data providers.
    
    Defines parameters for connection pooling, timeouts, and retry behavior.
    """
    
    # Connection settings
    timeout: int = Field(
        default=30, 
        description="Connection timeout in seconds"
    )
    pool_size: int = Field(
        default=5,
        description="Maximum number of connections to maintain in the pool"
    )
    keep_alive: bool = Field(
        default=True, 
        description="Whether to keep connections alive between requests"
    )
    
    # Retry settings
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Base delay between retries in seconds"
    )
    
    # Additional options
    ssl_verify: bool = Field(
        default=True,
        description="Whether to verify SSL certificates"
    )
    proxy: Optional[str] = Field(
        default=None,
        description="Optional proxy URL for connections"
    )
    
    # Extended connection options
    connection_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional connection-specific options"
    )
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "allow"


class RetryConfig(BaseModel):
    """
    Configuration for advanced retry behavior.
    
    Can be used to extend the basic retry settings in ConnectionConfig.
    """
    
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    base_delay: float = Field(
        default=1.0,
        description="Base delay between retries in seconds"
    )
    max_delay: float = Field(
        default=60.0,
        description="Maximum delay between retries in seconds"
    )
    jitter: bool = Field(
        default=True,
        description="Whether to add randomized jitter to retry delays"
    )
    jitter_factor: float = Field(
        default=0.1,
        description="Factor for jitter calculation (0.0-1.0)"
    )
    retry_on: Optional[Dict[str, bool]] = Field(
        default=None,
        description="Map of error types to retry decision (True/False)"
    )
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "allow"


class PoolConfig(BaseModel):
    """
    Configuration for advanced connection pooling.
    
    Can be used to extend the basic pool settings in ConnectionConfig.
    """
    
    min_size: int = Field(
        default=1,
        description="Minimum number of connections to maintain in the pool"
    )
    max_size: int = Field(
        default=10,
        description="Maximum number of connections to maintain in the pool"
    )
    max_idle: int = Field(
        default=5,
        description="Maximum number of idle connections to keep in the pool"
    )
    idle_timeout: int = Field(
        default=300,
        description="Time in seconds before idle connections are closed"
    )
    max_age: Optional[int] = Field(
        default=None,
        description="Maximum age of a connection in seconds (None for no limit)"
    )
    health_check_interval: int = Field(
        default=30,
        description="Interval in seconds for connection health checks"
    )
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "allow"