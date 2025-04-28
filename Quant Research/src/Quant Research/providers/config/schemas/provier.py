# src/quant_research/core/config/schemas/provider.py
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class ProviderType(str, Enum):
    """Types of data providers"""
    CRYPTO = "crypto"
    EQUITY = "equity"
    BLOCKCHAIN = "blockchain"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"


class ConnectionConfig(BaseModel):
    """Connection configuration for data providers"""
    
    # Connection settings
    timeout: int = Field(
        default=30, 
        ge=1, 
        le=300, 
        description="Connection timeout in seconds"
    )
    pool_size: int = Field(
        default=5, 
        ge=1, 
        le=100, 
        description="Maximum number of connections to maintain in the pool"
    )
    keep_alive: bool = Field(
        default=True, 
        description="Whether to keep connections alive between requests"
    )
    
    # Retry settings
    max_retries: int = Field(
        default=3, 
        ge=0, 
        le=10, 
        description="Maximum number of retry attempts"
    )
    retry_delay: float = Field(
        default=1.0, 
        ge=0.1, 
        le=60.0, 
        description="Base delay between retries in seconds"
    )
    
    class Config:
        """Pydantic config"""
        extra = "allow"


class ProviderConfig(BaseModel):
    """Base configuration for data providers"""
    
    # Provider identification
    name: str = Field(
        ..., 
        min_length=1, 
        max_length=50, 
        description="Provider name"
    )
    type: ProviderType = Field(
        default=ProviderType.CUSTOM, 
        description="Provider type category"
    )
    
    # Connection settings
    connection: ConnectionConfig = Field(
        default_factory=ConnectionConfig, 
        description="Connection settings"
    )
    
    # Environment variables
    env_prefix: str = Field(
        default="", 
        description="Prefix for environment variables"
    )
    
    # Cache settings
    enable_cache: bool = Field(
        default=True, 
        description="Whether to enable caching"
    )
    cache_ttl: int = Field(
        default=300, 
        ge=0, 
        description="Time-to-live for cache entries in seconds"
    )
    
    # Rate limiting
    rate_limit: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Rate limiting configuration"
    )
    
    # Validation-related fields
    require_auth: bool = Field(
        default=False, 
        description="Whether authentication is required"
    )
    api_keys: List[str] = Field(
        default_factory=list, 
        description="List of required API key names"
    )
    
    # Additional custom provider settings
    settings: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional provider-specific settings"
    )
    
    class Config:
        """Pydantic config"""
        validate_assignment = True
        extra = "allow"
        json_encoders = {
            # Add custom encoders as needed
        }
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get an API key from environment variables using the configured prefix
        
        Args:
            key_name: Name of the API key
            
        Returns:
            API key value or None if not found
        """
        import os
        env_var = f"{self.env_prefix}{key_name}"
        return os.getenv(env_var)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, removing sensitive data"""
        data = self.dict(exclude_none=True)
        
        # Remove potential sensitive information
        if 'api_keys' in data:
            data['api_keys'] = [f"{key}=***" for key in data['api_keys']]
        
        return data