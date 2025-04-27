# src/quant_research/core/config.py
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import yaml
import json
import os
from pathlib import Path


class ProviderType(str, Enum):
    """Enum of supported provider types"""
    CRYPTO = "crypto"
    EQUITIES = "equities"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"


class ConnectionConfig(BaseModel):
    """Configuration for connection settings"""
    timeout: int = Field(default=30, description="Connection timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    retry_delay: int = Field(default=5, description="Delay between retries in seconds")
    pool_size: int = Field(default=10, description="Connection pool size")
    keep_alive: bool = Field(default=True, description="Whether to keep connection alive")


class ProviderConfig(BaseModel):
    """Base configuration for data providers"""
    name: str = Field(..., description="Provider name")
    type: ProviderType = Field(..., description="Provider type")
    enabled: bool = Field(default=True, description="Whether the provider is enabled")
    connection: ConnectionConfig = Field(
        default_factory=ConnectionConfig,
        description="Connection settings"
    )
    
    # Provider-specific settings
    settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific settings"
    )
    
    # Authentication can be provided directly or via environment variables
    auth: Dict[str, Any] = Field(
        default_factory=dict,
        description="Authentication details"
    )
    
    # Environment variable prefix for this provider
    env_prefix: Optional[str] = Field(
        default=None,
        description="Prefix for environment variables"
    )
    
    @validator('auth', pre=True)
    def load_auth_from_env(cls, v, values):
        """Load authentication from environment variables if specified"""
        result = dict(v or {})
        
        # If env_prefix is set, look for environment variables
        if 'env_prefix' in values and values['env_prefix']:
            prefix = values['env_prefix']
            
            # Common auth keys
            for key in ['api_key', 'api_secret', 'token', 'username', 'password']:
                env_key = f"{prefix}_{key}".upper()
                if env_key in os.environ and key not in result:
                    result[key] = os.environ[env_key]
        
        return result
    
    class Config:
        extra = "allow"  # Allow extra fields for provider-specific configs


class ProvidersConfig(BaseModel):
    """Configuration for all providers"""
    providers: List[ProviderConfig] = Field(
        default_factory=list,
        description="List of provider configurations"
    )
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "ProvidersConfig":
        """Load configuration from a YAML or JSON file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.suffix in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif file_path.suffix == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return cls.parse_obj(config_data)
    
    def get_provider_config(self, name: str) -> Optional[ProviderConfig]:
        """Get a specific provider configuration by name"""
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None
    
    def get_enabled_providers(self, provider_type: Optional[ProviderType] = None) -> List[ProviderConfig]:
        """Get all enabled providers, optionally filtered by type"""
        providers = [p for p in self.providers if p.enabled]
        
        if provider_type:
            providers = [p for p in providers if p.type == provider_type]
            
        return providers