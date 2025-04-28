# src/quant_research/core/config.py
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, root_validator
import os
import re
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Types of data providers"""
    CRYPTO = "crypto"
    EQUITIES = "equities"
    ONCHAIN = "onchain"
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
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError("Timeout must be positive")
        if v > 300:
            logger.warning(f"Timeout value {v}s is unusually high")
        return v
    
    @validator('pool_size')
    def validate_pool_size(cls, v):
        if v <= 0:
            raise ValueError("Pool size must be positive")
        if v > 50:
            logger.warning(f"Pool size {v} is unusually high and may consume excessive resources")
        return v


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
            Path: str
        }
    
    @validator('name')
    def validate_name(cls, v):
        """Validate provider name format"""
        if not re.match(r'^[a-z][a-z0-9_]*$', v):
            raise ValueError(
                "Provider name must start with a lowercase letter and "
                "contain only lowercase letters, numbers, and underscores"
            )
        return v
    
    @validator('env_prefix')
    def validate_env_prefix(cls, v):
        """Validate environment variable prefix"""
        if v and not v.endswith('_'):
            # Automatically add trailing underscore if missing
            v = f"{v}_"
        return v
    
    @root_validator
    def validate_auth_requirements(cls, values):
        """Validate authentication requirements"""
        require_auth = values.get('require_auth', False)
        api_keys = values.get('api_keys', [])
        
        if require_auth and not api_keys:
            raise ValueError(
                "Provider requires authentication but no API key names specified. "
                "Please add required key names to api_keys field."
            )
        
        # Check if required API keys are in environment variables
        if require_auth and api_keys:
            env_prefix = values.get('env_prefix', '')
            missing_keys = []
            
            for key in api_keys:
                env_var = f"{env_prefix}{key}"
                if not os.getenv(env_var):
                    missing_keys.append(env_var)
            
            if missing_keys:
                logger.warning(
                    f"Missing required environment variables for authentication: {', '.join(missing_keys)}. "
                    f"Provider '{values.get('name')}' may fail to authenticate."
                )
        
        return values
    
    @root_validator
    def validate_rate_limit_config(cls, values):
        """Validate rate limiting configuration"""
        rate_limit = values.get('rate_limit')
        
        if rate_limit:
            if not isinstance(rate_limit, dict):
                raise ValueError("Rate limit configuration must be a dictionary")
            
            # Check for required rate limit fields
            required_fields = ['requests_per_second', 'burst']
            missing_fields = [field for field in required_fields if field not in rate_limit]
            
            if missing_fields:
                raise ValueError(f"Rate limit configuration missing required fields: {', '.join(missing_fields)}")
            
            # Validate rate limit values
            if rate_limit.get('requests_per_second', 0) <= 0:
                raise ValueError("requests_per_second must be positive")
            
            if rate_limit.get('burst', 0) <= 0:
                raise ValueError("burst must be positive")
        
        return values
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get an API key from environment variables using the configured prefix
        
        Args:
            key_name: Name of the API key
            
        Returns:
            API key value or None if not found
        """
        env_var = f"{self.env_prefix}{key_name}"
        return os.getenv(env_var)
    
    def validate_required_settings(self, required_settings: List[str]) -> List[str]:
        """
        Validate that required settings are present
        
        Args:
            required_settings: List of required setting names
            
        Returns:
            List of missing settings, empty if all required settings are present
        """
        missing = []
        for setting in required_settings:
            if setting not in self.settings:
                missing.append(setting)
        return missing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, removing sensitive data"""
        data = self.dict(exclude_none=True)
        
        # Remove potential sensitive information
        if 'api_keys' in data:
            data['api_keys'] = [f"{key}=***" for key in data['api_keys']]
        
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string, removing sensitive data"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file
        
        Args:
            file_path: Path to save the configuration file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(self.to_json())
        
        logger.info(f"Saved provider configuration to {file_path}")
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'ProviderConfig':
        """
        Load configuration from a JSON file
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Provider configuration
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)