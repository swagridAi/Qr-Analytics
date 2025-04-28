# src/quant_research/core/credentials.py
import os
import logging
import json
import base64
import hashlib
import re
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List, Set, Callable
from pathlib import Path

import keyring
from pydantic import BaseModel, Field, validator, SecretStr
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


logger = logging.getLogger(__name__)


class CredentialSource(str, Enum):
    """Source types for credentials"""
    ENV = "environment"
    KEYRING = "system_keyring"
    VAULT = "hashicorp_vault"
    FILE = "encrypted_file"
    CONFIG = "config_direct"  # Least secure, should be avoided


class CredentialType(str, Enum):
    """Types of credentials for providers"""
    API_KEY = "api_key"
    API_SECRET = "api_secret"
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    PASSWORD = "password"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    CONNECTION_STRING = "connection_string"


class CredentialConfig(BaseModel):
    """Configuration for a credential"""
    
    # Identifier
    name: str = Field(..., description="Unique name for this credential")
    
    # Metadata
    provider_id: str = Field(..., description="Provider this credential belongs to")
    credential_type: CredentialType = Field(..., description="Type of credential")
    description: Optional[str] = Field(None, description="Human-readable description")
    
    # Source configuration
    source: CredentialSource = Field(CredentialSource.ENV, description="Where to load from")
    env_var: Optional[str] = Field(None, description="Environment variable name if source is ENV")
    keyring_service: Optional[str] = Field(None, description="Keyring service name if source is KEYRING")
    vault_path: Optional[str] = Field(None, description="Vault secret path if source is VAULT")
    file_path: Optional[str] = Field(None, description="Path to credential file if source is FILE")
    
    # Direct value - discouraged except for testing
    value: Optional[SecretStr] = Field(None, description="Direct credential value (NOT RECOMMENDED)")
    
    # Security & validation
    required: bool = Field(True, description="Whether this credential is required")
    pattern: Optional[str] = Field(None, description="Regex pattern for validation")
    expires_at: Optional[datetime] = Field(None, description="Expiration date for this credential")
    rotation_days: Optional[int] = Field(None, description="Recommended rotation in days")
    last_rotated: Optional[datetime] = Field(None, description="When credential was last rotated")
    
    @validator('env_var')
    def validate_env_var(cls, v, values):
        """Validate environment variable name is provided if source is ENV"""
        if values.get('source') == CredentialSource.ENV and not v:
            raise ValueError("Environment variable name is required when source is ENV")
        return v
    
    @validator('keyring_service')
    def validate_keyring_service(cls, v, values):
        """Validate keyring service name is provided if source is KEYRING"""
        if values.get('source') == CredentialSource.KEYRING and not v:
            raise ValueError("Keyring service name is required when source is KEYRING")
        return v
    
    @validator('vault_path')
    def validate_vault_path(cls, v, values):
        """Validate vault path is provided if source is VAULT"""
        if values.get('source') == CredentialSource.VAULT and not v:
            raise ValueError("Vault path is required when source is VAULT")
        return v
    
    @validator('file_path')
    def validate_file_path(cls, v, values):
        """Validate file path is provided if source is FILE"""
        if values.get('source') == CredentialSource.FILE and not v:
            raise ValueError("File path is required when source is FILE")
        return v
    
    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True


class CredentialManagerConfig(BaseModel):
    """Configuration for the credential manager"""
    
    keyring_app_name: str = Field(default="quant_research", description="App name for system keyring")
    vault_addr: Optional[str] = Field(None, description="HashiCorp Vault address")
    vault_token_env_var: str = Field(default="VAULT_TOKEN", description="Env var for Vault token")
    encryption_key_env_var: str = Field(default="QUANT_RESEARCH_ENCRYPTION_KEY", 
                                        description="Env var for encryption key")
    mask_credentials_in_logs: bool = Field(default=True, description="Whether to mask credentials in logs")
    rotation_warning_days: int = Field(default=7, description="Days before expiry to warn")
    
    credentials_dir: Optional[str] = Field(None, description="Directory for encrypted credential files")
    
    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True


class CredentialManager:
    """
    Secure credential manager for API keys and sensitive authentication information.
    
    Features:
    - Multiple secure storage backends
    - Automatic credential masking in logs
    - Credential validation
    - Rotation tracking and warnings
    - Encryption for stored credentials
    
    Usage:
    ```python
    # Initialize manager
    cred_manager = CredentialManager()
    
    # Register credentials
    cred_manager.register_credential(CredentialConfig(
        name="binance_api_key",
        provider_id="crypto_ccxt",
        credential_type=CredentialType.API_KEY,
        source=CredentialSource.ENV,
        env_var="BINANCE_API_KEY",
        required=True
    ))
    
    # Get credentials (returns SecretStr)
    api_key = cred_manager.get_credential("binance_api_key")
    
    # Use in a secure way (only expose when needed)
    auth_header = f"Bearer {api_key.get_secret_value()}"
    ```
    """
    
    def __init__(self, config: Optional[CredentialManagerConfig] = None):
        """Initialize the credential manager"""
        self.config = config or CredentialManagerConfig()
        self._credentials: Dict[str, CredentialConfig] = {}
        self._cached_values: Dict[str, SecretStr] = {}
        self._load_attempted: Set[str] = set()
        
        # Setup directory for credential files if needed
        if self.config.credentials_dir:
            os.makedirs(self.config.credentials_dir, exist_ok=True)
        
        # Initialize backend connections as needed
        self._init_backends()
    
    def _init_backends(self):
        """Initialize connections to credential backends"""
        # Vault initialization if configured
        if self.config.vault_addr:
            try:
                import hvac
                vault_token = os.getenv(self.config.vault_token_env_var)
                if vault_token:
                    self._vault_client = hvac.Client(url=self.config.vault_addr, token=vault_token)
                    if not self._vault_client.is_authenticated():
                        logger.warning("Vault client failed to authenticate with provided token")
                        self._vault_client = None
                else:
                    logger.warning(f"Vault address configured but no token found in {self.config.vault_token_env_var}")
                    self._vault_client = None
            except ImportError:
                logger.warning("HashiCorp Vault support requested but hvac package not installed")
                self._vault_client = None
        else:
            self._vault_client = None
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate an encryption key for securing files"""
        key_env = os.getenv(self.config.encryption_key_env_var)
        
        if key_env:
            # Use provided key
            try:
                return base64.urlsafe_b64decode(key_env)
            except Exception as e:
                logger.error(f"Invalid encryption key format: {e}")
        
        # If no key found or invalid, try to get from keyring
        try:
            stored_key = keyring.get_password(self.config.keyring_app_name, "encryption_key")
            if stored_key:
                return base64.urlsafe_b64decode(stored_key)
        except Exception as e:
            logger.warning(f"Failed to retrieve encryption key from keyring: {e}")
        
        # If all else fails, generate a new key
        salt = b"quant_research_secure_salt"
        password = os.urandom(16)  # random password since we'll store it
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        # Store in keyring for future use
        try:
            keyring.set_password(self.config.keyring_app_name, "encryption_key", key.decode())
            logger.info("Generated and stored new encryption key in system keyring")
        except Exception as e:
            logger.warning(f"Failed to store encryption key in keyring: {e}")
        
        return base64.urlsafe_b64decode(key)
    
    def register_credential(self, credential_config: CredentialConfig) -> None:
        """
        Register a credential configuration
        
        Args:
            credential_config: Configuration for the credential
        """
        self._credentials[credential_config.name] = credential_config
        logger.debug(f"Registered credential '{credential_config.name}' for provider '{credential_config.provider_id}'")
    
    def register_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Register a credential from a dictionary
        
        Args:
            config_dict: Dictionary of credential configuration
        """
        credential = CredentialConfig.parse_obj(config_dict)
        self.register_credential(credential)
    
    def get_credential(self, name: str, refresh: bool = False) -> SecretStr:
        """
        Get a credential by name
        
        Args:
            name: Name of the credential
            refresh: Whether to force refresh from source
            
        Returns:
            SecretStr containing the credential value
            
        Raises:
            ValueError: If credential not registered
            RuntimeError: If credential loading fails
        """
        if name not in self._credentials:
            raise ValueError(f"Credential '{name}' not registered")
        
        # Return cached value if available and not refreshing
        if not refresh and name in self._cached_values:
            return self._cached_values[name]
        
        credential = self._credentials[name]
        
        # Check if credential is expired
        if credential.expires_at and credential.expires_at < datetime.now():
            logger.warning(f"Credential '{name}' has expired on {credential.expires_at}")
        
        # Check if credential needs rotation
        if credential.rotation_days and credential.last_rotated:
            rotation_date = credential.last_rotated + timedelta(days=credential.rotation_days)
            days_until_rotation = (rotation_date - datetime.now()).days
            
            if days_until_rotation <= 0:
                logger.warning(f"Credential '{name}' should be rotated immediately (overdue by {abs(days_until_rotation)} days)")
            elif days_until_rotation <= self.config.rotation_warning_days:
                logger.warning(f"Credential '{name}' should be rotated soon (in {days_until_rotation} days)")
        
        # Load credential based on source
        value = self._load_credential(credential)
        
        # Validate the credential
        if value is None or value.get_secret_value() == "":
            if credential.required:
                err_msg = f"Required credential '{name}' not found or empty"
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            else:
                logger.warning(f"Optional credential '{name}' not found")
                # Return empty string for optional credentials
                value = SecretStr("")
        elif credential.pattern:
            pattern = re.compile(credential.pattern)
            if not pattern.match(value.get_secret_value()):
                logger.warning(f"Credential '{name}' does not match required pattern")
        
        # Cache the result
        self._cached_values[name] = value
        
        return value
    
    def _load_credential(self, credential: CredentialConfig) -> Optional[SecretStr]:
        """
        Load a credential from its configured source
        
        Args:
            credential: Credential configuration
            
        Returns:
            SecretStr containing the credential value or None if not found
        """
        # Add to load attempted set for tracking
        self._load_attempted.add(credential.name)
        
        # Try to load from configured source
        try:
            if credential.source == CredentialSource.ENV:
                if not credential.env_var:
                    logger.error(f"Missing env_var for credential '{credential.name}'")
                    return None
                
                value = os.getenv(credential.env_var)
                return SecretStr(value) if value else None
            
            elif credential.source == CredentialSource.KEYRING:
                if not credential.keyring_service:
                    logger.error(f"Missing keyring_service for credential '{credential.name}'")
                    return None
                
                value = keyring.get_password(credential.keyring_service, credential.name)
                return SecretStr(value) if value else None
            
            elif credential.source == CredentialSource.VAULT:
                if not self._vault_client:
                    logger.error("Vault client not initialized but credential source is VAULT")
                    return None
                
                if not credential.vault_path:
                    logger.error(f"Missing vault_path for credential '{credential.name}'")
                    return None
                
                # Read from Vault
                secret = self._vault_client.secrets.kv.v2.read_secret_version(path=credential.vault_path)
                if secret and 'data' in secret and 'data' in secret['data']:
                    value = secret['data']['data'].get(credential.name)
                    return SecretStr(value) if value else None
                return None
            
            elif credential.source == CredentialSource.FILE:
                if not credential.file_path:
                    logger.error(f"Missing file_path for credential '{credential.name}'")
                    return None
                
                # Resolve path
                file_path = credential.file_path
                if self.config.credentials_dir:
                    file_path = os.path.join(self.config.credentials_dir, file_path)
                
                if not os.path.exists(file_path):
                    logger.warning(f"Credential file not found: {file_path}")
                    return None
                
                # Decrypt file
                key = self._get_encryption_key()
                fernet = Fernet(key)
                
                with open(file_path, 'rb') as f:
                    encrypted_data = f.read()
                
                try:
                    decrypted_data = fernet.decrypt(encrypted_data)
                    data = json.loads(decrypted_data.decode())
                    value = data.get(credential.name)
                    return SecretStr(value) if value else None
                except Exception as e:
                    logger.error(f"Failed to decrypt credential file: {e}")
                    return None
            
            elif credential.source == CredentialSource.CONFIG:
                # Direct value from config - least secure
                if credential.value:
                    logger.warning(f"Using direct credential value for '{credential.name}' - this is not secure for production")
                    return credential.value
                return None
            
        except Exception as e:
            logger.error(f"Error loading credential '{credential.name}': {e}")
            return None
    
    def store_credential(self, name: str, value: str, source: Optional[CredentialSource] = None) -> bool:
        """
        Store a credential securely
        
        Args:
            name: Name of the registered credential
            value: Value to store
            source: Override the source defined in registration
            
        Returns:
            bool indicating success
            
        Raises:
            ValueError: If credential not registered
        """
        if name not in self._credentials:
            raise ValueError(f"Credential '{name}' not registered")
        
        credential = self._credentials[name]
        source = source or credential.source
        
        try:
            if source == CredentialSource.ENV:
                logger.warning("Cannot programmatically set environment variables securely")
                return False
            
            elif source == CredentialSource.KEYRING:
                if not credential.keyring_service:
                    logger.error(f"Missing keyring_service for credential '{name}'")
                    return False
                
                keyring.set_password(credential.keyring_service, name, value)
                logger.info(f"Stored credential '{name}' in system keyring")
                return True
            
            elif source == CredentialSource.VAULT:
                if not self._vault_client:
                    logger.error("Vault client not initialized but credential source is VAULT")
                    return False
                
                if not credential.vault_path:
                    logger.error(f"Missing vault_path for credential '{name}'")
                    return False
                
                # Create or update secret
                self._vault_client.secrets.kv.v2.create_or_update_secret(
                    path=credential.vault_path,
                    secret={name: value}
                )
                logger.info(f"Stored credential '{name}' in Vault at {credential.vault_path}")
                return True
            
            elif source == CredentialSource.FILE:
                if not credential.file_path:
                    logger.error(f"Missing file_path for credential '{name}'")
                    return False
                
                # Resolve path
                file_path = credential.file_path
                if self.config.credentials_dir:
                    file_path = os.path.join(self.config.credentials_dir, file_path)
                
                # Load existing data if file exists
                data = {}
                if os.path.exists(file_path):
                    try:
                        key = self._get_encryption_key()
                        fernet = Fernet(key)
                        
                        with open(file_path, 'rb') as f:
                            encrypted_data = f.read()
                        
                        decrypted_data = fernet.decrypt(encrypted_data)
                        data = json.loads(decrypted_data.decode())
                    except Exception as e:
                        logger.warning(f"Failed to read existing credential file, creating new: {e}")
                
                # Update with new value
                data[name] = value
                
                # Encrypt and save
                key = self._get_encryption_key()
                fernet = Fernet(key)
                
                encrypted_data = fernet.encrypt(json.dumps(data).encode())
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                
                with open(file_path, 'wb') as f:
                    f.write(encrypted_data)
                
                # Set secure permissions
                os.chmod(file_path, 0o600)  # Only owner can read/write
                
                logger.info(f"Stored credential '{name}' in encrypted file")
                return True
            
            elif source == CredentialSource.CONFIG:
                # Update direct value
                credential.value = SecretStr(value)
                logger.warning(f"Stored credential '{name}' directly in config - this is not secure for production")
                return True
            
        except Exception as e:
            logger.error(f"Error storing credential '{name}': {e}")
            return False
    
    def rotate_credential(self, name: str, new_value: str) -> bool:
        """
        Rotate a credential to a new value and update last_rotated timestamp
        
        Args:
            name: Name of the credential
            new_value: New value for the credential
            
        Returns:
            bool indicating success
        """
        success = self.store_credential(name, new_value)
        if success:
            # Update rotation timestamp
            self._credentials[name].last_rotated = datetime.now()
            # Refresh cache
            self._cached_values[name] = SecretStr(new_value)
            logger.info(f"Rotated credential '{name}'")
        return success
    
    def get_credential_info(self, name: str) -> Dict[str, Any]:
        """
        Get metadata about a credential (without exposing the value)
        
        Args:
            name: Name of the credential
            
        Returns:
            Dict of credential metadata
            
        Raises:
            ValueError: If credential not registered
        """
        if name not in self._credentials:
            raise ValueError(f"Credential '{name}' not registered")
        
        credential = self._credentials[name]
        loaded = name in self._load_attempted
        
        return {
            "name": credential.name,
            "provider_id": credential.provider_id,
            "type": credential.credential_type,
            "source": credential.source,
            "required": credential.required,
            "loaded": loaded,
            "available": name in self._cached_values,
            "expires_at": credential.expires_at,
            "last_rotated": credential.last_rotated,
            "rotation_days": credential.rotation_days,
            "days_until_rotation": (
                (credential.last_rotated + timedelta(days=credential.rotation_days) - datetime.now()).days
                if credential.last_rotated and credential.rotation_days else None
            )
        }
    
    def list_credentials(self, provider_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered credentials with metadata
        
        Args:
            provider_id: Optional filter by provider
            
        Returns:
            List of credential metadata dictionaries
        """
        result = []
        
        for name, credential in self._credentials.items():
            if provider_id is None or credential.provider_id == provider_id:
                result.append(self.get_credential_info(name))
        
        return result
    
    def audit_credentials(self) -> Dict[str, Any]:
        """
        Perform an audit of all credentials
        
        Returns:
            Audit report with statistics and issues
        """
        total = len(self._credentials)
        loaded = len(self._load_attempted)
        available = len(self._cached_values)
        
        # Find issues
        missing_required = []
        expiring_soon = []
        rotation_needed = []
        invalid_pattern = []
        
        for name, credential in self._credentials.items():
            # Check required but missing
            if credential.required and (name not in self._cached_values or self._cached_values[name].get_secret_value() == ""):
                missing_required.append(name)
            
            # Check expiration
            if credential.expires_at:
                days_until_expiry = (credential.expires_at - datetime.now()).days
                if days_until_expiry <= self.config.rotation_warning_days:
                    expiring_soon.append((name, days_until_expiry))
            
            # Check rotation
            if credential.last_rotated and credential.rotation_days:
                rotation_date = credential.last_rotated + timedelta(days=credential.rotation_days)
                days_until_rotation = (rotation_date - datetime.now()).days
                if days_until_rotation <= self.config.rotation_warning_days:
                    rotation_needed.append((name, days_until_rotation))
            
            # Check pattern validation
            if credential.pattern and name in self._cached_values:
                pattern = re.compile(credential.pattern)
                if not pattern.match(self._cached_values[name].get_secret_value()):
                    invalid_pattern.append(name)
        
        return {
            "total_credentials": total,
            "loaded_credentials": loaded,
            "available_credentials": available,
            "issues": {
                "missing_required": missing_required,
                "expiring_soon": expiring_soon,
                "rotation_needed": rotation_needed,
                "invalid_pattern": invalid_pattern,
            },
            "issue_count": len(missing_required) + len(expiring_soon) + len(rotation_needed) + len(invalid_pattern)
        }
    
    def mask_value(self, value: str) -> str:
        """
        Mask a sensitive value for logging
        
        Args:
            value: Original sensitive value
            
        Returns:
            Masked value (e.g., "ABCDEF" -> "AB****")
        """
        if not value or not self.config.mask_credentials_in_logs:
            return value
        
        if len(value) <= 4:
            return "****"
        
        return value[:2] + "*" * (len(value) - 4) + value[-2:]
    
    def get_provider_credentials(self, provider_id: str) -> Dict[str, SecretStr]:
        """
        Get all credentials for a specific provider
        
        Args:
            provider_id: Provider identifier
            
        Returns:
            Dict mapping credential names to SecretStr values
        """
        result = {}
        
        for name, credential in self._credentials.items():
            if credential.provider_id == provider_id:
                try:
                    value = self.get_credential(name)
                    result[name] = value
                except Exception as e:
                    logger.warning(f"Failed to load credential '{name}' for provider '{provider_id}': {e}")
        
        return result
    
    def clear_cache(self) -> None:
        """Clear credential cache"""
        self._cached_values.clear()
        self._load_attempted.clear()
    
    def close(self) -> None:
        """Close any connections and cleanup resources"""
        self.clear_cache()
        
        # Close Vault client if exists
        if hasattr(self, '_vault_client') and self._vault_client:
            try:
                self._vault_client.close()
            except:
                pass


# Helper functions for secure logging
def redact_credentials(text: str, patterns: List[str] = None) -> str:
    """
    Redact credential values from log text
    
    Args:
        text: Original text
        patterns: List of regex patterns to match credentials
        
    Returns:
        Text with credentials redacted
    """
    if not text:
        return text
    
    # Default patterns for common credentials in logs
    default_patterns = [
        r'(api[_-]?key|apikey|key|token|secret|password|auth)["\']?\s*[:=]\s*["\']?([^"\',\s]+)["\']?',
        r'Authorization:\s*Bearer\s+([^\s]+)',
        r'Authorization:\s*Basic\s+([^\s]+)',
        r'([A-Za-z0-9+/]{40,}={0,2})',  # Base64 tokens
    ]
    
    patterns = patterns or default_patterns
    
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        # Replace capturing group with placeholder
        text = regex.sub(lambda m: m.group(0).replace(m.group(len(m.groups())), "****"), text)
    
    return text


# Create a logging filter that redacts credentials
class CredentialRedactionFilter(logging.Filter):
    """Logging filter to redact credentials from log records"""
    
    def __init__(self, patterns: Optional[List[str]] = None):
        """Initialize with optional custom patterns"""
        super().__init__()
        self.patterns = patterns
    
    def filter(self, record):
        """Filter log records to redact credentials"""
        if isinstance(record.msg, str):
            record.msg = redact_credentials(record.msg, self.patterns)
        
        # Also check args for strings
        if record.args:
            args = []
            for arg in record.args:
                if isinstance(arg, str):
                    args.append(redact_credentials(arg, self.patterns))
                else:
                    args.append(arg)
            record.args = tuple(args)
        
        return True


# Install credential redaction for the entire app
def install_credential_redaction(patterns: Optional[List[str]] = None):
    """
    Install credential redaction filter for all loggers
    
    Args:
        patterns: Optional list of regex patterns to match credentials
    """
    # Create and add the filter to the root logger
    credential_filter = CredentialRedactionFilter(patterns)
    root_logger = logging.getLogger()
    root_logger.addFilter(credential_filter)
    
    # Also add to the quant_research logger directly
    app_logger = logging.getLogger("quant_research")
    app_logger.addFilter(credential_filter)