# src/quant_research/providers/base.py
from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Dict, Any, AsyncIterator, Optional, List, TypeVar, Generic, Set

from ..core.config import ProviderConfig
from ..core.credentials import CredentialManager, CredentialConfig


logger = logging.getLogger(__name__)
T = TypeVar('T', bound=ProviderConfig)


class BaseProvider(Generic[T], ABC):
    """
    Abstract base class for all data providers.
    
    Each provider is responsible for:
    1. Connecting to a specific data source
    2. Retrieving data in a standardized format
    3. Handling errors and retries
    4. Properly managing resources and credentials
    
    Implementations should be thread-safe and handle rate limits appropriately.
    """
    
    def __init__(self, config: T, credential_manager: Optional[CredentialManager] = None):
        """
        Initialize the provider with configuration and optional credential manager.
        
        Args:
            config: Provider-specific configuration
            credential_manager: Optional credential manager for secure API key handling
        """
        self.config = config
        self._credential_manager = credential_manager
        self._required_credentials: Set[str] = set()
        
        # Register provider-specific credentials on init
        self._register_credentials()
    
    def _register_credentials(self) -> None:
        """
        Register required credentials for this provider.
        
        Override this in provider implementations to register 
        provider-specific credentials.
        """
        pass
    
    def register_credential(self, credential_config: CredentialConfig) -> None:
        """
        Register a credential for this provider.
        
        Args:
            credential_config: Credential configuration
        """
        if self._credential_manager:
            self._credential_manager.register_credential(credential_config)
            if credential_config.required:
                self._required_credentials.add(credential_config.name)
        else:
            logger.warning(
                f"Credential '{credential_config.name}' not registered: no credential manager available"
            )
    
    def get_credential(self, name: str) -> Optional[str]:
        """
        Get a credential by name.
        
        Args:
            name: Name of the credential
            
        Returns:
            Credential value or None if not available
            
        Note: This method should only be called when absolutely necessary to 
        avoid exposing credentials in memory.
        """
        if not self._credential_manager:
            logger.warning(f"Cannot retrieve credential '{name}': no credential manager available")
            return None
        
        try:
            secret = self._credential_manager.get_credential(name)
            return secret.get_secret_value() if secret else None
        except Exception as e:
            logger.error(f"Error retrieving credential '{name}': {e}")
            return None
    
    def check_credentials(self) -> bool:
        """
        Check if all required credentials are available.
        
        Returns:
            True if all required credentials are available, False otherwise
        """
        if not self._credential_manager:
            return len(self._required_credentials) == 0
        
        for name in self._required_credentials:
            try:
                secret = self._credential_manager.get_credential(name)
                if not secret or secret.get_secret_value() == "":
                    logger.error(f"Required credential '{name}' not available")
                    return False
            except Exception as e:
                logger.error(f"Error checking credential '{name}': {e}")
                return False
        
        return True
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the data source.
        
        This method should:
        - Initialize any required clients or connections
        - Authenticate if necessary
        - Verify connectivity
        - Raise exceptions if connection fails
        """
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if the provider is currently connected.
        
        Returns:
            bool: True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection and release resources.
        
        This method should ensure all resources are properly released
        to prevent memory leaks or hanging connections.
        """
        pass
    
    @abstractmethod
    async def fetch_data(self, **params) -> AsyncIterator[Dict[str, Any]]:
        """
        Fetch data from the provider as an asynchronous stream.
        
        Parameters can vary by provider, but common ones include:
        - symbols: List of symbols to fetch
        - start_time/end_time: Time range to fetch
        - limit: Maximum number of records
        
        Yields:
            Dict[str, Any]: Each data point in a standardized format
        """
        pass
    
    @abstractmethod
    async def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the provider and available data.
        
        Returns:
            Dict[str, Any]: Provider metadata
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the provider and connection.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Check if credentials are available
            credentials_ok = self.check_credentials()
            
            # Check connection status
            connected = await self.is_connected()
            if not connected and credentials_ok:
                await self.connect()
                connected = await self.is_connected()
            
            metadata = await self.get_metadata() if connected else {}
            
            return {
                "status": "healthy" if connected and credentials_ok else "unhealthy",
                "connected": connected,
                "credentials_ok": credentials_ok,
                "provider_type": self.__class__.__name__,
                "metadata": metadata
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "credentials_ok": False,
                "provider_type": self.__class__.__name__,
                "error": str(e)
            }
    
    async def __aenter__(self):
        """Context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.disconnect()