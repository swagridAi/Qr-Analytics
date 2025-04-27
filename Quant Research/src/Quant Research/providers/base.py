# src/quant_research/providers/base.py
from abc import ABC, abstractmethod
import asyncio
from typing import Dict, Any, AsyncIterator, Optional, List, TypeVar, Generic

from ..core.config import ProviderConfig


T = TypeVar('T', bound=ProviderConfig)


class BaseProvider(Generic[T], ABC):
    """
    Abstract base class for all data providers.
    
    Each provider is responsible for:
    1. Connecting to a specific data source
    2. Retrieving data in a standardized format
    3. Handling errors and retries
    4. Properly managing resources
    
    Implementations should be thread-safe and handle rate limits appropriately.
    """
    
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
            connected = await self.is_connected()
            if not connected:
                await self.connect()
                connected = await self.is_connected()
            
            metadata = await self.get_metadata()
            
            return {
                "status": "healthy" if connected else "unhealthy",
                "connected": connected,
                "provider_type": self.__class__.__name__,
                "metadata": metadata
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
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