# src/quant_research/providers/connection/types.py
"""Type definitions and interfaces for connection management."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Generic, Optional, Protocol, TypeVar

T = TypeVar('T')  # Type of connection object


class HealthCheck(Protocol, Generic[T]):
    """Protocol for health check functions."""
    
    async def __call__(self, conn: T) -> bool:
        """
        Check if a connection is healthy.
        
        Args:
            conn: The connection to check
            
        Returns:
            True if connection is healthy, False otherwise
        """
        ...


class Cleanup(Protocol, Generic[T]):
    """Protocol for cleanup functions."""
    
    async def __call__(self, conn: T) -> None:
        """
        Clean up resources associated with a connection.
        
        Args:
            conn: The connection to clean up
        """
        ...


class ConnectionFactory(Protocol, Generic[T]):
    """Protocol for connection factory functions."""
    
    async def __call__(self) -> T:
        """
        Create a new connection.
        
        Returns:
            A new connection object
        """
        ...


@dataclass
class ConnectionStats:
    """Statistics for connection management."""
    
    # Counts
    created_count: int = 0
    failed_count: int = 0
    retry_count: int = 0
    
    # Pool state
    pool_size: int = 0
    in_use: int = 0
    
    # Timing information
    last_created: Optional[datetime] = None
    last_failed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "pool_size": self.pool_size,
            "in_use": self.in_use,
            "created_count": self.created_count,
            "failed_count": self.failed_count,
            "retry_count": self.retry_count,
            "last_created": self.last_created.isoformat() if self.last_created else None,
            "last_failed": self.last_failed.isoformat() if self.last_failed else None,
        }