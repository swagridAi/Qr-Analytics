# src/quant_research/providers/connection/__init__.py
"""
Connection management for data providers.

This package provides utilities for managing connections to external services,
including connection pooling, retry logic, and health checking.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict, Generic, Optional, TypeVar

from ...core.config import ConnectionConfig
from ...core.errors import ConnectionError, RetryableError, TimeoutError
from .health import HealthChecker, HealthCheckResult, default_db_health_check, default_http_health_check
from .pool import ConnectionPool
from .retry import RetryHandler, with_retry
from .types import Cleanup, ConnectionFactory, ConnectionStats, HealthCheck

logger = logging.getLogger(__name__)
T = TypeVar('T')  # Type of connection object


class ConnectionManager(Generic[T]):
    """
    Manages connections to external services with pooling and retry logic.
    
    This class is maintained for backward compatibility.
    New code should use the more focused ConnectionPool and RetryHandler classes.
    
    Features:
    - Connection pooling
    - Automatic reconnection
    - Exponential backoff for retries
    - Connection health monitoring
    - Resource cleanup
    """
    
    def __init__(
        self,
        connection_factory: ConnectionFactory[T],
        config: ConnectionConfig,
        health_check: Optional[HealthCheck[T]] = None,
        cleanup: Optional[Cleanup[T]] = None
    ):
        """
        Initialize connection manager.
        
        Args:
            connection_factory: Async factory function to create a new connection
            config: Connection configuration
            health_check: Optional function to check connection health
            cleanup: Optional function to clean up a connection
        """
        self.connection_factory = connection_factory
        self.config = config
        
        # Set default health check and cleanup if not provided
        self.health_check = health_check or (lambda _: asyncio.sleep(0) or True)
        self.cleanup = cleanup or (lambda _: asyncio.sleep(0))
        
        # Initialize pool and retry handler
        self._pool = ConnectionPool(connection_factory, config, health_check, cleanup)
        self._retry_handler = RetryHandler(
            max_retries=config.max_retries,
            base_delay=config.retry_delay
        )
    
    async def initialize(self) -> None:
        """Initialize the connection pool with connections."""
        await self._pool.initialize()
    
    async def acquire(self) -> T:
        """
        Acquire a connection from the pool.
        
        Returns:
            A connection object
            
        Raises:
            ConnectionError: If unable to acquire a connection
        """
        # Wrap the async context manager for backward compatibility
        connection_ctx = self._pool.acquire()
        # Create a task to acquire the connection
        return await asyncio.create_task(self._acquire_connection(connection_ctx))
    
    async def _acquire_connection(self, connection_ctx):
        """Helper method to acquire a connection from a context manager."""
        async with connection_ctx as conn:
            return conn
    
    async def retry(
        self,
        func: Callable[[T], Awaitable[Any]],
        *,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None
    ) -> Any:
        """
        Execute a function with automatic retries.
        
        Args:
            func: Async function taking a connection
            max_retries: Max retry attempts (defaults to config)
            retry_delay: Base delay between retries (defaults to config)
            
        Returns:
            Result of the function call
            
        Raises:
            RetryableError: If max retries exceeded
        """
        # For backward compatibility, wrap the function to acquire a connection first
        async def wrapped_func(dummy=None):
            async with self._pool.acquire() as conn:
                return await func(conn)
        
        # Use the retry handler
        return await self._retry_handler.execute(
            wrapped_func,
            None,
            max_retries=max_retries,
            base_delay=retry_delay
        )
    
    async def close(self) -> None:
        """Close all connections and clean up resources."""
        await self._pool.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        stats = self._pool.get_stats()
        stats["retry_count"] = self._retry_handler.get_retry_count()
        return stats


# Export all relevant classes and functions
__all__ = [
    'ConnectionManager',
    'ConnectionPool',
    'RetryHandler',
    'HealthChecker',
    'HealthCheckResult',
    'with_retry',
    'default_http_health_check',
    'default_db_health_check',
]