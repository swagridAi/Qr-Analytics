# src/quant_research/providers/connection/pool.py
"""Connection pool management for data providers."""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Generic, Optional, Set, TypeVar

from ...core.config import ConnectionConfig
from ...core.errors import ConnectionError, TimeoutError
from .types import Cleanup, ConnectionFactory, ConnectionStats, HealthCheck

logger = logging.getLogger(__name__)
T = TypeVar('T')  # Type of connection object


class ConnectionPool(Generic[T]):
    """
    Manages a pool of connections to external services.
    
    Features:
    - Connection pooling with configurable size
    - Automatic health checks
    - Connection creation and cleanup
    - Resource management
    """
    
    def __init__(
        self,
        connection_factory: ConnectionFactory[T],
        config: ConnectionConfig,
        health_check: Optional[HealthCheck[T]] = None,
        cleanup: Optional[Cleanup[T]] = None
    ):
        """
        Initialize connection pool.
        
        Args:
            connection_factory: Async factory function to create a new connection
            config: Connection configuration
            health_check: Optional function to check connection health
            cleanup: Optional function to clean up a connection
        """
        self.connection_factory = connection_factory
        self.config = config
        self.health_check = health_check
        self.cleanup = cleanup
        
        # Connection pool
        self._pool: Set[T] = set()
        self._in_use: Set[T] = set()
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = ConnectionStats()
    
    async def initialize(self) -> None:
        """Initialize the connection pool with connections."""
        async with self._lock:
            for _ in range(self.config.pool_size):
                try:
                    conn = await self._create_connection()
                    self._pool.add(conn)
                    self._stats.pool_size = len(self._pool)
                except Exception as e:
                    logger.error(f"Failed to initialize connection: {e}")
    
    async def _create_connection(self) -> T:
        """
        Create a new connection with timeout.
        
        Returns:
            A new connection object
            
        Raises:
            TimeoutError: If connection creation times out
            ConnectionError: If connection creation fails
        """
        try:
            conn = await asyncio.wait_for(
                self.connection_factory(),
                timeout=self.config.timeout
            )
            self._stats.created_count += 1
            self._stats.last_created = datetime.now()
            return conn
        except asyncio.TimeoutError:
            self._stats.failed_count += 1
            self._stats.last_failed = datetime.now()
            raise TimeoutError(f"Connection timed out after {self.config.timeout}s")
        except Exception as e:
            self._stats.failed_count += 1
            self._stats.last_failed = datetime.now()
            raise ConnectionError(f"Failed to create connection: {e}")
    
    async def _check_connection(self, conn: T) -> bool:
        """
        Check if a connection is healthy.
        
        Args:
            conn: The connection to check
            
        Returns:
            True if the connection is healthy, False otherwise
        """
        if not self.health_check:
            return True
        
        try:
            return await asyncio.wait_for(
                self.health_check(conn),
                timeout=self.config.timeout
            )
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    @asynccontextmanager
    async def acquire(self) -> T:
        """
        Acquire a connection from the pool.
        
        Returns:
            A connection object
            
        Raises:
            ConnectionError: If unable to acquire a connection
        """
        conn = None
        
        # Try to get a connection from the pool
        async with self._lock:
            if self._pool:
                conn = self._pool.pop()
                self._in_use.add(conn)
                self._stats.pool_size = len(self._pool)
                self._stats.in_use = len(self._in_use)
        
        # If no connection in pool, create a new one
        if conn is None:
            conn = await self._create_connection()
            async with self._lock:
                self._in_use.add(conn)
                self._stats.in_use = len(self._in_use)
        
        # Check connection health
        is_healthy = await self._check_connection(conn)
        if not is_healthy:
            # Close unhealthy connection and try again
            await self._cleanup_connection(conn)
            async with self._lock:
                self._in_use.remove(conn)
                self._stats.in_use = len(self._in_use)
            
            # Create a new connection
            conn = await self._create_connection()
            async with self._lock:
                self._in_use.add(conn)
                self._stats.in_use = len(self._in_use)
        
        try:
            yield conn
        finally:
            # Return connection to pool if keep_alive
            async with self._lock:
                if conn in self._in_use:  # Check to prevent errors if already removed
                    self._in_use.remove(conn)
                    if self.config.keep_alive and len(self._pool) < self.config.pool_size:
                        self._pool.add(conn)
                    else:
                        # Close connection if not keeping alive
                        await self._cleanup_connection(conn)
                    self._stats.pool_size = len(self._pool)
                    self._stats.in_use = len(self._in_use)
    
    async def _cleanup_connection(self, conn: T) -> None:
        """
        Clean up resources associated with a connection.
        
        Args:
            conn: The connection to clean up
        """
        if self.cleanup:
            try:
                await self.cleanup(conn)
            except Exception as e:
                logger.warning(f"Error during connection cleanup: {e}")
    
    async def close(self) -> None:
        """Close all connections and clean up resources."""
        async with self._lock:
            # Close all pool connections
            for conn in self._pool:
                await self._cleanup_connection(conn)
            self._pool.clear()
            
            # Close all in-use connections
            for conn in self._in_use:
                await self._cleanup_connection(conn)
            self._in_use.clear()
            
            # Update stats
            self._stats.pool_size = 0
            self._stats.in_use = 0
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get connection pool statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self._stats.to_dict()