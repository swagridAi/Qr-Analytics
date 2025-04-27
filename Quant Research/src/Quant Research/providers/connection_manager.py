# src/quant_research/providers/connection_manager.py
import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Awaitable, TypeVar, Generic, Set
from contextlib import asynccontextmanager

from ..core.config import ConnectionConfig
from ..core.errors import ConnectionError, RetryableError, TimeoutError


logger = logging.getLogger(__name__)
T = TypeVar('T')


class ConnectionManager(Generic[T]):
    """
    Manages connections to external services with pooling and retry logic.
    
    Features:
    - Connection pooling
    - Automatic reconnection
    - Exponential backoff for retries
    - Connection health monitoring
    - Resource cleanup
    """
    
    def __init__(
        self,
        connection_factory: Callable[[], Awaitable[T]],
        config: ConnectionConfig,
        health_check: Optional[Callable[[T], Awaitable[bool]]] = None,
        cleanup: Optional[Callable[[T], Awaitable[None]]] = None
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
        self.health_check = health_check or (lambda _: asyncio.sleep(0) or True)
        self.cleanup = cleanup or (lambda _: asyncio.sleep(0))
        
        # Connection pool
        self._pool: Set[T] = set()
        self._in_use: Set[T] = set()
        self._lock = asyncio.Lock()
        
        # Statistics
        self._created_count = 0
        self._failed_count = 0
        self._retry_count = 0
    
    async def initialize(self) -> None:
        """Initialize the connection pool with connections"""
        async with self._lock:
            for _ in range(self.config.pool_size):
                try:
                    conn = await self._create_connection()
                    self._pool.add(conn)
                except Exception as e:
                    logger.error(f"Failed to initialize connection: {e}")
    
    async def _create_connection(self) -> T:
        """Create a new connection with timeout"""
        try:
            conn = await asyncio.wait_for(
                self.connection_factory(),
                timeout=self.config.timeout
            )
            self._created_count += 1
            return conn
        except asyncio.TimeoutError:
            self._failed_count += 1
            raise TimeoutError(f"Connection timed out after {self.config.timeout}s")
        except Exception as e:
            self._failed_count += 1
            raise ConnectionError(f"Failed to create connection: {e}")
    
    async def _check_connection(self, conn: T) -> bool:
        """Check if a connection is healthy"""
        try:
            return await asyncio.wait_for(
                self.health_check(conn),
                timeout=self.config.timeout
            )
        except Exception:
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
        
        # If no connection in pool, create a new one
        if conn is None:
            conn = await self._create_connection()
            async with self._lock:
                self._in_use.add(conn)
        
        # Check connection health
        is_healthy = await self._check_connection(conn)
        if not is_healthy:
            # Close unhealthy connection and try again
            await self.cleanup(conn)
            async with self._lock:
                self._in_use.remove(conn)
            
            # Create a new connection
            conn = await self._create_connection()
            async with self._lock:
                self._in_use.add(conn)
        
        try:
            yield conn
        finally:
            # Return connection to pool if keep_alive
            async with self._lock:
                self._in_use.remove(conn)
                if self.config.keep_alive and len(self._pool) < self.config.pool_size:
                    self._pool.add(conn)
                else:
                    # Close connection if not keeping alive
                    await self.cleanup(conn)
    
    async def retry(self, func: Callable[[T], Awaitable[Any]], *,
                   max_retries: Optional[int] = None,
                   retry_delay: Optional[int] = None) -> Any:
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
        max_retries = max_retries if max_retries is not None else self.config.max_retries
        retry_delay = retry_delay if retry_delay is not None else self.config.retry_delay
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                async with self.acquire() as conn:
                    return await func(conn)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    self._retry_count += 1
                    # Calculate backoff with jitter
                    delay = retry_delay * (2 ** attempt)
                    jitter = delay * 0.1 * (time.time() % 1)  # 10% jitter
                    total_delay = delay + jitter
                    
                    logger.warning(
                        f"Attempt {attempt+1}/{max_retries+1} failed: {e}. "
                        f"Retrying in {total_delay:.2f}s"
                    )
                    await asyncio.sleep(total_delay)
        
        raise RetryableError(
            f"Operation failed after {max_retries+1} attempts",
            original_error=last_error
        )
    
    async def close(self) -> None:
        """Close all connections and clean up resources"""
        async with self._lock:
            # Close all pool connections
            for conn in self._pool:
                await self.cleanup(conn)
            self._pool.clear()
            
            # Close all in-use connections
            for conn in self._in_use:
                await self.cleanup(conn)
            self._in_use.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "pool_size": len(self._pool),
            "in_use": len(self._in_use),
            "created_count": self._created_count,
            "failed_count": self._failed_count,
            "retry_count": self._retry_count,
        }