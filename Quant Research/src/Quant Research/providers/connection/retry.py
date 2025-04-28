# src/quant_research/providers/connection/retry.py
"""Retry mechanism with exponential backoff for data providers."""

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Awaitable, Callable, Generic, Optional, Type, TypeVar, Union

from ...core.errors import RetryableError

logger = logging.getLogger(__name__)
T = TypeVar('T')  # Type of connection object
R = TypeVar('R')  # Return type of function


class RetryHandler(Generic[T]):
    """
    Handles retry logic with configurable backoff strategies.
    
    Features:
    - Exponential backoff with jitter
    - Configurable retry attempts and delay
    - Customizable retry conditions
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.1,
        retryable_exceptions: Optional[tuple] = None
    ):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            jitter_factor: Random jitter factor (0-1) to add to delay
            retryable_exceptions: Tuple of exception types to retry
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor
        self.retryable_exceptions = retryable_exceptions or (Exception,)
        self._retry_count = 0
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay with exponential backoff and jitter.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        
        # Add jitter to prevent thundering herd
        jitter = delay * self.jitter_factor * random.random()
        
        return delay + jitter
    
    def _should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception to check
            
        Returns:
            True if should retry, False otherwise
        """
        # Always retry for RetryableError
        if isinstance(exception, RetryableError):
            return True
        
        # Check against retryable exceptions
        return isinstance(exception, self.retryable_exceptions)
    
    async def execute(
        self,
        func: Callable[[T], Awaitable[R]],
        conn: T,
        *,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None
    ) -> R:
        """
        Execute a function with retry logic.
        
        Args:
            func: Async function taking a connection
            conn: Connection object to pass to function
            max_retries: Override default max retries
            base_delay: Override default base delay
            
        Returns:
            Result of the function call
            
        Raises:
            RetryableError: If max retries exceeded
            Exception: Other exceptions not eligible for retry
        """
        max_attempts = max_retries if max_retries is not None else self.max_retries
        retry_delay = base_delay if base_delay is not None else self.base_delay
        
        last_error = None
        for attempt in range(max_attempts + 1):
            try:
                return await func(conn)
            except Exception as e:
                last_error = e
                
                # Check if we should retry
                if attempt < max_attempts and self._should_retry(e):
                    self._retry_count += 1
                    
                    # Calculate delay with backoff and jitter
                    delay = self._calculate_delay(attempt)
                    
                    logger.warning(
                        f"Attempt {attempt+1}/{max_attempts+1} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    # Don't retry non-retryable exceptions
                    if not self._should_retry(e):
                        raise
                    break
        
        # If we get here, we've exhausted retries
        raise RetryableError(
            f"Operation failed after {max_attempts+1} attempts",
            original_error=last_error
        )
    
    def get_retry_count(self) -> int:
        """
        Get the total number of retries performed.
        
        Returns:
            Number of retries
        """
        return self._retry_count
    
    def reset_stats(self) -> None:
        """Reset retry statistics."""
        self._retry_count = 0


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.1,
    retryable_exceptions: Optional[tuple] = None
):
    """
    Decorator for adding retry logic to async functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        jitter_factor: Random jitter factor (0-1) to add to delay
        retryable_exceptions: Tuple of exception types to retry
        
    Returns:
        Decorated function with retry capability
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = RetryHandler(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter_factor=jitter_factor,
                retryable_exceptions=retryable_exceptions
            )
            
            # Define a simple wrapper function to adapt to the RetryHandler.execute interface
            async def wrapped_func(conn=None):
                return await func(*args, **kwargs)
            
            return await handler.execute(wrapped_func, None)
        
        return wrapper
    
    return decorator