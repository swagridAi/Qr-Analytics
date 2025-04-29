"""
Task helpers for pipeline orchestration.

This module provides utility functions for creating standardized task types
that integrate with the pipeline framework.
"""

from typing import Any, Callable, Dict, Optional, TypeVar
from functools import wraps
import logging

from prefect import task as prefect_task

T = TypeVar('T')

def create_provider_task(name: Optional[str] = None, description: Optional[str] = None):
    """Create a standardized provider task."""
    def decorator(func: Callable):
        @prefect_task(name=name or func.__name__, description=description or func.__doc__)
        @wraps(func)
        async def wrapper(**kwargs):
            # Provider-specific standardized behavior
            return await func(**kwargs)
        return wrapper
    return decorator

def create_analytics_task(name: Optional[str] = None, description: Optional[str] = None):
    """Create a standardized analytics task."""
    def decorator(func: Callable):
        @prefect_task(name=name or func.__name__, description=description or func.__doc__)
        @wraps(func)
        def wrapper(**kwargs):
            # Analytics-specific standardized behavior
            return func(**kwargs)
        return wrapper
    return decorator

def create_backtest_task(name: Optional[str] = None, description: Optional[str] = None):
    """Create a standardized backtest task."""
    def decorator(func: Callable):
        @prefect_task(name=name or func.__name__, description=description or func.__doc__)
        @wraps(func)
        def wrapper(**kwargs):
            # Backtest-specific standardized behavior
            return func(**kwargs)
        return wrapper
    return decorator

# Direct task decorator for general-purpose tasks
def pipeline_task(name: Optional[str] = None, description: Optional[str] = None, 
                 retries: int = 0, retry_delay_seconds: int = 30):
    """Create a standard pipeline task with consistent logging and error handling."""
    def decorator(func: Callable):
        @prefect_task(
            name=name or func.__name__,
            description=description or func.__doc__,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(f"quant_research.pipelines.tasks.{func.__name__}")
            logger.info(f"Starting task: {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                logger.info(f"Task completed: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Task error in {func.__name__}: {str(e)}", exc_info=True)
                raise
                
        return wrapper
    return decorator