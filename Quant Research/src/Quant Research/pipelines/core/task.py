"""
Task framework for pipeline orchestration.

This module provides a unified approach to creating, monitoring, and executing
tasks within the pipeline framework, with a clear separation of concerns.
"""

import functools
import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

# Type definitions
T = TypeVar('T')
TaskFunc = TypeVar('TaskFunc', bound=Callable)

# Logger for this module
logger = logging.getLogger(__name__)


class TaskContext:
    """
    Context object for task execution.
    
    Contains metadata about the task and its execution environment.
    """
    
    def __init__(self, 
                 task_name: str,
                 task_type: str = "general",
                 monitor: Optional[Any] = None):
        self.task_name = task_name
        self.task_type = task_type
        self.monitor = monitor
        self.start_time = None
        self.end_time = None
        self.metrics = {}
    
    def start(self) -> None:
        """Mark the start of task execution."""
        self.start_time = time.time()
        if self.monitor:
            self.monitor.start_task(self.task_name)
    
    def complete(self, success: bool = True, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Mark the completion of task execution."""
        self.end_time = time.time()
        
        # Calculate duration
        if self.start_time:
            duration = self.end_time - self.start_time
            self.metrics["duration_seconds"] = duration
        
        # Add additional metrics
        if metrics:
            self.metrics.update(metrics)
        
        # Report to monitor if available
        if self.monitor:
            self.monitor.complete_task(self.task_name, success, self.metrics)
    
    def record_error(self, error: Exception) -> None:
        """Record an error that occurred during task execution."""
        if self.monitor:
            self.monitor.record_error(error, self.task_name)


def task(name: Optional[str] = None,
         description: Optional[str] = None,
         task_type: str = "general",
         tags: Optional[List[str]] = None,
         **task_options) -> Callable[[TaskFunc], TaskFunc]:
    """
    Core task decorator that works with both sync and async functions.
    
    Args:
        name: Task name (defaults to function name)
        description: Task description
        task_type: Type of task (general, provider, analytics, etc.)
        tags: List of tags for categorization
        **task_options: Additional options for task execution
        
    Returns:
        Decorated task function
    """
    def decorator(func: TaskFunc) -> TaskFunc:
        # Get function metadata
        task_name = name or func.__name__
        task_description = description or func.__doc__
        is_async = inspect.iscoroutinefunction(func)
        
        # Store task metadata
        setattr(func, "__task_name__", task_name)
        setattr(func, "__task_description__", task_description)
        setattr(func, "__task_type__", task_type)
        setattr(func, "__task_tags__", tags or [])
        setattr(func, "__task_options__", task_options)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract monitor from kwargs if present
            monitor = kwargs.pop("monitor", None)
            
            # Create task context
            context = TaskContext(task_name, task_type, monitor)
            context.start()
            
            try:
                # Execute task
                result = await func(*args, **kwargs)
                
                # Extract metrics from result if available
                metrics = extract_metrics(result, task_type)
                context.complete(True, metrics)
                
                return result
                
            except Exception as e:
                # Record error
                context.record_error(e)
                context.complete(False)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Extract monitor from kwargs if present
            monitor = kwargs.pop("monitor", None)
            
            # Create task context
            context = TaskContext(task_name, task_type, monitor)
            context.start()
            
            try:
                # Execute task
                result = func(*args, **kwargs)
                
                # Extract metrics from result if available
                metrics = extract_metrics(result, task_type)
                context.complete(True, metrics)
                
                return result
                
            except Exception as e:
                # Record error
                context.record_error(e)
                context.complete(False)
                raise
        
        # Return appropriate wrapper based on whether function is async
        wrapper = async_wrapper if is_async else sync_wrapper
        
        # Transfer task metadata to wrapper
        for attr in ["__task_name__", "__task_description__", "__task_type__", 
                     "__task_tags__", "__task_options__"]:
            setattr(wrapper, attr, getattr(func, attr))
        
        return cast(TaskFunc, wrapper)
    
    return decorator


def extract_metrics(result: Any, task_type: str) -> Dict[str, Any]:
    """
    Extract metrics from task result based on result type and task type.
    
    Args:
        result: Task result
        task_type: Type of task
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Extract metrics from DataFrame-like objects
    if hasattr(result, "shape"):
        try:
            metrics["rows"] = result.shape[0]
            if len(result.shape) > 1:
                metrics["columns"] = result.shape[1]
        except (AttributeError, IndexError):
            pass
        
        # Add memory usage if available
        if hasattr(result, "memory_usage"):
            try:
                metrics["memory_bytes"] = result.memory_usage(deep=True).sum()
            except:
                pass
    
    # Extract metrics from dictionary results
    elif isinstance(result, dict):
        # Include metrics keys if they exist
        for key in ["count", "rows", "records", "items"]:
            if key in result:
                metrics[key] = result[key]
                
        # Extract specific metrics based on task type
        if task_type == "backtest" and "sharpe_ratio" in result:
            metrics["sharpe_ratio"] = result["sharpe_ratio"]
            metrics["total_return"] = result.get("total_return", 0)
    
    return metrics


# Convenience functions for common task types
def provider_task(**kwargs):
    """Decorator for data provider tasks."""
    return task(task_type="provider", **kwargs)


def analytics_task(**kwargs):
    """Decorator for analytics and signal generation tasks."""
    return task(task_type="analytics", **kwargs)


def backtest_task(**kwargs):
    """Decorator for backtesting tasks."""
    return task(task_type="backtest", **kwargs)


def dashboard_task(**kwargs):
    """Decorator for dashboard and visualization tasks."""
    return task(task_type="dashboard", **kwargs)


# Optional Prefect integration - separated from core functionality
try:
    from prefect import task as prefect_task
    from prefect.context import get_run_context
    
    def register_with_prefect(func: Callable) -> Callable:
        """
        Register a task with Prefect.
        
        This function wraps a task with Prefect's task decorator,
        preserving all task metadata.
        """
        task_name = getattr(func, "__task_name__", func.__name__)
        task_description = getattr(func, "__task_description__", func.__doc__)
        task_tags = getattr(func, "__task_tags__", [])
        task_options = getattr(func, "__task_options__", {})
        
        # Extract Prefect-specific options
        prefect_options = {
            "name": task_name,
            "description": task_description,
            "tags": task_tags,
        }
        
        # Add other Prefect-specific options if present
        for option in ["retries", "retry_delay_seconds", "timeout_seconds"]:
            if option in task_options:
                prefect_options[option] = task_options[option]
        
        # Apply Prefect task decorator
        return prefect_task(**prefect_options)(func)
    
except ImportError:
    # Prefect is not available, provide dummy implementation
    def register_with_prefect(func: Callable) -> Callable:
        """Dummy implementation when Prefect is not available."""
        return func