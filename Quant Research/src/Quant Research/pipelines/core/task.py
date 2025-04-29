"""
Task framework for pipeline orchestration.

This module provides a unified approach to creating, monitoring, and executing
tasks within the pipeline framework.
"""

import asyncio
import functools
import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

from prefect import task as prefect_task

from quant_research.pipelines.core.monitoring import PipelineMonitor

T = TypeVar('T')

def pipeline_task(
    name: Optional[str] = None,
    description: Optional[str] = None,
    task_type: str = "general",
    retries: int = 0,
    retry_delay_seconds: int = 30,
    tags: Optional[List[str]] = None,
    timeout_seconds: Optional[int] = None,
    cache_key_fn: Optional[Callable] = None,
    cache_expiration: Optional[float] = None,
    skip_monitoring: bool = False
):
    """
    Unified task decorator for all pipeline tasks.
    
    Creates a standardized task with consistent monitoring, metrics collection,
    and error handling. Integrates with the PipelineMonitor system.
    
    Args:
        name: Task name (defaults to function name)
        description: Task description (defaults to function docstring)
        task_type: Type of task for metrics collection (provider, analytics, backtest, etc.)
        retries: Number of retries if task fails
        retry_delay_seconds: Seconds between retries
        tags: Tags for task categorization
        timeout_seconds: Task timeout in seconds
        cache_key_fn: Function to generate cache key
        cache_expiration: Cache expiration in seconds
        skip_monitoring: Whether to skip monitoring (for testing/special cases)
        
    Returns:
        Decorated task function
    """
    # Ensure we have tags list
    if tags is None:
        tags = []
    
    # Add task_type to tags
    if task_type and task_type not in tags:
        tags.append(task_type)
        
    def decorator(func: Callable) -> Callable:
        # Get function metadata
        task_name = name or func.__name__
        task_description = description or func.__doc__
        
        # Determine if function is async
        is_async = inspect.iscoroutinefunction(func)
        
        # Create Prefect task
        prefect_decorated = prefect_task(
            name=task_name,
            description=task_description,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            tags=tags,
            timeout_seconds=timeout_seconds,
            cache_key_fn=cache_key_fn,
            cache_expiration=cache_expiration
        )(func)
        
        @functools.wraps(prefect_decorated)
        async def async_wrapper(*args, **kwargs):
            # Initialize monitoring
            monitor = None
            if not skip_monitoring:
                monitor = kwargs.pop('monitor', None)
                # Try to get monitor from context if not provided
                if monitor is None:
                    try:
                        from prefect.context import get_run_context
                        ctx = get_run_context()
                        flow_name = getattr(ctx, 'flow_name', "unknown_flow")
                        flow_run_id = getattr(ctx, 'flow_run_id', None)
                        
                        if flow_name and flow_run_id:
                            from quant_research.pipelines.core.monitoring import PipelineMonitor
                            monitor = PipelineMonitor(flow_name, flow_run_id)
                    except Exception:
                        # Continue without monitor if can't create one
                        pass
            
            # Start task monitoring if monitor exists
            if monitor:
                monitor.start_task(task_name)
                
            start_time = time.time()
            result = None
            success = False
            
            try:
                # Execute task
                result = await prefect_decorated(*args, **kwargs)
                success = True
                
                # Record metrics for the result
                if monitor:
                    metrics = {}
                    
                    # Add data metrics if result is DataFrame
                    if hasattr(result, 'shape'):
                        metrics['rows_processed'] = result.shape[0]
                        metrics['columns_processed'] = result.shape[1]
                    
                    # Add duration
                    metrics['duration_seconds'] = time.time() - start_time
                    
                    # Add task-type specific metrics
                    if task_type == "provider":
                        if hasattr(result, 'memory_usage'):
                            metrics['memory_usage_bytes'] = result.memory_usage(deep=True).sum()
                    elif task_type == "analytics":
                        # Add analytics-specific metrics
                        if hasattr(result, 'nunique'):
                            metrics['signal_count'] = result.nunique().get('signal_type', 0)
                    elif task_type == "backtest":
                        # Add backtest-specific metrics
                        if isinstance(result, dict) and 'sharpe_ratio' in result:
                            metrics['sharpe_ratio'] = result['sharpe_ratio']
                            metrics['total_return'] = result.get('total_return', 0)
                    
                    # Record metrics and complete task
                    monitor.complete_task(task_name, success, metrics)
                
                return result
                
            except Exception as e:
                # Record error if monitor exists
                if monitor:
                    monitor.record_error(e, task_name)
                    monitor.complete_task(task_name, False)
                raise
                
        @functools.wraps(prefect_decorated)
        def sync_wrapper(*args, **kwargs):
            # Initialize monitoring
            monitor = None
            if not skip_monitoring:
                monitor = kwargs.pop('monitor', None)
                # Try to get monitor from context if not provided
                if monitor is None:
                    try:
                        from prefect.context import get_run_context
                        ctx = get_run_context()
                        flow_name = getattr(ctx, 'flow_name', "unknown_flow")
                        flow_run_id = getattr(ctx, 'flow_run_id', None)
                        
                        if flow_name and flow_run_id:
                            from quant_research.pipelines.core.monitoring import PipelineMonitor
                            monitor = PipelineMonitor(flow_name, flow_run_id)
                    except Exception:
                        # Continue without monitor if can't create one
                        pass
            
            # Start task monitoring if monitor exists
            if monitor:
                monitor.start_task(task_name)
                
            start_time = time.time()
            result = None
            success = False
            
            try:
                # Execute task
                result = prefect_decorated(*args, **kwargs)
                success = True
                
                # Record metrics for the result
                if monitor:
                    metrics = {}
                    
                    # Add data metrics if result is DataFrame
                    if hasattr(result, 'shape'):
                        metrics['rows_processed'] = result.shape[0]
                        metrics['columns_processed'] = result.shape[1]
                    
                    # Add duration
                    metrics['duration_seconds'] = time.time() - start_time
                    
                    # Add task-type specific metrics
                    if task_type == "provider":
                        if hasattr(result, 'memory_usage'):
                            metrics['memory_usage_bytes'] = result.memory_usage(deep=True).sum()
                    elif task_type == "analytics":
                        # Add analytics-specific metrics
                        if hasattr(result, 'nunique'):
                            metrics['signal_count'] = result.nunique().get('signal_type', 0)
                    elif task_type == "backtest":
                        # Add backtest-specific metrics
                        if isinstance(result, dict) and 'sharpe_ratio' in result:
                            metrics['sharpe_ratio'] = result['sharpe_ratio']
                            metrics['total_return'] = result.get('total_return', 0)
                    
                    # Record metrics and complete task
                    monitor.complete_task(task_name, success, metrics)
                
                return result
                
            except Exception as e:
                # Record error if monitor exists
                if monitor:
                    monitor.record_error(e, task_name)
                    monitor.complete_task(task_name, False)
                raise
                
        # Return appropriate wrapper based on whether function is async
        if is_async:
            return async_wrapper
        return sync_wrapper
        
    return decorator

# Convenience aliases for common task types
def provider_task(**kwargs):
    """Decorator for provider tasks."""
    return pipeline_task(task_type="provider", **kwargs)

def analytics_task(**kwargs):
    """Decorator for analytics tasks."""
    return pipeline_task(task_type="analytics", **kwargs)

def backtest_task(**kwargs):
    """Decorator for backtest tasks."""
    return pipeline_task(task_type="backtest", **kwargs)

def dashboard_task(**kwargs):
    """Decorator for dashboard tasks."""
    return pipeline_task(task_type="dashboard", **kwargs)