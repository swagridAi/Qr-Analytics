"""
Pipeline Monitoring and Observability

This module provides monitoring, logging, metrics collection, and observability
capabilities for the quant_research pipeline framework. It includes:

1. Structured logging configuration
2. Pipeline execution metrics
3. Performance profiling
4. Status tracking and reporting
5. Error monitoring and alerting
6. Integration with Prefect observability
"""

import logging
import time
import json
import os
import traceback
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar

import pandas as pd
from prefect import get_run_logger
import prefect.context as prefect_context

# Type for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


class PipelineMetrics:
    """
    Track and store execution metrics for pipelines.
    
    This class collects metrics on pipeline executions including timing,
    data volumes, success rates, and resource utilization.
    """
    
    def __init__(self, pipeline_name: str, run_id: Optional[str] = None):
        """
        Initialize pipeline metrics tracker.
        
        Args:
            pipeline_name: Name of the pipeline
            run_id: Optional unique identifier for this run
        """
        self.pipeline_name = pipeline_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = None
        self.end_time = None
        self.task_metrics = {}
        self.metrics = {
            "pipeline_name": pipeline_name,
            "run_id": self.run_id,
            "status": "initialized",
            "tasks_total": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "data_processed_rows": 0,
            "data_processed_bytes": 0,
            "errors": [],
        }
    
    def start(self) -> None:
        """Mark the start of pipeline execution."""
        self.start_time = time.time()
        self.metrics["start_time"] = datetime.fromtimestamp(self.start_time).isoformat()
        self.metrics["status"] = "running"
    
    def complete(self, success: bool = True) -> None:
        """
        Mark the completion of pipeline execution.
        
        Args:
            success: Whether the pipeline completed successfully
        """
        self.end_time = time.time()
        self.metrics["end_time"] = datetime.fromtimestamp(self.end_time).isoformat()
        self.metrics["duration_seconds"] = self.end_time - self.start_time
        self.metrics["status"] = "success" if success else "failed"
    
    def record_task_start(self, task_name: str) -> None:
        """
        Record the start of a task.
        
        Args:
            task_name: Name of the task
        """
        if task_name not in self.task_metrics:
            self.task_metrics[task_name] = {
                "status": "running",
                "start_time": time.time(),
                "attempts": 1
            }
            self.metrics["tasks_total"] += 1
        else:
            # Task is being retried
            self.task_metrics[task_name]["attempts"] += 1
            self.task_metrics[task_name]["status"] = "running"
            self.task_metrics[task_name]["retry_time"] = time.time()
    
    def record_task_completion(self, task_name: str, success: bool, 
                              output_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Record the completion of a task.
        
        Args:
            task_name: Name of the task
            success: Whether the task completed successfully
            output_metrics: Optional additional metrics from the task
        """
        if task_name not in self.task_metrics:
            self.record_task_start(task_name)
        
        task_metrics = self.task_metrics[task_name]
        task_metrics["end_time"] = time.time()
        
        if "start_time" in task_metrics:
            task_metrics["duration_seconds"] = task_metrics["end_time"] - task_metrics["start_time"]
        
        task_metrics["status"] = "success" if success else "failed"
        
        if output_metrics:
            task_metrics.update(output_metrics)
        
        if success:
            self.metrics["tasks_completed"] += 1
        else:
            self.metrics["tasks_failed"] += 1
    
    def record_data_metrics(self, rows: int, bytes_size: Optional[int] = None) -> None:
        """
        Record metrics about processed data volume.
        
        Args:
            rows: Number of data rows processed
            bytes_size: Optional size of data in bytes
        """
        self.metrics["data_processed_rows"] += rows
        
        if bytes_size:
            self.metrics["data_processed_bytes"] += bytes_size
    
    def record_error(self, error: Exception, task_name: Optional[str] = None) -> None:
        """
        Record an error that occurred during pipeline execution.
        
        Args:
            error: The exception that occurred
            task_name: Optional name of the task where the error occurred
        """
        error_info = {
            "time": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        
        if task_name:
            error_info["task_name"] = task_name
            
            # Update task metrics
            if task_name in self.task_metrics:
                self.task_metrics[task_name]["error"] = {
                    "type": type(error).__name__,
                    "message": str(error)
                }
        
        self.metrics["errors"].append(error_info)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to a dictionary.
        
        Returns:
            Dictionary with all metrics
        """
        result = self.metrics.copy()
        result["tasks"] = self.task_metrics
        return result
    
    def save_to_file(self, directory: str) -> str:
        """
        Save metrics to a JSON file.
        
        Args:
            directory: Directory to save the metrics file
            
        Returns:
            Path to the saved file
        """
        os.makedirs(directory, exist_ok=True)
        
        filename = f"{self.pipeline_name}_{self.run_id}_metrics.json"
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert pipeline metrics to a DataFrame for analysis.
        
        Returns:
            DataFrame with metrics
        """
        tasks_df = pd.DataFrame([
            {
                "pipeline_name": self.pipeline_name,
                "run_id": self.run_id,
                "task_name": task_name,
                **metrics
            }
            for task_name, metrics in self.task_metrics.items()
        ])
        
        return tasks_df


class PipelineLogger:
    """
    Enhanced logger for pipeline operations with structured logging support.
    
    Provides consistent logging across pipeline components with context
    enrichment and integration with monitoring systems.
    """
    
    def __init__(self, pipeline_name: str, run_id: Optional[str] = None):
        """
        Initialize pipeline logger.
        
        Args:
            pipeline_name: Name of the pipeline
            run_id: Optional unique identifier for this run
        """
        self.pipeline_name = pipeline_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get logger for the pipeline
        self.logger = logging.getLogger(f"quant_research.pipelines.{pipeline_name}")
        
        # Get Prefect logger when running in Prefect
        try:
            self.prefect_logger = get_run_logger()
        except:
            self.prefect_logger = None
    
    def _log(self, level: int, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Log a message with context enrichment.
        
        Args:
            level: Logging level
            msg: Message to log
            extra: Extra context data
            **kwargs: Additional logging arguments
        """
        # Prepare context for structured logging
        log_extra = {
            "pipeline_name": self.pipeline_name,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add extra context if provided
        if extra:
            log_extra.update(extra)
        
        # Log to standard logger
        self.logger.log(level, msg, extra=log_extra, **kwargs)
        
        # Also log to Prefect if available
        if self.prefect_logger:
            self.prefect_logger.log(level, msg, **kwargs)
    
    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, msg, extra, **kwargs)
    
    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, msg, extra, **kwargs)
    
    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, msg, extra, **kwargs)
    
    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, msg, extra, **kwargs)
    
    def critical(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, msg, extra, **kwargs)
    
    def task_start(self, task_name: str) -> None:
        """
        Log task start with standard format.
        
        Args:
            task_name: Name of the task
        """
        self.info(f"Starting task: {task_name}", extra={"task": task_name, "event": "task_start"})
    
    def task_complete(self, task_name: str, duration: Optional[float] = None) -> None:
        """
        Log task completion with standard format.
        
        Args:
            task_name: Name of the task
            duration: Optional duration in seconds
        """
        extra = {"task": task_name, "event": "task_complete"}
        if duration:
            extra["duration_seconds"] = duration
            self.info(f"Completed task: {task_name} in {duration:.2f} seconds", extra=extra)
        else:
            self.info(f"Completed task: {task_name}", extra=extra)
    
    def task_error(self, task_name: str, error: Exception) -> None:
        """
        Log task error with standard format.
        
        Args:
            task_name: Name of the task
            error: The exception that occurred
        """
        self.error(
            f"Error in task {task_name}: {str(error)}",
            extra={
                "task": task_name,
                "event": "task_error",
                "error_type": type(error).__name__,
                "error_message": str(error)
            },
            exc_info=True
        )


class PipelineMonitor:
    """
    Central monitoring system for pipeline execution.
    
    Combines metrics collection, logging, and status tracking for
    comprehensive pipeline observability.
    """
    
    def __init__(self, pipeline_name: str, run_id: Optional[str] = None, 
                log_level: int = logging.INFO):
        """
        Initialize pipeline monitor.
        
        Args:
            pipeline_name: Name of the pipeline
            run_id: Optional unique identifier for this run
            log_level: Logging level for this pipeline
        """
        self.pipeline_name = pipeline_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = PipelineMetrics(pipeline_name, run_id)
        self.logger = PipelineLogger(pipeline_name, run_id)
        
        # Set logging level for this pipeline
        logging.getLogger(f"quant_research.pipelines.{pipeline_name}").setLevel(log_level)
    
    def start_pipeline(self) -> None:
        """Mark the start of pipeline execution."""
        self.metrics.start()
        self.logger.info(
            f"Starting pipeline: {self.pipeline_name}",
            extra={"event": "pipeline_start", "run_id": self.run_id}
        )
    
    def complete_pipeline(self, success: bool = True) -> None:
        """
        Mark the completion of pipeline execution.
        
        Args:
            success: Whether the pipeline completed successfully
        """
        self.metrics.complete(success)
        
        duration = "unknown"
        if self.metrics.start_time and self.metrics.end_time:
            duration = f"{self.metrics.end_time - self.metrics.start_time:.2f} seconds"
        
        if success:
            self.logger.info(
                f"Pipeline completed successfully in {duration}",
                extra={
                    "event": "pipeline_complete",
                    "status": "success",
                    "duration": duration
                }
            )
        else:
            self.logger.error(
                f"Pipeline failed after {duration}",
                extra={
                    "event": "pipeline_complete",
                    "status": "failed",
                    "duration": duration
                }
            )
    
    def start_task(self, task_name: str) -> None:
        """
        Record and log the start of a task.
        
        Args:
            task_name: Name of the task
        """
        self.metrics.record_task_start(task_name)
        self.logger.task_start(task_name)
    
    def complete_task(self, task_name: str, success: bool = True, 
                     metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Record and log the completion of a task.
        
        Args:
            task_name: Name of the task
            success: Whether the task completed successfully
            metrics: Optional additional metrics from the task
        """
        self.metrics.record_task_completion(task_name, success, metrics)
        
        task_metrics = self.metrics.task_metrics.get(task_name, {})
        duration = task_metrics.get("duration_seconds")
        
        if success:
            self.logger.task_complete(task_name, duration)
        else:
            self.logger.error(
                f"Task failed: {task_name}" + (f" after {duration:.2f} seconds" if duration else ""),
                extra={
                    "task": task_name,
                    "event": "task_error",
                    "duration_seconds": duration
                }
            )
    
    def record_error(self, error: Exception, task_name: Optional[str] = None) -> None:
        """
        Record and log an error.
        
        Args:
            error: The exception that occurred
            task_name: Optional name of the task where the error occurred
        """
        self.metrics.record_error(error, task_name)
        
        if task_name:
            self.logger.task_error(task_name, error)
        else:
            self.logger.error(
                f"Pipeline error: {str(error)}",
                extra={
                    "event": "pipeline_error",
                    "error_type": type(error).__name__,
                    "error_message": str(error)
                },
                exc_info=True
            )
    
    def record_data_metrics(self, df: pd.DataFrame, description: str = "data") -> None:
        """
        Record metrics about a DataFrame.
        
        Args:
            df: DataFrame being processed
            description: Description of the data
        """
        rows = len(df)
        
        # Estimate memory usage
        try:
            bytes_size = df.memory_usage(deep=True).sum()
        except:
            bytes_size = None
        
        self.metrics.record_data_metrics(rows, bytes_size)
        
        self.logger.info(
            f"Processed {rows} rows of {description}" + 
            (f" ({bytes_size / 1024 / 1024:.2f} MB)" if bytes_size else ""),
            extra={
                "event": "data_processed",
                "rows": rows,
                "bytes": bytes_size,
                "data_description": description
            }
        )
    
    def save_metrics(self, directory: str) -> str:
        """
        Save metrics to a file.
        
        Args:
            directory: Directory to save metrics
            
        Returns:
            Path to the saved metrics file
        """
        filepath = self.metrics.save_to_file(directory)
        self.logger.info(
            f"Saved pipeline metrics to {filepath}",
            extra={"event": "metrics_saved", "metrics_path": filepath}
        )
        return filepath


def monitor_task(task_name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator for monitoring task execution.
    
    This decorator wraps a task function to record metrics and log events
    at task start and completion. It integrates with the PipelineMonitor
    if one is available.
    
    Args:
        task_name: Optional name for the task (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        nonlocal task_name
        if task_name is None:
            task_name = func.__name__
            
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get monitor from Prefect context
            monitor = None
            try:
                if hasattr(prefect_context, 'get_run_context'):
                    ctx = prefect_context.get_run_context()
                    flow_run_id = getattr(ctx, 'flow_run_id', None)
                    flow_name = getattr(ctx, 'flow_name', "unknown_flow")
                    
                    if flow_run_id:
                        monitor = PipelineMonitor(flow_name, flow_run_id)
            except Exception:
                # If no Prefect context, continue without monitor
                pass
            
            # If no Prefect monitor, check kwargs for monitor
            if not monitor and 'monitor' in kwargs:
                monitor = kwargs.pop('monitor')
            
            # Record start
            start_time = time.time()
            if monitor:
                monitor.start_task(task_name)
            
            try:
                # Execute task
                result = func(*args, **kwargs)
                
                # Record metrics for dataframe results
                if monitor and isinstance(result, pd.DataFrame):
                    monitor.record_data_metrics(result, task_name)
                
                # Record successful completion
                if monitor:
                    # Calculate task-specific metrics
                    task_metrics = {
                        "duration_seconds": time.time() - start_time
                    }
                    
                    # Add data metrics if result is a DataFrame
                    if isinstance(result, pd.DataFrame):
                        task_metrics["rows_processed"] = len(result)
                    
                    monitor.complete_task(task_name, True, task_metrics)
                
                return result
                
            except Exception as e:
                # Record error
                if monitor:
                    monitor.record_error(e, task_name)
                    monitor.complete_task(task_name, False)
                raise
                
        return wrapper
    
    return decorator


def configure_monitoring() -> None:
    """
    Configure global monitoring settings.
    
    Sets up logging formats, handlers, and default configuration
    for pipeline monitoring.
    """
    # Create formatter for structured logging
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger for quant_research.pipelines
    pipeline_logger = logging.getLogger("quant_research.pipelines")
    pipeline_logger.setLevel(logging.INFO)
    
    # Add handler if not already added
    if not pipeline_logger.handlers:
        pipeline_logger.addHandler(console_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    pipeline_logger.propagate = False


# Initialize monitoring on module import
configure_monitoring()