"""
Task definitions and utilities for pipeline orchestration.

This module provides the core abstractions and utilities for defining, 
registering, and executing pipeline tasks. It serves as a bridge between 
the pipeline orchestration layer and the underlying functional components.

Tasks are the basic building blocks of pipelines, encapsulating specific
operations that can be composed into workflows. Each task type interfaces
with a specific component of the system (providers, analytics, backtest, etc.)
and handles the integration details.
"""

import logging
import time
import inspect
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_type_hints
import asyncio

from pydantic import BaseModel, Field, validator, create_model
from prefect import task as prefect_task

from quant_research.core.models import Signal, PriceBar

# Type variables for task functions
T = TypeVar('T')
Result = TypeVar('Result')

# Configure module logger
logger = logging.getLogger("quant_research.pipelines.task")


class TaskParams(BaseModel):
    """
    Base model for task parameters.
    
    This class provides parameter validation for tasks. Task-specific
    parameter models should inherit from this class.
    
    Attributes:
        name: Optional custom name for the task
        description: Optional description of what the task does
        retry: Number of retry attempts if the task fails
        retry_delay: Delay between retry attempts in seconds
        timeout: Maximum execution time in seconds before timing out
    """
    name: Optional[str] = Field(None, description="Custom name for the task")
    description: Optional[str] = Field(None, description="Description of what the task does")
    retry: int = Field(0, description="Number of retry attempts if the task fails")
    retry_delay: int = Field(30, description="Delay between retry attempts in seconds")
    timeout: Optional[int] = Field(None, description="Maximum execution time in seconds")
    
    @validator('retry')
    def validate_retry(cls, v):
        if v < 0:
            raise ValueError("Retry count cannot be negative")
        return v
    
    @validator('retry_delay')
    def validate_retry_delay(cls, v):
        if v < 0:
            raise ValueError("Retry delay cannot be negative")
        return v
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    class Config:
        extra = "forbid"  # Forbid extra parameters not defined in the model


class Task(ABC):
    """
    Base class for all pipeline tasks.
    
    This abstract class defines the common interface and functionality for
    all tasks in the pipeline. It provides methods for parameter validation,
    execution tracking, and result handling.
    
    Attributes:
        name: Name of the task (defaults to class name)
        description: Description of what the task does
        params: Validated parameters for the task
        logger: Logger instance for this task
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the task.
        
        Args:
            **kwargs: Task parameters to validate
        """
        self.params = self._validate_params(kwargs)
        self.name = self.params.name or self.__class__.__name__
        self.description = self.params.description or self.__doc__.split('\n')[0] if self.__doc__ else "No description"
        self.logger = logging.getLogger(f"quant_research.pipelines.task.{self.name}")
    
    def _validate_params(self, params: Dict[str, Any]) -> TaskParams:
        """
        Validate task parameters against the task's parameter model.
        
        This method should be overridden by subclasses to use their
        specific parameter models.
        
        Args:
            params: Dictionary of parameters to validate
            
        Returns:
            Validated parameter model instance
        """
        return TaskParams(**params)
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the task and return the result.
        
        Args:
            **kwargs: Additional runtime parameters
            
        Returns:
            Task execution result
        """
        pass
    
    async def run(self, **kwargs) -> Any:
        """
        Run the task with timing, logging, and error handling.
        
        Args:
            **kwargs: Additional runtime parameters
            
        Returns:
            Task execution result
        """
        start_time = time.time()
        self.logger.info(f"Starting task: {self.name}")
        
        try:
            # Execute the task
            result = await self.execute(**kwargs)
            
            # Log completion
            duration = time.time() - start_time
            self.logger.info(f"Task {self.name} completed in {duration:.2f} seconds")
            
            return result
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            self.logger.error(f"Task {self.name} failed after {duration:.2f} seconds: {str(e)}", exc_info=True)
            
            # Handle retries if configured
            if self.params.retry > 0:
                for attempt in range(1, self.params.retry + 1):
                    self.logger.info(f"Retrying task {self.name} (attempt {attempt}/{self.params.retry}) in {self.params.retry_delay}s...")
                    await asyncio.sleep(self.params.retry_delay)
                    
                    try:
                        result = await self.execute(**kwargs)
                        self.logger.info(f"Task {self.name} succeeded on retry {attempt}")
                        return result
                    except Exception as retry_e:
                        self.logger.error(f"Retry {attempt} failed: {str(retry_e)}")
            
            # Re-raise the exception after all retries fail
            raise

    def to_prefect_task(self) -> Callable:
        """
        Convert this task to a Prefect task.
        
        Returns:
            Prefect task function
        """
        @prefect_task(
            name=self.name,
            description=self.description,
            retries=self.params.retry,
            retry_delay_seconds=self.params.retry_delay,
            timeout_seconds=self.params.timeout
        )
        async def prefect_task_fn(**kwargs):
            return await self.run(**kwargs)
        
        return prefect_task_fn


class TaskRegistry:
    """
    Registry for task implementations.
    
    This class maintains a registry of available tasks and provides methods
    to register, retrieve, and instantiate them.
    """
    
    _registry: Dict[str, Type[Task]] = {}
    
    @classmethod
    def register(cls, task_class: Type[Task]) -> None:
        """
        Register a task class.
        
        Args:
            task_class: Task class to register
            
        Raises:
            TypeError: If the class does not inherit from Task
        """
        if not issubclass(task_class, Task):
            raise TypeError(f"Class {task_class.__name__} must inherit from Task")
        
        task_name = task_class.__name__
        cls._registry[task_name] = task_class
        logger.debug(f"Registered task: {task_name}")
    
    @classmethod
    def get_task_class(cls, task_name: str) -> Optional[Type[Task]]:
        """
        Get a task class by name.
        
        Args:
            task_name: Name of the task class
            
        Returns:
            Task class or None if not found
        """
        return cls._registry.get(task_name)
    
    @classmethod
    def create_task(cls, task_name: str, **params) -> Optional[Task]:
        """
        Create a task instance by name.
        
        Args:
            task_name: Name of the task to create
            **params: Parameters to pass to the task constructor
            
        Returns:
            Task instance or None if not found
            
        Raises:
            ValueError: If the task name is not registered
        """
        task_class = cls.get_task_class(task_name)
        if task_class is None:
            raise ValueError(f"Task not found: {task_name}")
        
        try:
            return task_class(**params)
        except Exception as e:
            logger.error(f"Error creating task {task_name}: {str(e)}")
            raise
    
    @classmethod
    def list_tasks(cls) -> List[str]:
        """
        Get a list of all registered task names.
        
        Returns:
            List of task names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_task_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered tasks.
        
        Returns:
            Dictionary mapping task names to their information
        """
        info = {}
        for task_name, task_class in cls._registry.items():
            info[task_name] = {
                "name": task_name,
                "description": task_class.__doc__.split('\n')[0] if task_class.__doc__ else "No description",
                "module": task_class.__module__
            }
        return info


# Decorator for registering task classes
def register_task(task_class: Optional[Type[Task]] = None):
    """
    Decorator to register a task class with the registry.
    
    Usage:
        @register_task
        class MyTask(Task):
            ...
    
    Args:
        task_class: Task class to register
        
    Returns:
        The task class after registration
    """
    def decorator(cls):
        TaskRegistry.register(cls)
        return cls
    
    if task_class is not None:
        return decorator(task_class)
    return decorator


# Function decorator for creating simple tasks
def task(
    name: Optional[str] = None,
    description: Optional[str] = None,
    retry: int = 0,
    retry_delay: int = 30,
    timeout: Optional[int] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to create a simple task from a function.
    
    This decorator provides a convenient way to create tasks from functions
    without defining a full Task class.
    
    Usage:
        @task(name="fetch_data", retry=3)
        async def fetch_market_data(provider_id: str, **params) -> pd.DataFrame:
            # Implementation
            ...
    
    Args:
        name: Name for the task
        description: Description of what the task does
        retry: Number of retry attempts
        retry_delay: Delay between retries in seconds
        timeout: Maximum execution time in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Get function signature for parameter validation
        sig = inspect.signature(func)
        
        # Create a function-specific Task class
        task_name = name or func.__name__
        
        @register_task
        class FunctionTask(Task):
            """Task created from a function."""
            
            def _validate_params(self, params: Dict[str, Any]) -> TaskParams:
                """Validate using the base TaskParams model."""
                return TaskParams(**params)
            
            async def execute(self, **kwargs) -> Any:
                """Execute the wrapped function."""
                # Check for missing required parameters
                for param_name, param in sig.parameters.items():
                    if param.default == inspect.Parameter.empty and param_name not in kwargs:
                        raise ValueError(f"Missing required parameter: {param_name}")
                
                # Call the function with the provided parameters
                if asyncio.iscoroutinefunction(func):
                    return await func(**kwargs)
                else:
                    return func(**kwargs)
        
        # Create a task instance with the provided parameters
        task_instance = FunctionTask(
            name=task_name,
            description=description or func.__doc__,
            retry=retry,
            retry_delay=retry_delay,
            timeout=timeout
        )
        
        # Create wrapper function that runs the task
        @wraps(func)
        async def wrapper(**kwargs):
            return await task_instance.run(**kwargs)
        
        # Store the task instance on the wrapper for access
        wrapper.task = task_instance
        wrapper.to_prefect_task = task_instance.to_prefect_task
        
        return wrapper
    
    return decorator


# Task base classes for different components

class ProviderTask(Task):
    """
    Base class for tasks that interface with data providers.
    
    This class provides common functionality for tasks that fetch data
    from external sources using the providers module.
    """
    
    class Params(TaskParams):
        """Parameters for provider tasks."""
        provider_id: str = Field(..., description="ID of the provider to use")
        provider_config: Dict[str, Any] = Field({}, description="Provider configuration")
    
    def _validate_params(self, params: Dict[str, Any]) -> Params:
        """Validate using the ProviderTask.Params model."""
        return self.Params(**params)


class AnalyticsTask(Task):
    """
    Base class for tasks that perform analytics operations.
    
    This class provides common functionality for tasks that generate signals
    using the analytics module.
    """
    
    class Params(TaskParams):
        """Parameters for analytics tasks."""
        input_data_path: Optional[str] = Field(None, description="Path to input data")
        output_path: Optional[str] = Field(None, description="Path to save output signals")
    
    def _validate_params(self, params: Dict[str, Any]) -> Params:
        """Validate using the AnalyticsTask.Params model."""
        return self.Params(**params)


class BacktestTask(Task):
    """
    Base class for tasks that perform backtesting operations.
    
    This class provides common functionality for tasks that evaluate trading
    strategies using the backtest module.
    """
    
    class Params(TaskParams):
        """Parameters for backtest tasks."""
        signals_path: Optional[str] = Field(None, description="Path to signal data")
        prices_path: Optional[str] = Field(None, description="Path to price data")
        strategy: Optional[str] = Field(None, description="Strategy to use")
        strategy_params: Dict[str, Any] = Field({}, description="Strategy parameters")
    
    def _validate_params(self, params: Dict[str, Any]) -> Params:
        """Validate using the BacktestTask.Params model."""
        return self.Params(**params)


class DashboardTask(Task):
    """
    Base class for tasks that generate visualizations and reports.
    
    This class provides common functionality for tasks that create
    visualizations and reports using the dashboard module.
    """
    
    class Params(TaskParams):
        """Parameters for dashboard tasks."""
        input_data: Optional[Dict[str, Any]] = Field(None, description="Input data for visualization")
        output_path: Optional[str] = Field(None, description="Path to save output")
        template: Optional[str] = Field(None, description="Template to use for the report")
    
    def _validate_params(self, params: Dict[str, Any]) -> Params:
        """Validate using the DashboardTask.Params model."""
        return self.Params(**params)