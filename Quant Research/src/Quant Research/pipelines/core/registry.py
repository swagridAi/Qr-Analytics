"""
Registry for pipeline tasks and workflows.

This module provides a centralized registry for tasks and workflows,
enabling discovery, lookup, and creation by name. It follows a similar
pattern to the provider and signal generator registries.
"""

import importlib
import inspect
import logging
import os
import pkgutil
from functools import wraps
from typing import Dict, List, Any, Callable, Optional, Type, Union

import prefect
from prefect import task, flow

from quant_research.pipelines.core.workflow import Pipeline

logger = logging.getLogger("quant_research.pipelines.registry")


class TaskRegistry:
    """
    Registry for pipeline tasks.
    
    Tasks are registered with a unique name and can be looked up
    and created by name. This allows for dynamic task creation
    in pipeline workflows.
    """
    
    # Internal storage for registered tasks
    _tasks: Dict[str, Callable] = {}
    
    # Set to track discovered modules to avoid duplicates
    _discovered_modules: set = set()
    
    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        """
        Register a task function in the registry.
        
        Args:
            name: Optional custom name for the task. If not provided,
                  the function name will be used.
                  
        Returns:
            Decorator function that registers the task
            
        Usage:
            @TaskRegistry.register("fetch_data")
            @task
            def fetch_market_data(...):
                ...
        """
        def decorator(func: Callable) -> Callable:
            # Determine task name
            task_name = name or func.__name__
            
            # Check if the function is already a Prefect task
            if not hasattr(func, "task_run"):
                logger.warning(
                    f"Function {task_name} is not a Prefect task. "
                    "Consider using @task before @TaskRegistry.register."
                )
            
            # Register the task
            cls._tasks[task_name] = func
            logger.debug(f"Registered task: {task_name}")
            
            return func
        
        # Handle case where decorator is used without parentheses
        if callable(name):
            func = name
            name = func.__name__
            return decorator(func)
        
        return decorator
    
    @classmethod
    def get_task(cls, name: str) -> Optional[Callable]:
        """
        Get a registered task by name.
        
        Args:
            name: Name of the task to retrieve
            
        Returns:
            Task function or None if not found
        """
        return cls._tasks.get(name)
    
    @classmethod
    def list_tasks(cls) -> List[str]:
        """
        List all registered task names.
        
        Returns:
            List of registered task names
        """
        return list(cls._tasks.keys())
    
    @classmethod
    def discover_tasks(cls) -> int:
        """
        Discover and register tasks in the tasks directory.
        
        This method scans the tasks directory for modules and registers
        any functions decorated with @task found in those modules.
        
        Returns:
            Number of tasks discovered and registered
        """
        import quant_research.pipelines.tasks as tasks_pkg
        
        count = 0
        
        # Get package path and prefix
        package_path = tasks_pkg.__path__
        prefix = tasks_pkg.__name__ + "."
        
        # Iterate through modules
        for _, module_name, is_pkg in pkgutil.iter_modules(package_path, prefix):
            # Skip packages and modules already discovered
            if is_pkg or module_name in cls._discovered_modules:
                continue
            
            try:
                # Import the module
                module = importlib.import_module(module_name)
                cls._discovered_modules.add(module_name)
                
                # Find task functions in the module
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    
                    # Check if it's a function with the task_run attribute (Prefect task)
                    if inspect.isfunction(item) and hasattr(item, "task_run"):
                        # Register if not already registered
                        if item_name not in cls._tasks.values():
                            cls.register(item_name)(item)
                            count += 1
                            
            except Exception as e:
                logger.error(f"Error discovering tasks in {module_name}: {e}")
        
        return count


class WorkflowRegistry:
    """
    Registry for pipeline workflows.
    
    Workflows are registered with a unique name and can be looked up
    and created by name. This allows for dynamic workflow creation
    and execution.
    """
    
    # Internal storage for registered workflows
    _workflows: Dict[str, Union[Type[Pipeline], Callable]] = {}
    
    # Set to track discovered modules to avoid duplicates
    _discovered_modules: set = set()
    
    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        """
        Register a workflow class or flow function.
        
        Args:
            name: Optional custom name for the workflow. If not provided,
                  the class/function name will be used.
                  
        Returns:
            Decorator function that registers the workflow
            
        Usage:
            @WorkflowRegistry.register("crypto_daily")
            class CryptoDailyPipeline(Pipeline):
                ...
                
            or
            
            @WorkflowRegistry.register("crypto_intraday")
            @flow
            def crypto_intraday_flow(...):
                ...
        """
        def decorator(obj: Union[Type[Pipeline], Callable]) -> Union[Type[Pipeline], Callable]:
            # Determine workflow name
            workflow_name = name or obj.__name__
            
            # Check if it's a Pipeline class or Prefect flow
            is_valid = (inspect.isclass(obj) and issubclass(obj, Pipeline)) or \
                      (inspect.isfunction(obj) and hasattr(obj, "fn"))
            
            if not is_valid:
                logger.warning(
                    f"Object {workflow_name} is neither a Pipeline subclass nor a Prefect flow. "
                    "It may not work properly in the pipeline system."
                )
            
            # Register the workflow
            cls._workflows[workflow_name] = obj
            logger.debug(f"Registered workflow: {workflow_name}")
            
            return obj
        
        # Handle case where decorator is used without parentheses
        if callable(name):
            obj = name
            name = obj.__name__
            return decorator(obj)
        
        return decorator
    
    @classmethod
    def get_workflow(cls, name: str) -> Optional[Union[Type[Pipeline], Callable]]:
        """
        Get a registered workflow by name.
        
        Args:
            name: Name of the workflow to retrieve
            
        Returns:
            Workflow class, flow function, or None if not found
        """
        return cls._workflows.get(name)
    
    @classmethod
    def create_workflow(cls, name: str, **kwargs) -> Optional[Pipeline]:
        """
        Create a workflow instance by name.
        
        Args:
            name: Name of the workflow to create
            **kwargs: Parameters to pass to the workflow constructor
            
        Returns:
            Pipeline instance or None if not found
        """
        workflow_class = cls.get_workflow(name)
        
        if workflow_class is None:
            return None
        
        # Handle both Pipeline classes and Prefect flows
        if inspect.isclass(workflow_class) and issubclass(workflow_class, Pipeline):
            return workflow_class(**kwargs)
        else:
            # For Prefect flows, we don't instantiate them, just return the function
            return workflow_class
    
    @classmethod
    def list_workflows(cls) -> List[str]:
        """
        List all registered workflow names.
        
        Returns:
            List of registered workflow names
        """
        return list(cls._workflows.keys())
    
    @classmethod
    def discover_workflows(cls) -> int:
        """
        Discover and register workflows in the workflows directory.
        
        This method scans the workflows directory for modules and registers
        any Pipeline subclasses or functions decorated with @flow.
        
        Returns:
            Number of workflows discovered and registered
        """
        import quant_research.pipelines.workflows as workflows_pkg
        
        count = 0
        
        # Get package path and prefix
        package_path = workflows_pkg.__path__
        prefix = workflows_pkg.__name__ + "."
        
        # Iterate through modules
        for _, module_name, is_pkg in pkgutil.iter_modules(package_path, prefix):
            # Skip packages and modules already discovered
            if is_pkg or module_name in cls._discovered_modules:
                continue
            
            try:
                # Import the module
                module = importlib.import_module(module_name)
                cls._discovered_modules.add(module_name)
                
                # Find workflow classes and flow functions in the module
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    
                    # Check if it's a Pipeline subclass
                    if inspect.isclass(item) and issubclass(item, Pipeline) and item != Pipeline:
                        cls.register(item_name)(item)
                        count += 1
                    
                    # Check if it's a Prefect flow function
                    elif inspect.isfunction(item) and hasattr(item, "fn"):
                        cls.register(item_name)(item)
                        count += 1
                        
            except Exception as e:
                logger.error(f"Error discovering workflows in {module_name}: {e}")
        
        return count


def register_task(name: Optional[str] = None) -> Callable:
    """
    Decorator to register a task with the TaskRegistry.
    
    This is a convenience wrapper that combines @task with @TaskRegistry.register.
    
    Args:
        name: Optional custom name for the task
        
    Returns:
        Decorator function
        
    Usage:
        @register_task("fetch_data")
        def fetch_market_data(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        # Create Prefect task
        task_func = task(func)
        
        # Register with registry
        registered_task = TaskRegistry.register(name)(task_func)
        
        return registered_task
    
    # Handle case where decorator is used without parentheses
    if callable(name):
        func = name
        name = func.__name__
        return decorator(func)
    
    return decorator


def register_workflow(name: Optional[str] = None) -> Callable:
    """
    Decorator to register a workflow with the WorkflowRegistry.
    
    For functions, this combines @flow with @WorkflowRegistry.register.
    For classes, it just registers the class with @WorkflowRegistry.register.
    
    Args:
        name: Optional custom name for the workflow
        
    Returns:
        Decorator function
        
    Usage:
        @register_workflow("crypto_daily")
        def crypto_daily_pipeline(...):
            ...
            
        or
        
        @register_workflow("crypto_intraday")
        class CryptoIntradayPipeline(Pipeline):
            ...
    """
    def decorator(obj: Union[Callable, Type[Pipeline]]) -> Union[Callable, Type[Pipeline]]:
        # For functions, create Prefect flow
        if inspect.isfunction(obj):
            flow_func = flow(obj)
            registered_workflow = WorkflowRegistry.register(name)(flow_func)
            return registered_workflow
        
        # For Pipeline classes, just register
        registered_workflow = WorkflowRegistry.register(name)(obj)
        return registered_workflow
    
    # Handle case where decorator is used without parentheses
    if callable(name):
        obj = name
        name = obj.__name__
        return decorator(obj)
    
    return decorator


# Initialize registries by discovering tasks and workflows
def initialize_registries():
    """
    Initialize the task and workflow registries by discovering
    all available tasks and workflows.
    """
    task_count = TaskRegistry.discover_tasks()
    workflow_count = WorkflowRegistry.discover_workflows()
    
    logger.info(f"Discovered {task_count} tasks and {workflow_count} workflows")