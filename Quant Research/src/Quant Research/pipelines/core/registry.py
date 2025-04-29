"""
Task lookup for pipeline orchestration.

This module provides a simple way to look up tasks by name without the complex
registration and discovery mechanisms of the original registry.
"""

from typing import Dict, Callable, Any, Optional
import importlib

# Simple mapping of task name to task function
_task_map: Dict[str, Callable] = {}

def register_task(name: str, task_func: Callable) -> None:
    """Register a task function with a name."""
    _task_map[name] = task_func

def get_task(name: str) -> Optional[Callable]:
    """Get a task function by name."""
    return _task_map.get(name)

def import_tasks(module_path: str) -> None:
    """Import tasks from a module, making them available for lookup."""
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import tasks from {module_path}: {e}")