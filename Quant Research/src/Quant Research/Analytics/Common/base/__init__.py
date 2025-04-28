"""
Analytics Engine Base Classes

This module provides the foundation for the analytics engine, including base classes
for signal generators, parameter validation, registry pattern, and pipeline implementation.

The module has been refactored to follow better separation of concerns:
- signal_params.py - Parameter validation using Pydantic
- signal_generator.py - Base class for signal generators
- signal_registry.py - Registry pattern for generator discovery
- signal_pipeline.py - Pipeline pattern for sequential execution

For backward compatibility, all classes are re-exported from this module.
"""

# Import all classes for backward compatibility
from .signal_params import SignalGeneratorParams
from .signal_generator import SignalGenerator
from .signal_registry import SignalGeneratorRegistry
from .signal_pipeline import SignalPipeline

# Define public API
__all__ = [
    'SignalGeneratorParams',
    'SignalGenerator',
    'SignalGeneratorRegistry',
    'SignalPipeline'
]