# pipelines/core/workflow.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Type, TypeVar
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
import yaml
import json

from prefect import flow, task
from pydantic import BaseModel, ValidationError

from quant_research.pipelines.core.monitoring import PipelineMonitor
from quant_research.pipelines.core.config import PipelineConfig, load_config_from_file

T = TypeVar('T', bound=PipelineConfig)

class PipelineState:
    """State tracking for pipeline execution."""
    
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    
    def __init__(self):
        self.status = self.INITIALIZED
        self.start_time = None
        self.end_time = None
        self.current_stage = None
        self.error = None
        self.stage_results = {}
        self.final_results = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary."""
        return {
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_stage": self.current_stage,
            "error": str(self.error) if self.error else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() 
                               if self.start_time and self.end_time else None
        }


class Pipeline(ABC):
    """
    Enhanced base class for all data pipelines.
    
    This class provides a comprehensive framework for building data pipelines:
    - Configuration loading and validation
    - Execution tracking and monitoring
    - Standardized stage execution
    - Consistent error handling
    - Results management and persistence
    
    Concrete pipelines should:
    1. Override the config_class class variable
    2. Implement the execute() method
    3. Use the provided stage_* methods for structured execution
    """
    
    # Class variables to be overridden by subclasses
    name: str = "base_pipeline"
    description: str = "Base pipeline class"
    version: str = "0.1.0"
    config_class: Type[BaseModel] = PipelineConfig
    
    def __init__(self, config: Union[Dict[str, Any], BaseModel, str, Path]):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration as dict, Pydantic model, or path to config file
        
        Raises:
            ValueError: If configuration validation fails
        """
        # Process and validate configuration
        self.config = self._process_config(config)
        
        # Set up pipeline components
        self.logger = logging.getLogger(f"quant_research.pipelines.{self.__class__.__name__}")
        self.state = PipelineState()
        
        # Set up monitoring
        self.monitor = PipelineMonitor(
            pipeline_name=self.name,
            run_id=getattr(self.config, 'run_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        )
        
        # Ensure output directories exist
        self._ensure_directories()
    
    def _process_config(self, config: Union[Dict[str, Any], BaseModel, str, Path]) -> BaseModel:
        """
        Process and validate configuration.
        
        Args:
            config: Configuration in various formats
            
        Returns:
            Validated configuration as Pydantic model
            
        Raises:
            ValueError: If configuration validation fails
        """
        try:
            if isinstance(config, (str, Path)):
                # Load from file
                if self.config_class is PipelineConfig:
                    return load_config_from_file(config)
                else:
                    # Load raw config and pass to specific config class
                    path = Path(config)
                    if path.suffix.lower() in ['.yaml', '.yml']:
                        with open(path, 'r') as f:
                            config_dict = yaml.safe_load(f)
                    elif path.suffix.lower() == '.json':
                        with open(path, 'r') as f:
                            config_dict = json.load(f)
                    else:
                        raise ValueError(f"Unsupported configuration file format: {path.suffix}")
                    
                    return self.config_class(**config_dict)
            
            elif isinstance(config, dict):
                # Convert dict to Pydantic model
                return self.config_class(**config)
            
            elif isinstance(config, BaseModel):
                # If it's already a Pydantic model
                if isinstance(config, self.config_class):
                    return config
                # If it's a different model, convert to dict first
                return self.config_class(**config.dict())
            
            else:
                raise ValueError(f"Unsupported configuration type: {type(config)}")
                
        except ValidationError as e:
            error_msg = f"Configuration validation failed: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        # Create data directory if specified
        if hasattr(self.config, 'data_dir'):
            os.makedirs(self.config.data_dir, exist_ok=True)
        
        # Create results directory if specified
        if hasattr(self.config, 'results_dir'):
            os.makedirs(self.config.results_dir, exist_ok=True)
    
    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """
        Execute the pipeline and return results.
        
        This method should be implemented by all concrete pipeline classes.
        It should use the stage_* methods for structured execution.
        
        Returns:
            Dictionary with pipeline results
        """
        pass
    
    async def run(self) -> Dict[str, Any]:
        """
        Run the pipeline with monitoring, timing and error handling.
        
        This method should not be overridden by subclasses. Instead,
        override the execute() method.
        
        Returns:
            Dictionary with pipeline results
            
        Raises:
            Exception: If pipeline execution fails
        """
        self.state.status = PipelineState.RUNNING
        self.state.start_time = datetime.now()
        
        # Start monitoring
        self.monitor.start_pipeline()
        self.logger.info(f"Starting pipeline: {self.name} (version {self.version})")
        
        try:
            # Execute the pipeline
            results = await self.execute()
            
            # Store results and update state
            self.state.final_results = results
            self.state.status = PipelineState.COMPLETED
            
            # Log successful completion
            self.logger.info(f"Pipeline completed successfully: {self.name}")
            self.monitor.complete_pipeline(success=True)
            
            return results
            
        except Exception as e:
            # Handle error
            self.state.status = PipelineState.FAILED
            self.state.error = e
            
            # Log error details
            error_details = f"{type(e).__name__}: {str(e)}"
            trace = traceback.format_exc()
            self.logger.error(f"Pipeline failed: {error_details}\n{trace}")
            
            # Record in monitoring
            self.monitor.record_error(e)
            self.monitor.complete_pipeline(success=False)
            
            # Save error report if possible
            try:
                if hasattr(self.config, 'results_dir'):
                    error_file = Path(self.config.results_dir) / f"{self.name}_error_report.json"
                    error_report = {
                        "pipeline": self.name,
                        "version": self.version,
                        "status": "failed",
                        "error": error_details,
                        "traceback": trace,
                        "timestamp": datetime.now().isoformat(),
                        "state": self.state.to_dict()
                    }
                    with open(error_file, 'w') as f:
                        json.dump(error_report, f, indent=2)
                    self.logger.info(f"Error report saved to {error_file}")
            except Exception as save_error:
                self.logger.warning(f"Failed to save error report: {save_error}")
            
            # Re-raise the original exception
            raise
            
        finally:
            # Finalize state
            self.state.end_time = datetime.now()
            duration = (self.state.end_time - self.state.start_time).total_seconds()
            self.logger.info(f"Pipeline execution took {duration:.2f} seconds")
            
            # Save metrics if possible
            try:
                if hasattr(self.config, 'results_dir'):
                    metrics_file = self.monitor.save_metrics(self.config.results_dir)
                    self.logger.info(f"Pipeline metrics saved to {metrics_file}")
            except Exception as metrics_error:
                self.logger.warning(f"Failed to save metrics: {metrics_error}")
    
    async def stage_begin(self, stage_name: str) -> None:
        """
        Begin a pipeline stage with monitoring.
        
        Args:
            stage_name: Name of the stage
        """
        self.state.current_stage = stage_name
        self.logger.info(f"Beginning stage: {stage_name}")
        self.monitor.start_task(stage_name)
    
    async def stage_complete(self, stage_name: str, result: Any = None) -> Any:
        """
        Complete a pipeline stage with monitoring.
        
        Args:
            stage_name: Name of the stage
            result: Optional result from the stage
            
        Returns:
            The result parameter, for chaining
        """
        # Store result
        if result is not None:
            self.state.stage_results[stage_name] = result
            
            # If result is a DataFrame, record data metrics
            if hasattr(result, 'shape') and callable(getattr(result, 'shape', None)):
                try:
                    self.monitor.record_data_metrics(result, description=stage_name)
                except Exception as e:
                    self.logger.warning(f"Failed to record data metrics for {stage_name}: {e}")
        
        # Log and monitor
        self.logger.info(f"Completed stage: {stage_name}")
        
        # Extract metrics if available
        metrics = None
        if hasattr(result, 'shape'):
            try:
                metrics = {"rows": result.shape[0], "columns": result.shape[1]}
            except (AttributeError, IndexError):
                pass
        
        self.monitor.complete_task(stage_name, success=True, metrics=metrics)
        
        return result
    
    async def stage_error(self, stage_name: str, error: Exception) -> None:
        """
        Record error in a pipeline stage.
        
        Args:
            stage_name: Name of the stage
            error: The exception that occurred
        """
        self.logger.error(f"Error in stage {stage_name}: {str(error)}", exc_info=True)
        self.monitor.record_error(error, stage_name)
        self.monitor.complete_task(stage_name, success=False)
    
    def save_results(self, path: Optional[str] = None) -> str:
        """
        Save pipeline results to a file.
        
        Args:
            path: Optional path to save results. If not provided,
                 a default path will be used based on config.
                 
        Returns:
            Path to the saved file
        """
        if path is None:
            if not hasattr(self.config, 'results_dir'):
                raise ValueError("No results_dir in configuration and no path provided")
            
            path = os.path.join(
                self.config.results_dir,
                f"{self.name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare results with metadata
        results_with_metadata = {
            "pipeline": self.name,
            "version": self.version,
            "execution_date": datetime.now().isoformat(),
            "status": self.state.status,
            "duration_seconds": (self.state.end_time - self.state.start_time).total_seconds()
                               if self.state.start_time and self.state.end_time else None,
            "results": self.state.final_results
        }
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        self.logger.info(f"Pipeline results saved to {path}")
        return path

    @classmethod
    def create_from_config(cls, config_path: Union[str, Path]) -> 'Pipeline':
        """
        Create a pipeline instance from a configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Pipeline instance
            
        Raises:
            ValueError: If configuration validation fails
        """
        return cls(config_path)


class PrefectPipeline(Pipeline):
    """
    Pipeline implementation that integrates with Prefect.
    
    This class extends the base Pipeline to add Prefect-specific functionality,
    making it easier to deploy workflows to Prefect.
    """
    
    def __init__(self, config: Union[Dict[str, Any], BaseModel, str, Path]):
        """Initialize with Prefect integration."""
        super().__init__(config)
        self._flow = None
    
    @property
    def flow_name(self) -> str:
        """Get the name for the Prefect flow."""
        return f"{self.name}_flow"
    
    def as_flow(self):
        """
        Convert the pipeline to a Prefect flow.
        
        Returns:
            Prefect flow function
        """
        @flow(name=self.flow_name)
        async def _pipeline_flow(config_path: Optional[str] = None, **kwargs):
            # Load config if path provided, otherwise use existing config
            if config_path:
                config = self._process_config(config_path)
            else:
                config = self.config
                
            # Update config with any kwargs
            if kwargs and isinstance(config, dict):
                config.update(kwargs)
            
            # Run the pipeline
            return await self.run()
        
        self._flow = _pipeline_flow
        return self._flow