# pipelines/core/workflow.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import prefect
from prefect import flow, task

class Pipeline(ABC):
    """
    Base class for all data pipelines.
    
    Defines the common interface and functionality for all pipelines
    in the system, including configuration validation, execution tracking,
    and result handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"quant_research.pipelines.{self.__class__.__name__}")
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """Execute the pipeline and return results."""
        pass
    
    async def run(self) -> Dict[str, Any]:
        """Run the pipeline with timing and logging."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting pipeline: {self.__class__.__name__}")
        
        try:
            results = await self.execute()
            self.results = results
            return results
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            raise
        finally:
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            self.logger.info(f"Pipeline completed in {duration:.2f} seconds")