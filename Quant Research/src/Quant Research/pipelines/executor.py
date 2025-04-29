# pipelines/executor.py
import asyncio
import importlib
import logging
from typing import Dict, Any, Optional
import yaml

from prefect.client import get_client
from prefect.deployments import Deployment

logger = logging.getLogger("quant_research.pipelines.executor")

class PipelineExecutor:
    """
    Executor for running pipelines either synchronously or asynchronously.
    
    This class provides methods to:
    1. Load pipeline configurations
    2. Execute pipelines directly (sync) for development
    3. Deploy and run pipelines via Prefect (async) for production
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    @staticmethod
    async def run_pipeline(pipeline_name: str, config_path: str) -> Dict[str, Any]:
        """
        Run a pipeline synchronously for development.
        
        Args:
            pipeline_name: Name of the pipeline module
            config_path: Path to the configuration file
            
        Returns:
            Pipeline results
        """
        try:
            # Import the pipeline module
            module_name = f"quant_research.pipelines.workflows.{pipeline_name}"
            module = importlib.import_module(module_name)
            
            # Get the flow function
            flow_name = f"{pipeline_name}_flow"
            flow_func = getattr(module, flow_name)
            
            # Run the flow
            return await flow_func(config_path=config_path)
            
        except ImportError:
            logger.error(f"Pipeline module not found: {pipeline_name}")
            raise ValueError(f"Pipeline not found: {pipeline_name}")
        except AttributeError:
            logger.error(f"Flow function not found: {flow_name}")
            raise ValueError(f"Flow function not found: {flow_name}")
    
    @staticmethod
    async def deploy_pipeline(
        pipeline_name: str,
        config_path: str,
        deployment_name: Optional[str] = None
    ) -> str:
        """
        Deploy a pipeline to Prefect for production.
        
        Args:
            pipeline_name: Name of the pipeline module
            config_path: Path to the configuration file
            deployment_name: Optional name for the deployment
            
        Returns:
            Deployment ID
        """
        # Import the pipeline module
        module_name = f"quant_research.pipelines.workflows.{pipeline_name}"
        module = importlib.import_module(module_name)
        
        # Get the flow function
        flow_name = f"{pipeline_name}_flow"
        flow_func = getattr(module, flow_name)
        
        # Create deployment name if not provided
        if deployment_name is None:
            deployment_name = f"{pipeline_name}_deployment"
        
        # Create deployment
        deployment = Deployment.build_from_flow(
            flow=flow_func,
            name=deployment_name,
            parameters={"config_path": config_path},
            tags=["quant_research", pipeline_name]
        )
        
        # Apply deployment
        deployment_id = await deployment.apply()
        
        return deployment_id