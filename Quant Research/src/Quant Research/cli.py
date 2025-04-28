# src/quant_research/cli.py
import click
import yaml
import os
import logging
from prefect.executors import LocalDaskExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Quant Research CLI"""
    pass

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--parallel/--no-parallel', default=False, 
              help='Run with parallel execution')
def run(config_file, parallel):
    """Run a pipeline with the given configuration file"""
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Running pipeline with config: {config_file}")
    
    # Detect which pipeline to run based on config
    pipeline_type = os.path.basename(config_file).split('.')[0]
    
    # Import the appropriate pipeline
    if 'crypto' in pipeline_type and 'mean_reversion' in pipeline_type:
        from quant_research.pipelines.crypto_mean_reversion import create_pipeline
    elif 'equities' in pipeline_type:
        from quant_research.pipelines.equities_daily import create_pipeline
    else:
        logger.error(f"Unknown pipeline type: {pipeline_type}")
        return
    
    # Create and run flow
    flow = create_pipeline()
    
    # Set executor based on parallel flag
    if parallel:
        executor = LocalDaskExecutor()
    else:
        executor = None
    
    # Map config sections to parameters
    params = {
        "exchange_config": config.get("provider", {}),
        "zscore_params": config.get("analytics", {}).get("zscore", {}),
        "strategy_params": config.get("backtest", {}).get("strategy_params", {})
    }
    
    # Run the flow
    flow.run(executor=executor, parameters=params)

if __name__ == "__main__":
    cli()