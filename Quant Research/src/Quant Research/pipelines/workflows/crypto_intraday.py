# pipelines/workflows/crypto_intraday.py
import os
from typing import Dict, Any
import yaml
from prefect import flow

from quant_research.pipelines.tasks.provider_tasks import fetch_market_data
from quant_research.pipelines.tasks.analytics_tasks import generate_signals
from quant_research.pipelines.tasks.backtest_tasks import run_backtest
from quant_research.pipelines.core.workflow import Pipeline

class CryptoIntradayPipeline(Pipeline):
    """
    Pipeline for cryptocurrency intraday trading strategy research.
    
    This pipeline:
    1. Fetches OHLCV data from cryptocurrency exchanges
    2. Generates trading signals using various analytics
    3. Backtests trading strategies with the signals
    4. Produces performance reports
    """
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the pipeline flow."""
        config = self.config
        
        # Create data directory if it doesn't exist
        data_dir = config.get("data_dir", "./data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Step 1: Fetch data from provider
        provider_config = config["provider"]
        prices_path = f"{data_dir}/prices.parquet"
        
        prices_path = await fetch_market_data.fn(
            provider_id=provider_config["id"],
            config=provider_config["config"],
            output_path=prices_path,
            **provider_config.get("fetch_params", {})
        )
        
        # Step 2: Generate signals
        analytics_config = config["analytics"]
        signals_path = f"{data_dir}/signals.parquet"
        
        signals_path = await generate_signals.fn(
            data_path=prices_path,
            analytics_config=analytics_config,
            output_path=signals_path
        )
        
        # Step 3: Run backtest
        backtest_config = config["backtest"]
        results_dir = config.get("results_dir", "./results")
        
        metrics = await run_backtest.fn(
            signals_path=signals_path,
            prices_path=prices_path,
            backtest_config=backtest_config,
            output_dir=results_dir
        )
        
        # Return results
        return {
            "prices_path": prices_path,
            "signals_path": signals_path,
            "backtest_metrics": metrics
        }

@flow(name="crypto_intraday")
async def crypto_intraday_flow(config_path: str) -> Dict[str, Any]:
    """
    Prefect flow for the cryptocurrency intraday pipeline.
    
    Args:
        config_path: Path to the pipeline configuration file
        
    Returns:
        Pipeline results
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create and run pipeline
    pipeline = CryptoIntradayPipeline(config)
    results = await pipeline.run()
    
    return results