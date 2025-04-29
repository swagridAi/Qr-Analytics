"""
Backtest tasks for pipeline workflows.

This module contains tasks for backtest operations in the pipeline.
"""

from typing import Dict, Any
import os

from quant_research.pipelines.core.task import backtest_task
from quant_research.backtest.engine import BacktestEngine
from quant_research.core.storage import load_dataframe

@backtest_task(
    name="run_backtest",
    description="Run a backtest with signals and prices data",
    retries=1,
    timeout_seconds=3600  # 1 hour timeout for long backtests
)
def run_backtest(
    signals_path: str,
    prices_path: str,
    backtest_config: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Run a backtest with the provided signals and prices.
    
    Args:
        signals_path: Path to signals data
        prices_path: Path to price data
        backtest_config: Backtest configuration
        output_dir: Directory to save results
        
    Returns:
        Backtest metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Update config with data paths
    config = backtest_config.copy()
    config["signals_file"] = signals_path
    config["prices_file"] = prices_path
    config["results_dir"] = output_dir
    
    # Create and run engine
    engine = BacktestEngine(config)
    success, message = engine.run_backtest()
    
    if not success:
        raise RuntimeError(f"Backtest failed: {message}")
    
    # Get metrics
    metrics = engine.get_metrics()
    
    # Generate plots if configured
    if config.get("generate_plots", True):
        fig = engine.plot_equity_curve()
        fig.savefig(f"{output_dir}/{engine.backtest_id}/equity_curve.png")
    
    return metrics