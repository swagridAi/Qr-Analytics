"""
Cryptocurrency intraday trading pipeline.

This module defines a pipeline for cryptocurrency intraday trading research.
"""

import os
from typing import Dict, Any
import yaml
from prefect import flow, task

from quant_research.pipelines.tasks.provider_tasks import fetch_market_data
from quant_research.pipelines.tasks.analytics_tasks import generate_signals
from quant_research.pipelines.tasks.backtest_tasks import run_backtest
from quant_research.pipelines.tasks.dashboard_tasks import generate_performance_report, plot_performance_metrics

@flow(name="crypto_intraday")
async def crypto_intraday_flow(config_path: str) -> Dict[str, Any]:
    """
    Pipeline for cryptocurrency intraday trading strategy research.
    
    This pipeline:
    1. Fetches OHLCV data from cryptocurrency exchanges
    2. Generates trading signals using various analytics
    3. Backtests trading strategies with the signals
    4. Produces performance reports
    
    Args:
        config_path: Path to the pipeline configuration file
        
    Returns:
        Pipeline results
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create data directory if it doesn't exist
    data_dir = config.get("data_dir", "./data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 1: Fetch data from provider
    provider_config = config["provider"]
    prices_path = f"{data_dir}/prices.parquet"
    
    prices_path = await fetch_market_data(
        provider_id=provider_config["id"],
        config=provider_config["config"],
        output_path=prices_path,
        **provider_config.get("fetch_params", {})
    )
    
    # Step 2: Generate signals
    analytics_config = config["analytics"]["modules"]
    signals_path = f"{data_dir}/signals.parquet"
    
    signals_path = await generate_signals(
        data_path=prices_path,
        analytics_config=analytics_config,
        output_path=signals_path
    )
    
    # Step 3: Run backtest
    backtest_config = config["backtest"]
    results_dir = config.get("results_dir", "./results")
    
    metrics = await run_backtest(
        signals_path=signals_path,
        prices_path=prices_path,
        backtest_config=backtest_config,
        output_dir=results_dir
    )
    
    # Step 4: Generate report and visualizations
    report_path = f"{results_dir}/report.json"
    plot_path = f"{results_dir}/metrics_summary.png"
    
    report_path = await generate_performance_report(
        metrics=metrics,
        output_path=report_path,
        title=f"Performance Report: {config.get('name', 'Crypto Intraday')}"
    )
    
    plot_path = await plot_performance_metrics(
        metrics=metrics,
        output_path=plot_path
    )
    
    # Return results
    return {
        "prices_path": prices_path,
        "signals_path": signals_path,
        "backtest_metrics": metrics,
        "report_path": report_path,
        "plot_path": plot_path
    }