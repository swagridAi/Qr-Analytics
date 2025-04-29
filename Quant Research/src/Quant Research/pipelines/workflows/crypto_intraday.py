"""
Cryptocurrency intraday trading pipeline.

This module defines a pipeline for cryptocurrency intraday trading research.
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from prefect import flow

from quant_research.pipelines.core.monitoring import PipelineMonitor
from quant_research.pipelines.core.workflow import Pipeline
from quant_research.pipelines.tasks.provider_tasks import fetch_market_data
from quant_research.pipelines.tasks.analytics_tasks import generate_signals, filter_signals
from quant_research.pipelines.tasks.backtest_tasks import run_backtest
from quant_research.pipelines.tasks.dashboard_tasks import (
    generate_performance_report, 
    plot_performance_metrics,
    generate_signals_dashboard
)

class CryptoIntradayPipeline(Pipeline):
    """
    Pipeline for cryptocurrency intraday trading strategy research.
    
    This pipeline:
    1. Fetches OHLCV data from cryptocurrency exchanges
    2. Generates trading signals using various analytics
    3. Backtests trading strategies with the signals
    4. Produces performance reports and visualizations
    """
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the pipeline and return results."""
        # Extract configuration
        config = self.config
        
        # Create data and results directories
        data_dir = config.get("data_dir", "./data")
        results_dir = config.get("results_dir", "./results")
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create run-specific directories with timestamp
        run_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        run_data_dir = Path(data_dir) / run_id
        run_results_dir = Path(results_dir) / run_id
        
        os.makedirs(run_data_dir, exist_ok=True)
        os.makedirs(run_results_dir, exist_ok=True)
        
        # Set up monitoring
        monitor = PipelineMonitor(
            pipeline_name="crypto_intraday",
            run_id=run_id
        )
        
        try:
            # Start pipeline monitoring
            monitor.start_pipeline()
            
            # Step 1: Fetch data from provider
            provider_config = config["provider"]
            prices_path = str(run_data_dir / "prices.parquet")
            
            self.logger.info(f"Fetching market data from {provider_config['id']}")
            prices_path = await fetch_market_data(
                provider_id=provider_config["id"],
                config=provider_config["config"],
                output_path=prices_path,
                monitor=monitor,
                **provider_config.get("fetch_params", {})
            )
            
            # Step 2: Generate signals
            analytics_config = config["analytics"]["modules"]
            signals_path = str(run_data_dir / "signals.parquet")
            
            self.logger.info(f"Generating signals using {len(analytics_config)} analytics modules")
            signals_path = await generate_signals(
                data_path=prices_path,
                analytics_config=analytics_config,
                output_path=signals_path,
                combine_signals=config["analytics"].get("combine_signals", True),
                monitor=monitor
            )
            
            # Optional: Filter signals
            if "filter_criteria" in config["analytics"]:
                self.logger.info("Filtering signals based on criteria")
                filtered_signals_path = str(run_data_dir / "filtered_signals.parquet")
                signals_path = filter_signals(
                    signals_path=signals_path,
                    filter_criteria=config["analytics"]["filter_criteria"],
                    output_path=filtered_signals_path,
                    monitor=monitor
                )
            
            # Step 3: Run backtest
            backtest_config = config["backtest"]
            
            self.logger.info(f"Running backtest with {backtest_config['strategy']} strategy")
            metrics = await run_backtest(
                signals_path=signals_path,
                prices_path=prices_path,
                backtest_config=backtest_config,
                output_dir=str(run_results_dir),
                monitor=monitor
            )
            
            # Step 4: Generate report and visualizations
            report_path = str(run_results_dir / "report.json")
            plot_path = str(run_results_dir / "metrics_summary.png")
            
            self.logger.info("Generating performance report")
            report_path = await generate_performance_report(
                metrics=metrics,
                output_path=report_path,
                title=f"Performance Report: {config.get('name', 'Crypto Intraday')}",
                include_market_data=True,
                market_data_path=prices_path,
                monitor=monitor
            )
            
            self.logger.info("Creating performance visualizations")
            plot_path = await plot_performance_metrics(
                metrics=metrics,
                output_path=plot_path,
                figsize=(12, 10),
                include_detailed_metrics=True,
                monitor=monitor
            )
            
            # Additional dashboard visualization
            dashboard_path = str(run_results_dir / "signals_dashboard.png")
            
            self.logger.info("Creating signals dashboard")
            dashboard_path = await generate_signals_dashboard(
                signals_path=signals_path,
                prices_path=prices_path,
                output_path=dashboard_path,
                top_n_signals=10,
                include_price_chart=True,
                monitor=monitor
            )
            
            # Mark pipeline as complete
            monitor.complete_pipeline(success=True)
            
            # Save monitoring metrics
            metrics_path = monitor.save_metrics(str(run_results_dir))
            
            # Return results
            return {
                "run_id": run_id,
                "prices_path": prices_path,
                "signals_path": signals_path,
                "backtest_metrics": metrics,
                "report_path": report_path,
                "plot_path": plot_path,
                "dashboard_path": dashboard_path,
                "metrics_path": metrics_path
            }
            
        except Exception as e:
            # Record error and mark pipeline as failed
            monitor.record_error(e)
            monitor.complete_pipeline(success=False)
            
            # Save metrics even on failure
            try:
                monitor.save_metrics(str(run_results_dir))
            except Exception:
                pass
                
            # Re-raise the exception
            raise

@flow(name="crypto_intraday")
async def crypto_intraday_flow(config_path: str) -> Dict[str, Any]:
    """
    Pipeline for cryptocurrency intraday trading strategy research.
    
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
    return await pipeline.run()