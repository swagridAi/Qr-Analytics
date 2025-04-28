# src/quant_research/pipelines/crypto_mean_reversion.py
import prefect
from prefect import task, Flow, Parameter
from prefect.tasks.prefect import create_flow_run
import pandas as pd
import logging
from datetime import datetime, timedelta

from quant_research.providers.crypto_ccxt import CCXTProvider, CCXTProviderConfig
from quant_research.analytics.stat_arb.zscore import generate_signal, ZScoreParams
from quant_research.backtest.engine import BacktestEngine
from quant_research.core.storage import save_dataframe, load_dataframe

logger = logging.getLogger(__name__)

@task(name="fetch_crypto_data")
def fetch_market_data(config_dict, start_date, end_date):
    """Fetch cryptocurrency data from exchange"""
    logger.info(f"Fetching market data from {start_date} to {end_date}")
    
    # Create provider config from dict
    config = CCXTProviderConfig(**config_dict)
    
    # Initialize provider
    provider = CCXTProvider(config)
    
    # Connect to exchange
    prefect.context.get("logger").info(f"Connecting to {config.exchange}")
    await provider.connect()
    
    # Fetch data
    data = []
    async for item in provider.fetch_data(
        symbols=config.symbols,
        since=start_date,
        timeframe=config.timeframe
    ):
        data.append(item)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to parquet
    save_dataframe(df, f"data/raw/{config.exchange}_data.parquet")
    
    return df

@task(name="calculate_zscore_signals")
def calculate_signals(prices_df, z_params_dict):
    """Calculate z-score signals for mean reversion"""
    logger.info("Calculating z-score signals")
    
    # Create z-score parameters
    z_params = ZScoreParams(**z_params_dict)
    
    # Define pairs for analysis - simplified example
    pairs = []
    symbols = prices_df['symbol'].unique()
    
    # Create some sample pairs if we have multiple symbols
    if len(symbols) >= 2:
        # Simple example - could be replaced with cointegration finder
        pairs = [(symbols[0], symbols[1], 1.0)]
    
    # Generate signals
    signals_df = generate_signal(prices_df, pairs, z_params)
    
    # Save signals
    save_dataframe(signals_df, "data/signals/zscore_signals.parquet")
    
    return signals_df

@task(name="run_backtest")
def run_backtest(signals_df, prices_df, strategy_params):
    """Run backtest with signals"""
    logger.info("Running backtest with generated signals")
    
    # Create backtest config
    backtest_config = {
        "signals_file": "signals.parquet",  # loaded by engine
        "prices_file": "prices.parquet",    # loaded by engine
        "strategy": "mean_reversion",       # use mean reversion strategy
        "strategy_params": strategy_params,
        "results_dir": "./results"
    }
    
    # Save input files for engine to read
    save_dataframe(signals_df, f"data/{backtest_config['signals_file']}")
    save_dataframe(prices_df, f"data/{backtest_config['prices_file']}")
    
    # Create and run engine
    engine = BacktestEngine(backtest_config)
    success, message = engine.run_backtest()
    
    if not success:
        raise Exception(f"Backtest failed: {message}")
    
    # Get metrics
    metrics = engine.get_metrics()
    logger.info(f"Backtest completed with Sharpe ratio: {metrics.get('sharpe_ratio', 0)}")
    
    return metrics

@task(name="generate_report")
def generate_report(metrics, backtest_id):
    """Generate performance report"""
    logger.info(f"Generating report for backtest {backtest_id}")
    
    # In a real implementation, this might:
    # 1. Create visualizations
    # 2. Generate PDF report
    # 3. Update dashboard data
    
    # For this example, just log metrics
    logger.info(f"Performance metrics: {metrics}")
    
    return metrics

def create_pipeline():
    """Create the orchestration pipeline"""
    with Flow("Crypto Mean Reversion Pipeline") as flow:
        # Parameters
        exchange_config = Parameter("exchange_config", default={
            "name": "crypto_ccxt",
            "exchange": "binance",
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1h"
        })
        
        zscore_params = Parameter("zscore_params", default={
            "window": 60,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5
        })
        
        strategy_params = Parameter("strategy_params", default={
            "max_leverage": 1.0,
            "asymmetric_sizing": True,
            "stop_loss_pct": 2.0
        })
        
        start_date = Parameter("start_date", default=(datetime.now() - timedelta(days=30)))
        end_date = Parameter("end_date", default=datetime.now())
        
        # Task execution
        prices_df = fetch_market_data(exchange_config, start_date, end_date)
        signals_df = calculate_signals(prices_df, zscore_params)
        metrics = run_backtest(signals_df, prices_df, strategy_params)
        report = generate_report(metrics, metrics["backtest_id"])
    
    return flow

# For command-line execution
if __name__ == "__main__":
    flow = create_pipeline()
    flow.run()