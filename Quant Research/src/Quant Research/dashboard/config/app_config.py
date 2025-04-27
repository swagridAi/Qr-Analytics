"""
Application configuration constants for the Quant Research dashboard.

This module provides centralized configuration values used throughout
the dashboard application.
"""

from pathlib import Path

# Strategy options
STRATEGIES = [
    "momentum",
    "mean_reversion", 
    "cross_exchange_arbitrage", 
    "adaptive_regime", 
    "rl_execution"
]

# Timeframe options
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Default symbols if none available from data
DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]

# Path configurations
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
DEFAULT_CONFIG_PATH = Path(PROJECT_ROOT, "configs", "crypto_intraday.yaml")
DATA_PATH = Path(PROJECT_ROOT, "data")

# Chart styling
CHART_STYLES = {
    "candlestick": {
        "increasing_color": "#26a69a",
        "decreasing_color": "#ef5350"
    },
    "volume": {
        "color": "rgba(100, 100, 200, 0.5)"
    },
    "signals": {
        "buy": {
            "symbol": "triangle-up",
            "size": 10,
            "color": "green"
        },
        "sell": {
            "symbol": "triangle-down",
            "size": 10,
            "color": "red"
        }
    }
}

# Strategy parameter presets
STRATEGY_PRESETS = {
    "momentum": {
        "conservative": {
            "lookback_period": 50,
            "threshold": 0.8
        },
        "aggressive": {
            "lookback_period": 20,
            "threshold": 0.5
        }
    },
    "mean_reversion": {
        "conservative": {
            "z_score_window": 50,
            "entry_threshold": 2.5,
            "exit_threshold": 0.5
        },
        "aggressive": {
            "z_score_window": 20,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5
        }
    }
}

# Performance metrics configuration
PERFORMANCE_METRICS = [
    {"name": "Sharpe Ratio", "description": "Risk-adjusted return relative to volatility"},
    {"name": "Sortino Ratio", "description": "Risk-adjusted return relative to downside volatility"},
    {"name": "Max Drawdown", "description": "Maximum peak-to-trough decline"},
    {"name": "Win Rate", "description": "Percentage of profitable trades"},
    {"name": "Profit Factor", "description": "Ratio of gross profits to gross losses"},
    {"name": "Recovery Time", "description": "Time to recover from maximum drawdown"}
]

# Default chart parameters
DEFAULT_CHART_HEIGHT = 600
DEFAULT_MA_PERIODS = [20, 50, 200]