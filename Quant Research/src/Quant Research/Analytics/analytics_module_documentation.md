How to Use the Analytics Module in a Quant Research Pipeline

This guide explains how to integrate and use the analytics module within the quantitative research pipeline. The analytics module is the heart of the signal generation process, taking financial time series data and producing standardized trading signals.

Overview

The analytics module provides a collection of sophisticated analysis tools for:

Volatility modeling: Both parametric (GARCH) and non-parametric (realized volatility) methods

Regime detection: Identifying market states using HMM and change-point detection

Statistical arbitrage: Finding mean-reverting relationships using cointegration and z-scores

Sentiment analysis: Extracting trading signals from social media sentiment

Each component follows a consistent interface pattern, making it easy to integrate into your research pipeline.

Module Structure

analytics/
├── __init__.py
├── common/              # Shared utilities and base classes
├── volatility/          # Volatility models
│   ├── realized_vol.py  # Range-based and realized volatility estimators
│   └── garch.py         # GARCH family models
├── regimes/             # Market regime identification
│   ├── hmm.py           # Hidden Markov Models
│   └── change_point.py  # Change point detection
├── stat_arb/            # Statistical arbitrage
│   ├── cointegration.py # Cointegration analysis
│   └── zscore.py        # Z-score signal generation
└── sentiment_alpha/     # Sentiment analysis
    └── sentiment_alpha_module.py  # NLP-based sentiment signals

Basic Usage Pattern

All analytics components follow a standardized interface with a generate_signal() function:

from quant_research.analytics.volatility import generate_signal as vol_signal
from quant_research.analytics.regimes import generate_signal as regime_signal
from quant_research.analytics.stat_arb import generate_signal as arb_signal
from quant_research.analytics.sentiment_alpha import generate_signal as sentiment_signal

# Generate signals for different analytics
volatility_signals = vol_signal(price_data, estimator="yang_zhang", window=21)
regime_signals = regime_signal(price_data, method="hmm", n_states=3)

Integrating with the Pipeline

Step 1: Data Preparation

import pandas as pd
from quant_research.analytics.common.data_prep import ensure_datetime_index, validate_dataframe

# Prepare your data
df = ensure_datetime_index(raw_data, timestamp_col="timestamp")
df, errors = validate_dataframe(df, required_columns=["open", "high", "low", "close", "volume"])
if errors:
    logger.warning(f"Data validation issues: {'; '.join(errors)}")

Step 2: Running Analytics Modules

# Generate volatility signals
from quant_research.analytics.volatility import generate_signal

volatility_signals = generate_signal(
    df,
    estimator="garch",
    p=1,
    q=1,
    window=252,
    output_file="data/signals/volatility_signals.parquet"
)

# Generate regime signals
from quant_research.analytics.regimes import generate_signal

regime_signals = generate_signal(
    df,
    method="hmm",
    n_states=3,
    features=["returns", "volatility"],
    output_file="data/signals/regime_signals.parquet"
)

Step 3: Using Multiple Signals Together

from quant_research.analytics.common.base import SignalPipeline

# Create generators
vol_gen = VolatilitySignalGenerator(estimator="yang_zhang", window=21)
regime_gen = RegimeDetectorSignalGenerator(method="hmm", n_states=3)
zscore_gen = ZScoreSignalGenerator(window=60, entry_threshold=2.0, exit_threshold=0.5)

# Create pipeline
pipeline = SignalPipeline([vol_gen, regime_gen, zscore_gen])

# Run all generators and get results
results = pipeline.run(df)

# Access individual results
vol_signals = results["VolatilitySignalGenerator"]
regime_signals = results["RegimeDetectorSignalGenerator"]
zscore_signals = results["ZScoreSignalGenerator"]


from quant_research.analytics.volatility import generate_signal as vol_signal
from quant_research.analytics.regimes import generate_signal as regime_signal
from quant_research.analytics.stat_arb import generate_signal as pair_signal
from quant_research.backtest.engine import run_backtest
from quant_research.backtest.strategies import AdaptiveRegime


# Step 1: Generate volatility signals
vol_signals = vol_signal(
    data,
    estimator="garch",
    p=1, q=1,
    window=252,
    output_file="data/signals/btc_volatility.parquet"
)

# Step 2: Generate regime signals
regime_signals = regime_signal(
    data,
    method="hmm",
    n_states=3,
    output_file="data/signals/btc_regimes.parquet"
)

# Realized volatility using Yang-Zhang estimator
yz_signals = generate_signal(
    df,
    estimator="yang_zhang",
    window=21,
    annualize=True
)

# GARCH volatility forecast
garch_signals = generate_signal(
    df,
    estimator="garch",
    p=1, q=1,
    dist="t",
    window=252,
    horizon=5
)

Regime Detection

from quant_research.analytics.regimes import generate_signal, state_analysis

# Hidden Markov Model regime detection
hmm_regimes = generate_signal(
    df,
    method="hmm",
    n_states=3,
    features=["returns", "volatility", "volume_change"]
)

# Analyze regime characteristics
regime_stats = state_analysis(df, hmm_regimes)
print(f"Regime 0 volatility: {regime_stats[0]['volatility_annualized']:.2%}")
print(f"Regime 1 sharpe ratio: {regime_stats[1]['sharpe_ratio']:.2f}")

# Change point detection
cp_regimes = generate_signal(
    df,
    method="change_point",
    n_bkps=10,
    model="rbf"
)

Statistical Arbitrage

from quant_research.analytics.stat_arb import generate_signal, find_cointegrated_pairs

# Find cointegrated pairs in a universe
pairs = find_cointegrated_pairs(
    universe_df,
    significance_level=0.05,
    max_half_life=30,
    min_half_life=1
)

# Generate trading signals based on z-scores
signals = generate_signal(
    universe_df,
    asset_pairs=[(p.asset1, p.asset2) for p in pairs],
    entry_zscore=2.0,
    exit_zscore=0.5
)

Sentiment Analysis

from quant_research.analytics.sentiment_alpha import generate_signal

# Generate sentiment-based signals
sentiment_signals = generate_signal(
    tweet_df,
    price_df=price_data,
    text_col="content",
    time_col="created_at",
    model_name="finbert/finbert",
    window_sizes=[1, 3, 5, 7, 14]
)

Best Practices

Input Validation: Always validate your input data before passing it to analytics functions.

Signal Storage: Save generated signals to parquet files for persistence and later analysis.

Log Levels: Set appropriate log levels:

import logging
logging.getLogger("quant_research.analytics").setLevel(logging.INFO)

Parameter Tuning: Use the validation notebooks to find optimal parameters for each model.

Chain Processing: Use the output of one analytics module as input to another when building complex signals.

Error Handling: Implement try/except blocks:

try:
    signals = generate_signal(df, **params)
except Exception as e:
    logger.error(f"Signal generation failed: {e}")
    signals = pd.DataFrame()

Conclusion

The analytics module is a powerful framework for generating trading signals from financial time series data. By following the consistent interface pattern and integrating the module into your pipeline, you can generate standardized signals that can be used by the backtesting engine to evaluate trading strategies.

