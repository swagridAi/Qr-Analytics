"""
Statistical arbitrage cointegration analysis module.

This module provides functionality for detecting cointegrated pairs,
estimating reversion parameters, and generating trading signals.
"""

# Standard library imports
import itertools
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

# Local imports
from quant_research.core.models import Signal

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Data Structures
# --------------------------------------------------------------------------------------


@dataclass
class CointegrationResult:
    """Results from cointegration testing."""
    asset1: str
    asset2: str
    is_cointegrated: bool
    p_value: float
    half_life: float
    coint_t_stat: float
    coint_coef: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    spread_series: Optional[pd.Series] = None


@dataclass
class CointegrationParams:
    """Parameters for cointegration testing and signal generation."""
    # Cointegration test parameters
    significance_level: float = 0.05
    max_half_life: float = 30
    min_half_life: float = 1
    
    # Signal generation parameters
    lookback_window: int = 60
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    signal_expiry: int = 5
    
    # Additional options
    parallel: bool = False
    filter_momentum: bool = False
    momentum_window: int = 5


# --------------------------------------------------------------------------------------
# Statistical Functions
# --------------------------------------------------------------------------------------


def estimate_half_life(spread: pd.Series) -> float:
    """
    Estimate the half-life of mean reversion for a spread series.
    
    Args:
        spread: Time series of the spread between two assets
        
    Returns:
        Half-life in terms of the time units of the spread series
    """
    # Lagged version of spread
    spread_lag = spread.shift(1)
    # Change in spread
    spread_diff = spread - spread_lag
    
    # Remove NAs from lagged values
    spread_lag = spread_lag.dropna()
    spread = spread.iloc[1:]
    spread_diff = spread_diff.dropna()
    
    # Perform linear regression: spread_diff = beta * spread_lag + alpha
    spread_lag_with_const = sm.add_constant(spread_lag)
    model = sm.OLS(spread_diff, spread_lag_with_const)
    result = model.fit()
    
    # Extract beta
    beta = result.params[1]
    
    # Calculate half-life: t such that 0.5 = exp(beta * t)
    if beta >= 0:
        # Not mean-reverting
        return np.inf
    else:
        half_life = -np.log(2) / beta
        return half_life


def test_cointegration(
    asset1_prices: pd.Series, 
    asset2_prices: pd.Series,
    params: Optional[CointegrationParams] = None
) -> CointegrationResult:
    """
    Test for cointegration between two price series.
    
    Args:
        asset1_prices: Price series for first asset
        asset2_prices: Price series for second asset
        params: Parameters for testing, or None to use defaults
        
    Returns:
        CointegrationResult object with test results
    """
    # Use default parameters if none provided
    if params is None:
        params = CointegrationParams()
    
    # Make sure series are aligned
    joined = pd.concat([asset1_prices, asset2_prices], axis=1).dropna()
    
    if len(joined) < 60:  # Need enough data points
        logger.warning(
            f"Not enough data to test cointegration between "
            f"{asset1_prices.name} and {asset2_prices.name}"
        )
        return CointegrationResult(
            asset1=asset1_prices.name,
            asset2=asset2_prices.name,
            is_cointegrated=False,
            p_value=1.0,
            half_life=np.inf,
            coint_t_stat=0.0,
            coint_coef=0.0,
            hedge_ratio=0.0,
            spread_mean=0.0,
            spread_std=0.0
        )
    
    # Extract aligned series
    a1 = joined.iloc[:, 0]
    a2 = joined.iloc[:, 1]
    
    # Perform cointegration test
    coint_result = coint(a1, a2)
    t_stat, p_val, critical_values = coint_result
    
    # Determine hedge ratio by regressing asset1 on asset2
    model = sm.OLS(a1, sm.add_constant(a2)).fit()
    hedge_ratio = model.params[1]
    
    # Calculate spread series
    spread = a1 - hedge_ratio * a2
    
    # Calculate half-life of mean reversion
    hl = estimate_half_life(spread)
    
    # Determine if pair is cointegrated based on p-value and half-life
    is_cointegrated = (
        (p_val < params.significance_level) and 
        (hl < params.max_half_life) and 
        (hl > params.min_half_life)
    )
    
    return CointegrationResult(
        asset1=a1.name,
        asset2=a2.name,
        is_cointegrated=is_cointegrated,
        p_value=p_val,
        half_life=hl,
        coint_t_stat=t_stat,
        coint_coef=model.params[0],
        hedge_ratio=hedge_ratio,
        spread_mean=spread.mean(),
        spread_std=spread.std(),
        spread_series=spread
    )


# --------------------------------------------------------------------------------------
# Pair Discovery Functions
# --------------------------------------------------------------------------------------


def find_cointegrated_pairs(
    price_df: pd.DataFrame,
    params: Optional[CointegrationParams] = None
) -> List[CointegrationResult]:
    """
    Find all cointegrated pairs in a universe of assets.
    
    Args:
        price_df: DataFrame with price series for multiple assets (columns)
        params: Parameters for testing, or None to use defaults
        
    Returns:
        List of CointegrationResult objects for cointegrated pairs
    """
    # Use default parameters if none provided
    if params is None:
        params = CointegrationParams()
    
    assets = price_df.columns.tolist()
    n = len(assets)
    cointegrated_pairs = []
    
    # If parallel processing is requested and available, use it
    if params.parallel:
        # This is a placeholder for parallel implementation
        logger.warning("Parallel processing requested but not implemented. Using serial processing.")
    
    # Iterate through all possible pairs
    pairs_tested = 0
    total_pairs = n * (n - 1) // 2
    
    for i, j in itertools.combinations(range(n), 2):
        asset1, asset2 = assets[i], assets[j]
        
        # Test for cointegration
        result = test_cointegration(
            asset1_prices=price_df[asset1],
            asset2_prices=price_df[asset2],
            params=params
        )
        
        # Log progress
        pairs_tested += 1
        if pairs_tested % 100 == 0 or pairs_tested == total_pairs:
            logger.info(f"Tested {pairs_tested}/{total_pairs} pairs")
        
        # If cointegrated, add to results
        if result.is_cointegrated:
            cointegrated_pairs.append(result)
            logger.info(
                f"Found cointegrated pair: {asset1} - {asset2} "
                f"(p-value: {result.p_value:.4f}, half-life: {result.half_life:.2f})"
            )
    
    logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs out of {pairs_tested} tested")
    return cointegrated_pairs


# --------------------------------------------------------------------------------------
# Z-Score and Signal Generation Functions
# --------------------------------------------------------------------------------------


def calculate_zscore(spread: pd.Series, lookback: int = 60) -> pd.Series:
    """
    Calculate rolling z-score for a spread series.
    
    Args:
        spread: Spread time series
        lookback: Window for calculating rolling mean and std
        
    Returns:
        Series with rolling z-scores
    """
    # Calculate rolling mean and std
    rolling_mean = spread.rolling(window=lookback).mean()
    rolling_std = spread.rolling(window=lookback).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    # Calculate z-score
    zscore = (spread - rolling_mean) / rolling_std
    
    return zscore


def _create_signal(
    timestamp, 
    asset: str, 
    signal_type: int, 
    z: float, 
    entry_zscore: float,
    pair_info: Optional[Dict] = None,
    signal_expiry: int = 5,
    strength_multiplier: float = 1.0
) -> Signal:
    """
    Helper function to create a Signal object.
    
    Args:
        timestamp: Signal timestamp
        asset: Asset identifier
        signal_type: 1 (long), -1 (short), or 0 (exit)
        z: Current z-score value
        entry_zscore: Threshold for entry
        pair_info: Optional pair trading metadata
        signal_expiry: How many periods a signal remains valid
        strength_multiplier: Multiplier for signal strength
        
    Returns:
        Signal object
    """
    # Determine signal strength based on z-score and signal type
    if signal_type == 0:  # Exit signal
        strength = 1.0
    else:  # Entry signal (long or short)
        # Cap strength at 3x
        strength = min(abs(z) / entry_zscore, 3.0) * strength_multiplier
    
    # Create metadata
    metadata = {
        'zscore': z,
        'strategy': 'stat_arb_cointegration',
        'expiry': signal_expiry
    }
    
    # Add pair information if available
    if pair_info:
        metadata.update(pair_info)
    
    return Signal(
        timestamp=timestamp,
        asset=asset,
        signal=signal_type,
        strength=strength,
        metadata=metadata
    )


def _generate_pair_signals(
    timestamp,
    z: float,
    asset1: str,
    asset2: str,
    hedge_ratio: float,
    entry_zscore: float,
    exit_zscore: float,
    signal_expiry: int
) -> List[Signal]:
    """
    Generate signals for a pair at a specific timestamp.
    
    Args:
        timestamp: Signal timestamp
        z: Current z-score value
        asset1: First asset in pair
        asset2: Second asset in pair
        hedge_ratio: Hedge ratio between assets
        entry_zscore: Threshold for entry
        exit_zscore: Threshold for exit
        signal_expiry: How many periods a signal remains valid
        
    Returns:
        List of Signal objects
    """
    signals = []
    pair_info = {
        'pair': f"{asset1}-{asset2}",
        'hedge_ratio': hedge_ratio
    }
    
    # Long the spread when z-score is very negative
    if z < -entry_zscore:
        # Long asset1, short asset2
        signals.append(_create_signal(
            timestamp=timestamp,
            asset=asset1,
            signal_type=1,  # Long
            z=z,
            entry_zscore=entry_zscore,
            pair_info=pair_info,
            signal_expiry=signal_expiry
        ))
        
        signals.append(_create_signal(
            timestamp=timestamp,
            asset=asset2,
            signal_type=-1,  # Short
            z=z,
            entry_zscore=entry_zscore,
            pair_info=pair_info,
            signal_expiry=signal_expiry,
            strength_multiplier=hedge_ratio
        ))
    
    # Short the spread when z-score is very positive
    elif z > entry_zscore:
        # Short asset1, long asset2
        signals.append(_create_signal(
            timestamp=timestamp,
            asset=asset1,
            signal_type=-1,  # Short
            z=z,
            entry_zscore=entry_zscore,
            pair_info=pair_info,
            signal_expiry=signal_expiry
        ))
        
        signals.append(_create_signal(
            timestamp=timestamp,
            asset=asset2,
            signal_type=1,  # Long
            z=z,
            entry_zscore=entry_zscore,
            pair_info=pair_info,
            signal_expiry=signal_expiry,
            strength_multiplier=hedge_ratio
        ))
    
    # Exit signals when z-score is close to zero
    elif abs(z) < exit_zscore:
        # Generate exit signals for both assets
        for asset, hedge in [(asset1, 1.0), (asset2, hedge_ratio)]:
            signals.append(_create_signal(
                timestamp=timestamp,
                asset=asset,
                signal_type=0,  # Exit
                z=z,
                entry_zscore=entry_zscore,
                pair_info=pair_info,
                signal_expiry=signal_expiry
            ))
    
    return signals


# --------------------------------------------------------------------------------------
# Main API Functions
# --------------------------------------------------------------------------------------


def generate_signal(
    df: pd.DataFrame,
    asset_pairs: Optional[List[Tuple[str, str]]] = None,
    params: Optional[CointegrationParams] = None
) -> pd.DataFrame:
    """
    Generate trading signals based on cointegration analysis.
    
    Args:
        df: DataFrame with price data (columns are assets)
        asset_pairs: List of asset pairs to analyze, or None to test all pairs
        params: Parameters for testing and signal generation, or None to use defaults
        
    Returns:
        DataFrame with trading signals
    """
    # Use default parameters if none provided
    if params is None:
        params = CointegrationParams()
    
    # Get pairs to analyze
    if asset_pairs is None:
        # Find cointegrated pairs if not provided
        coint_results = find_cointegrated_pairs(df, params=params)
        asset_pairs = [(r.asset1, r.asset2) for r in coint_results]
    
    # If no pairs found, return empty signal DataFrame
    if not asset_pairs:
        logger.warning("No cointegrated pairs found or provided")
        return pd.DataFrame(columns=['timestamp', 'asset', 'signal', 'strength', 'metadata'])
    
    # Create signal DataFrame
    signals = []
    
    # For each pair, calculate spread and generate signals
    for asset1, asset2 in asset_pairs:
        # Skip if either asset is missing
        if asset1 not in df.columns or asset2 not in df.columns:
            logger.warning(f"Asset pair {asset1}-{asset2} not found in data")
            continue
            
        # Get price series
        price1 = df[asset1]
        price2 = df[asset2]
        
        # Calculate cointegration
        coint_result = test_cointegration(
            asset1_prices=price1, 
            asset2_prices=price2,
            params=params
        )
        
        # Skip if not cointegrated
        if not coint_result.is_cointegrated:
            continue
            
        # Get spread
        spread = coint_result.spread_series
        
        # Calculate z-score
        zscore = calculate_zscore(spread, lookback=params.lookback_window)
        
        # Generate signals for each timestamp
        for timestamp, z in zscore.items():
            # Skip if z-score is NaN (e.g., during initialization period)
            if np.isnan(z):
                continue
            
            # Generate signals for this pair at this timestamp
            pair_signals = _generate_pair_signals(
                timestamp=timestamp,
                z=z,
                asset1=asset1,
                asset2=asset2,
                hedge_ratio=coint_result.hedge_ratio,
                entry_zscore=params.entry_zscore,
                exit_zscore=params.exit_zscore,
                signal_expiry=params.signal_expiry
            )
            
            signals.extend(pair_signals)
    
    # Convert signals to DataFrame
    if signals:
        # Convert list of Signal objects to a dataframe
        signal_dicts = [s.__dict__ for s in signals]
        signal_df = pd.DataFrame(signal_dicts)
    else:
        # Create empty DataFrame with correct columns
        signal_df = pd.DataFrame(columns=['timestamp', 'asset', 'signal', 'strength', 'metadata'])
    
    return signal_df


def run_cointegration_analysis(
    price_df: pd.DataFrame,
    output_path: Optional[str] = None,
    params: Optional[CointegrationParams] = None
) -> Tuple[pd.DataFrame, List[CointegrationResult]]:
    """
    Run complete cointegration analysis and generate signals.
    
    Args:
        price_df: DataFrame with price data
        output_path: Path to write signals.parquet, or None to skip writing
        params: Parameters for testing and signal generation, or None to use defaults
        
    Returns:
        Tuple of (signal_df, cointegration_results)
    """
    # Use default parameters if none provided
    if params is None:
        params = CointegrationParams()
    
    # Find cointegrated pairs
    coint_results = find_cointegrated_pairs(price_df, params=params)
    
    # Generate signals
    signal_df = generate_signal(
        df=price_df,
        asset_pairs=[(r.asset1, r.asset2) for r in coint_results],
        params=params
    )
    
    # Write signals to parquet if path is provided
    if output_path and not signal_df.empty:
        signal_df.to_parquet(output_path)
        logger.info(f"Wrote {len(signal_df)} signals to {output_path}")
    
    return signal_df, coint_results