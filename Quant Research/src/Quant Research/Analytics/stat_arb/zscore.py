"""
Z-score calculation and signal generation for statistical arbitrage strategies.

This module provides advanced methods for calculating z-scores of spread series,
normalizing them across pairs, and generating trading signals based on statistically
significant deviations from equilibrium.
"""

# Standard library imports
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Local imports
from quant_research.core.models import Signal

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Data Structures
# --------------------------------------------------------------------------------------


@dataclass
class ZScoreParams:
    """Parameters for z-score calculation and signal generation."""
    # Z-score calculation parameters
    window: int = 60
    min_periods: Optional[int] = None
    method: str = 'rolling'  # Options: 'rolling', 'ewma', 'regime_adjusted'
    
    # Signal generation parameters
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    signal_expiry: int = 5
    signal_strength_cap: float = 3.0
    
    # Advanced options
    normalize_cross_sectional: bool = False
    filter_momentum: bool = False
    momentum_window: int = 5
    vol_window: int = 252
    regime_threshold: float = 1.5
    top_n_pairs: Optional[int] = None


@dataclass
class SpreadAnalysisResult:
    """Results from spread analysis."""
    spread: pd.Series
    zscore: pd.Series
    hedge_ratio: float
    half_life: float
    mean: float
    std: float
    current_zscore: float
    pct_extreme_pos: float  # % of time z > 2
    pct_extreme_neg: float  # % of time z < -2
    pct_neutral: float  # % of time -0.5 < z < 0.5


# --------------------------------------------------------------------------------------
# Z-Score Calculation Methods
# --------------------------------------------------------------------------------------


def calculate_rolling_zscore(
    series: pd.Series, 
    window: int = 60,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate z-score using rolling window.
    
    Args:
        series: Time series to calculate z-score for
        window: Rolling window size
        min_periods: Minimum number of observations required
        
    Returns:
        Series of z-scores
    """
    if min_periods is None:
        min_periods = window // 2
        
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    z_score = (series - rolling_mean) / rolling_std
    return z_score


def calculate_ewma_zscore(
    series: pd.Series,
    span: int = 60,
    min_periods: int = 30
) -> pd.Series:
    """
    Calculate z-score using exponentially weighted moving averages.
    
    Args:
        series: Time series to calculate z-score for
        span: Span parameter for ewm
        min_periods: Minimum number of observations
        
    Returns:
        Series of z-scores
    """
    ewma_mean = series.ewm(span=span, min_periods=min_periods).mean()
    ewma_std = series.ewm(span=span, min_periods=min_periods).std()
    
    # Avoid division by zero
    ewma_std = ewma_std.replace(0, np.nan)
    
    z_score = (series - ewma_mean) / ewma_std
    return z_score


def calculate_regime_adjusted_zscore(
    series: pd.Series,
    window: int = 60,
    vol_window: int = 252,
    regime_threshold: float = 1.5
) -> pd.Series:
    """
    Calculate regime-adjusted z-score, scaling by volatility of volatility.
    
    Args:
        series: Time series to calculate z-score for
        window: Rolling window for z-score calculation
        vol_window: Window for volatility regime detection
        regime_threshold: Threshold to identify high vol regimes
        
    Returns:
        Series of regime-adjusted z-scores
    """
    # Calculate regular z-score
    z_score = calculate_rolling_zscore(series, window)
    
    # Calculate volatility of volatility
    rolling_std = series.rolling(window=window).std()
    vol_of_vol = rolling_std.rolling(window=vol_window).std() / rolling_std.rolling(window=vol_window).mean()
    
    # Calculate regime scaling factor (higher vol -> lower scaling)
    scaling = 1.0 / np.maximum(1.0, vol_of_vol / vol_of_vol.rolling(window=vol_window).median())
    
    # Apply scaling
    adjusted_z = z_score * scaling
    
    return adjusted_z


def get_zscore_calculator(method: str):
    """
    Get the appropriate z-score calculation function based on method name.
    
    Args:
        method: Name of method ('rolling', 'ewma', 'regime_adjusted')
        
    Returns:
        Function that calculates z-scores
    """
    calculators = {
        'rolling': calculate_rolling_zscore,
        'ewma': calculate_ewma_zscore,
        'regime_adjusted': calculate_regime_adjusted_zscore
    }
    
    if method not in calculators:
        raise ValueError(f"Unknown z-score method: {method}")
    
    return calculators[method]


# --------------------------------------------------------------------------------------
# Z-Score Enhancement Methods
# --------------------------------------------------------------------------------------


def filter_zscore_by_momentum(
    z_score: pd.Series,
    momentum_window: int = 5
) -> pd.Series:
    """
    Filter z-score by its momentum (to avoid trading against strong trends).
    
    Args:
        z_score: Z-score series
        momentum_window: Window to calculate momentum
        
    Returns:
        Filtered z-score series
    """
    # Calculate momentum as rate of change
    momentum = z_score.diff(momentum_window)
    
    # Only keep z-scores where momentum is in the same direction
    # (e.g., negative z-score with negative momentum, or positive z-score with positive momentum)
    filtered_z = z_score.copy()
    opposite_momentum = (z_score * momentum < 0) & (abs(momentum) > 0.5)
    filtered_z[opposite_momentum] = np.nan
    
    return filtered_z


def normalize_zscores_cross_sectional(
    z_scores: Dict[str, pd.Series]
) -> Dict[str, pd.Series]:
    """
    Normalize z-scores across multiple pairs/instruments.
    
    Args:
        z_scores: Dictionary of z-score series by pair name
        
    Returns:
        Dictionary of normalized z-score series
    """
    if not z_scores:
        return {}
        
    # Combine all z-scores into a DataFrame
    df = pd.DataFrame(z_scores)
    
    # Calculate cross-sectional mean and std for each timestamp
    cs_mean = df.mean(axis=1)
    cs_std = df.std(axis=1)
    
    # Normalize z-scores
    normalized = {}
    for name, z in z_scores.items():
        normalized_z = (z - cs_mean) / cs_std.replace(0, 1.0)  # Avoid division by zero
        normalized[name] = normalized_z
        
    return normalized


def rank_zscores(
    z_scores: Dict[str, pd.Series],
    ascending: bool = True,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Rank z-scores across multiple pairs at each timestamp.
    
    Args:
        z_scores: Dictionary of z-score series by pair name
        ascending: If True, ranks from most negative to most positive
        top_n: Number of top/bottom pairs to keep, or None for all
        
    Returns:
        DataFrame with pairs ranked by z-score at each timestamp
    """
    if not z_scores:
        return pd.DataFrame()
        
    # Combine z-scores into a DataFrame
    df = pd.DataFrame(z_scores)
    
    # Rank z-scores (smaller z-scores get lower ranks if ascending)
    ranked = df.rank(axis=1, ascending=ascending)
    
    # Filter to top N pairs if specified
    if top_n is not None and not df.empty:
        # For each timestamp, keep only top N pairs
        filtered = pd.DataFrame(index=ranked.index)
        
        for idx, row in ranked.iterrows():
            top_pairs = row.nsmallest(top_n).index
            filtered.loc[idx, top_pairs] = df.loc[idx, top_pairs]
            
        return filtered
    
    return df


# --------------------------------------------------------------------------------------
# Signal Generation Functions
# --------------------------------------------------------------------------------------


def generate_signals_from_zscore(
    z_score: pd.Series,
    asset: str,
    params: Optional[ZScoreParams] = None
) -> List[Signal]:
    """
    Generate trading signals based on z-score thresholds.
    
    Args:
        z_score: Z-score time series
        asset: Asset identifier
        params: Parameters for signal generation, or None to use defaults
        
    Returns:
        List of Signal objects
    """
    # Use default parameters if none provided
    if params is None:
        params = ZScoreParams()
    
    signals = []
    
    # Apply momentum filter if requested
    if params.filter_momentum:
        z_score = filter_zscore_by_momentum(z_score, params.momentum_window)
    
    # Generate signals for each timestamp
    for timestamp, z in z_score.items():
        # Skip NaN values
        if pd.isna(z):
            continue
            
        # Long signal when z-score is negative (price below mean)
        if z < -params.entry_threshold:
            signal = Signal(
                timestamp=timestamp,
                asset=asset,
                signal=1,  # Long
                strength=min(abs(z) / params.entry_threshold, params.signal_strength_cap),
                metadata={
                    'zscore': z,
                    'strategy': f"zscore_{params.method}",
                    'expiry': params.signal_expiry
                }
            )
            signals.append(signal)
            
        # Short signal when z-score is positive (price above mean)
        elif z > params.entry_threshold:
            signal = Signal(
                timestamp=timestamp,
                asset=asset,
                signal=-1,  # Short
                strength=min(abs(z) / params.entry_threshold, params.signal_strength_cap),
                metadata={
                    'zscore': z,
                    'strategy': f"zscore_{params.method}",
                    'expiry': params.signal_expiry
                }
            )
            signals.append(signal)
            
        # Exit signal when z-score is close to zero
        elif abs(z) < params.exit_threshold:
            signal = Signal(
                timestamp=timestamp,
                asset=asset,
                signal=0,  # Exit
                strength=1.0 - (abs(z) / params.exit_threshold),
                metadata={
                    'zscore': z,
                    'strategy': f"zscore_{params.method}",
                    'expiry': params.signal_expiry
                }
            )
            signals.append(signal)
    
    return signals


def generate_signals_pair_trading(
    z_score: pd.Series,
    asset1: str,
    asset2: str,
    hedge_ratio: float,
    params: Optional[ZScoreParams] = None
) -> List[Signal]:
    """
    Generate pair trading signals based on z-score.
    
    Args:
        z_score: Z-score of the spread
        asset1: First asset (the one being bought/sold directly)
        asset2: Second asset (the hedge)
        hedge_ratio: Ratio for hedging positions
        params: Parameters for signal generation, or None to use defaults
        
    Returns:
        List of Signal objects for both assets
    """
    # Use default parameters if none provided
    if params is None:
        params = ZScoreParams()
    
    signals = []
    
    # Generate signals for each timestamp
    for timestamp, z in z_score.items():
        # Skip NaN values
        if pd.isna(z):
            continue
            
        # When z-score is negative: buy asset1, sell asset2
        if z < -params.entry_threshold:
            # Signal for asset1 (long)
            signals.append(Signal(
                timestamp=timestamp,
                asset=asset1,
                signal=1,  # Long
                strength=min(abs(z) / params.entry_threshold, params.signal_strength_cap),
                metadata={
                    'pair': f"{asset1}-{asset2}",
                    'zscore': z,
                    'hedge_ratio': hedge_ratio,
                    'strategy': f"zscore_{params.method}",
                    'expiry': params.signal_expiry
                }
            ))
            
            # Signal for asset2 (short)
            signals.append(Signal(
                timestamp=timestamp,
                asset=asset2,
                signal=-1,  # Short
                strength=min(abs(z) / params.entry_threshold * hedge_ratio, params.signal_strength_cap),
                metadata={
                    'pair': f"{asset1}-{asset2}",
                    'zscore': z,
                    'hedge_ratio': hedge_ratio,
                    'strategy': f"zscore_{params.method}",
                    'expiry': params.signal_expiry
                }
            ))
            
        # When z-score is positive: sell asset1, buy asset2
        elif z > params.entry_threshold:
            # Signal for asset1 (short)
            signals.append(Signal(
                timestamp=timestamp,
                asset=asset1,
                signal=-1,  # Short
                strength=min(abs(z) / params.entry_threshold, params.signal_strength_cap),
                metadata={
                    'pair': f"{asset1}-{asset2}",
                    'zscore': z,
                    'hedge_ratio': hedge_ratio,
                    'strategy': f"zscore_{params.method}",
                    'expiry': params.signal_expiry
                }
            ))
            
            # Signal for asset2 (long)
            signals.append(Signal(
                timestamp=timestamp,
                asset=asset2,
                signal=1,  # Long
                strength=min(abs(z) / params.entry_threshold * hedge_ratio, params.signal_strength_cap),
                metadata={
                    'pair': f"{asset1}-{asset2}",
                    'zscore': z,
                    'hedge_ratio': hedge_ratio,
                    'strategy': f"zscore_{params.method}",
                    'expiry': params.signal_expiry
                }
            ))
            
        # When z-score is close to zero: exit positions
        elif abs(z) < params.exit_threshold:
            # Exit signal for both assets
            for asset, hr in [(asset1, 1.0), (asset2, hedge_ratio)]:
                signals.append(Signal(
                    timestamp=timestamp,
                    asset=asset,
                    signal=0,  # Exit
                    strength=1.0 - (abs(z) / params.exit_threshold),
                    metadata={
                        'pair': f"{asset1}-{asset2}",
                        'zscore': z,
                        'hedge_ratio': hedge_ratio,
                        'strategy': f"zscore_{params.method}",
                        'expiry': params.signal_expiry
                    }
                ))
    
    return signals


# --------------------------------------------------------------------------------------
# Main API Functions
# --------------------------------------------------------------------------------------


def generate_signal(
    df: pd.DataFrame,
    pairs: List[Tuple[str, str, float]] = None,  # (asset1, asset2, hedge_ratio)
    params: Optional[ZScoreParams] = None
) -> pd.DataFrame:
    """
    Generate trading signals based on z-score analysis.
    
    Args:
        df: DataFrame with price data (columns are assets)
        pairs: List of tuples with (asset1, asset2, hedge_ratio)
        params: Parameters for z-score calculation and signal generation, or None to use defaults
        
    Returns:
        DataFrame with trading signals
    """
    # Use default parameters if none provided
    if params is None:
        params = ZScoreParams()
    
    all_signals = []
    
    # If no pairs provided, return empty DataFrame
    if not pairs:
        logger.warning("No pairs provided for z-score analysis")
        return pd.DataFrame(columns=['timestamp', 'asset', 'signal', 'strength', 'metadata'])
    
    # Calculate z-scores for each pair
    z_scores = {}
    for asset1, asset2, hedge_ratio in pairs:
        # Calculate spread
        if asset1 not in df.columns or asset2 not in df.columns:
            logger.warning(f"Skipping pair {asset1}-{asset2} - assets not found in data")
            continue
            
        spread = df[asset1] - hedge_ratio * df[asset2]
        
        # Get appropriate z-score calculator
        calculator = get_zscore_calculator(params.method)
        
        # Calculate z-score
        min_periods = params.min_periods if params.min_periods else params.window // 2
        
        if params.method == 'rolling':
            z_score = calculator(spread, window=params.window, min_periods=min_periods)
        elif params.method == 'ewma':
            z_score = calculator(spread, span=params.window, min_periods=min_periods)
        elif params.method == 'regime_adjusted':
            z_score = calculator(
                spread,
                window=params.window,
                vol_window=params.vol_window,
                regime_threshold=params.regime_threshold
            )
            
        z_scores[f"{asset1}-{asset2}"] = z_score
    
    # If no z-scores calculated, return empty DataFrame
    if not z_scores:
        logger.warning("No valid pairs found for z-score calculation")
        return pd.DataFrame(columns=['timestamp', 'asset', 'signal', 'strength', 'metadata'])
    
    # Normalize z-scores across pairs if requested
    if params.normalize_cross_sectional and len(z_scores) > 1:
        z_scores = normalize_zscores_cross_sectional(z_scores)
    
    # Filter to top N pairs by z-score magnitude if requested
    if params.top_n_pairs is not None and len(z_scores) > params.top_n_pairs:
        # Create ranking of pairs at each timestamp
        ranked = rank_zscores(
            {k: z_scores[k].abs() for k in z_scores},  # Rank by absolute z-score
            ascending=False,  # Highest values first
            top_n=params.top_n_pairs
        )
        
        # Filter z_scores to only include top pairs at each timestamp
        filtered_z_scores = {}
        for pair, zseries in z_scores.items():
            filtered = zseries.copy()
            # Set z-score to NaN when pair is not in top N at that timestamp
            valid_timestamps = ranked.index[~ranked[pair].isna()] if pair in ranked else []
            filtered[~filtered.index.isin(valid_timestamps)] = np.nan
            filtered_z_scores[pair] = filtered
            
        z_scores = filtered_z_scores
    
    # Generate signals for each pair
    for idx, (asset1, asset2, hedge_ratio) in enumerate(pairs):
        pair_key = f"{asset1}-{asset2}"
        if pair_key not in z_scores:
            continue
            
        z_score = z_scores[pair_key]
        
        # Filter by momentum if requested
        if params.filter_momentum:
            z_score = filter_zscore_by_momentum(
                z_score, 
                momentum_window=params.momentum_window
            )
        
        # Generate signals
        pair_signals = generate_signals_pair_trading(
            z_score=z_score,
            asset1=asset1,
            asset2=asset2,
            hedge_ratio=hedge_ratio,
            params=params
        )
        
        all_signals.extend(pair_signals)
    
    # Convert list of signals to DataFrame
    if all_signals:
        signal_dicts = [s.__dict__ for s in all_signals]
        signal_df = pd.DataFrame(signal_dicts)
    else:
        # Create empty DataFrame with correct columns
        signal_df = pd.DataFrame(columns=['timestamp', 'asset', 'signal', 'strength', 'metadata'])
    
    return signal_df


def analyze_spread(
    df: pd.DataFrame,
    asset1: str,
    asset2: str,
    hedge_ratio: Optional[float] = None,
    params: Optional[ZScoreParams] = None
) -> SpreadAnalysisResult:
    """
    Analyze a spread between two assets, calculating key metrics.
    
    Args:
        df: DataFrame with price data
        asset1: First asset in pair
        asset2: Second asset in pair
        hedge_ratio: Fixed hedge ratio or None to estimate it
        params: Parameters for z-score calculation, or None to use defaults
        
    Returns:
        SpreadAnalysisResult with spread metrics
    """
    # Use default parameters if none provided
    if params is None:
        params = ZScoreParams()
    
    # Calculate or use provided hedge ratio
    lookback_window = params.window * 2  # Use longer lookback for stability
    
    if hedge_ratio is None:
        # Use recent data to estimate hedge ratio
        recent_data = df.iloc[-lookback_window:].dropna(subset=[asset1, asset2])
        model = sm.OLS(recent_data[asset1], sm.add_constant(recent_data[asset2])).fit()
        hedge_ratio = model.params[1]
    
    # Calculate spread
    spread = df[asset1] - hedge_ratio * df[asset2]
    
    # Calculate z-score
    calculator = get_zscore_calculator(params.method)
    
    if params.method == 'rolling':
        z_score = calculator(spread, window=params.window)
    elif params.method == 'ewma':
        z_score = calculator(spread, span=params.window)
    elif params.method == 'regime_adjusted':
        z_score = calculator(
            spread,
            window=params.window,
            vol_window=params.vol_window,
            regime_threshold=params.regime_threshold
        )
    
    # Calculate half-life of mean reversion
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    
    # Remove NAs
    valid_idx = ~(spread_lag.isna() | spread_diff.isna())
    X = sm.add_constant(spread_lag[valid_idx])
    y = spread_diff[valid_idx]
    
    # Run regression
    model = sm.OLS(y, X).fit()
    beta = model.params[1]
    
    # Calculate half-life
    if beta >= 0:
        half_life = np.inf  # Not mean-reverting
    else:
        half_life = -np.log(2) / beta
    
    # Calculate summary statistics
    mean = spread.mean()
    std = spread.std()
    current_zscore = z_score.iloc[-1] if not z_score.empty else np.nan
    
    # Calculate time in different z-score zones
    extreme_pos = (z_score > 2).mean() * 100  # % of time z > 2
    extreme_neg = (z_score < -2).mean() * 100  # % of time z < -2
    neutral = ((z_score > -0.5) & (z_score < 0.5)).mean() * 100  # % of time -0.5 < z < 0.5
    
    return SpreadAnalysisResult(
        spread=spread,
        zscore=z_score,
        hedge_ratio=hedge_ratio,
        half_life=half_life,
        mean=mean,
        std=std,
        current_zscore=current_zscore,
        pct_extreme_pos=extreme_pos,
        pct_extreme_neg=extreme_neg,
        pct_neutral=neutral
    )