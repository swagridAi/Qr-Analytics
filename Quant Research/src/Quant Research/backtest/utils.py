"""
Utility functions for signal processing in backtesting strategies.

This module provides common utilities for filtering signals, validating data,
normalizing positions, and other shared functionality across different strategies.
"""

import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, TypeVar, Protocol, NamedTuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)

# Type definitions
DataFrame = TypeVar('DataFrame', bound=pd.DataFrame)

class DataFrameValidator(Protocol):
    """Protocol for DataFrame validation functions."""
    def __call__(self, df: DataFrame, **kwargs) -> Tuple[bool, List[str]]: ...


# ===============================================================================
# Configuration Classes
# ===============================================================================

@dataclass
class ValidationConfig:
    """Configuration for DataFrame validation."""
    required_columns: List[str]
    caller_name: str = "unknown"


@dataclass
class FilterConfig:
    """Configuration for signal filtering."""
    strategy_name: str
    required_columns: Optional[List[str]] = None
    default_values: Optional[Dict[str, Any]] = None
    caller_name: str = "unknown"


@dataclass
class PriceConfig:
    """Configuration for price data preparation."""
    required_columns: Optional[List[str]] = None
    derived_columns: Optional[Dict[str, Optional[str]]] = None
    default_values: Optional[Dict[str, Any]] = None
    caller_name: str = "unknown"


@dataclass
class VolatilityConfig:
    """Configuration for volatility calculation."""
    lookback: int = 20
    min_periods: int = 5
    asset_column: str = 'asset_id'
    time_column: str = 'timestamp'
    price_column: str = 'close'
    vol_column: str = 'volatility'
    annualization_factor: int = 252


class ValidationResult(NamedTuple):
    """Result of a validation operation."""
    valid: bool
    missing_columns: List[str]
    df: DataFrame


# ===============================================================================
# Decorators
# ===============================================================================

def log_operation(func):
    """Decorator to log function entry and exit."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        caller = kwargs.get('caller_name', 'unknown')
        logger.debug(f"Starting {func.__name__} operation from {caller}")
        result = func(*args, **kwargs)
        logger.debug(f"Completed {func.__name__} operation")
        return result
    return wrapper


def validate_input_df(func):
    """Decorator to validate input DataFrame before processing."""
    @wraps(func)
    def wrapper(df: DataFrame, *args, **kwargs):
        if df is None or df.empty:
            caller = kwargs.get('caller_name', 'unknown')
            logger.warning(f"Empty DataFrame provided to {func.__name__} from {caller}")
            # Get column names from function docstring or default
            if hasattr(func, '__annotations__') and 'return' in func.__annotations__:
                ret_type = func.__annotations__['return']
                # If returning a DataFrame, return empty with expected columns
                if getattr(ret_type, '__origin__', None) in (DataFrame, pd.DataFrame):
                    return pd.DataFrame()
            return pd.DataFrame()
        return func(df, *args, **kwargs)
    return wrapper


# ===============================================================================
# Core Validation Functions
# ===============================================================================

def validate_dataframe(
    df: DataFrame, 
    required_columns: List[str],
    caller_name: str = "unknown"
) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        caller_name: Name of the caller for logging purposes
        
    Returns:
        (is_valid, missing_columns): Tuple of validation result and list of missing columns
    """
    if df is None or df.empty:
        logger.warning(f"Empty DataFrame passed to {caller_name}")
        return False, []
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing_columns) == 0
    
    if not is_valid:
        logger.warning(f"DataFrame missing required columns for {caller_name}: {missing_columns}")
    
    return is_valid, missing_columns


def ensure_columns(
    df: DataFrame, 
    default_values: Dict[str, Any],
    caller_name: str = "unknown"
) -> DataFrame:
    """
    Ensure a DataFrame has all specified columns, filling in defaults if missing.
    
    Args:
        df: DataFrame to process
        default_values: Dict mapping column names to default values
        caller_name: Name of the caller for logging purposes
        
    Returns:
        DataFrame with all required columns
    """
    if df is None or df.empty:
        logger.warning(f"Empty DataFrame passed to {caller_name}")
        return df
    
    result = df.copy()
    
    for col, default_value in default_values.items():
        if col not in result.columns:
            if default_value is None and 'close' in result.columns:
                # Special case for bid/ask
                if col == 'bid':
                    result[col] = result['close'] * 0.9999
                    logger.info(f"Created '{col}' column from 'close' in {caller_name}")
                elif col == 'ask':
                    result[col] = result['close'] * 1.0001
                    logger.info(f"Created '{col}' column from 'close' in {caller_name}")
                else:
                    result[col] = result['close']
                    logger.info(f"Created '{col}' column from 'close' in {caller_name}")
            else:
                result[col] = default_value
                logger.info(f"Added missing '{col}' column with default value in {caller_name}")
    
    return result


def convert_datetime_columns(df: DataFrame, datetime_columns: List[str] = ['timestamp']) -> DataFrame:
    """
    Convert specified columns to datetime type.
    
    Args:
        df: DataFrame to process
        datetime_columns: List of column names to convert to datetime
        
    Returns:
        DataFrame with converted datetime columns
    """
    if df is None or df.empty:
        return df
    
    result = df.copy()
    
    for col in datetime_columns:
        if col in result.columns and not pd.api.types.is_datetime64_any_dtype(result[col]):
            result[col] = pd.to_datetime(result[col])
            logger.debug(f"Converted {col} column to datetime")
    
    return result


def sort_by_time(df: DataFrame, time_column: str = 'timestamp') -> DataFrame:
    """
    Sort DataFrame by timestamp column.
    
    Args:
        df: DataFrame to sort
        time_column: Column name containing timestamps
        
    Returns:
        Sorted DataFrame
    """
    if df is None or df.empty or time_column not in df.columns:
        return df
    
    return df.sort_values(time_column)


# ===============================================================================
# Signal Processing Functions
# ===============================================================================

@validate_input_df
@log_operation
def filter_signals_by_strategy(
    signals_df: DataFrame, 
    config: Optional[FilterConfig] = None,
    **kwargs
) -> DataFrame:
    """
    Filter signals DataFrame for a specific strategy with validation.
    
    Args:
        signals_df: DataFrame with all signals
        config: FilterConfig object or None
        **kwargs: Alternative way to specify config parameters
        
    Returns:
        DataFrame with filtered signals for the specified strategy
    
    Example:
        >>> momentum_signals = filter_signals_by_strategy(
        >>>     signals_df, 
        >>>     FilterConfig(
        >>>         strategy_name='momentum',
        >>>         required_columns=['direction', 'strength'],
        >>>         default_values={'direction': 0.0, 'strength': 0.5},
        >>>         caller_name=self.name
        >>>     )
        >>> )
    """
    # Process configuration
    if config is None:
        config = FilterConfig(
            strategy_name=kwargs.get('strategy_name', kwargs.get('strategy', '')),
            required_columns=kwargs.get('required_columns'),
            default_values=kwargs.get('default_values'),
            caller_name=kwargs.get('caller_name', 'unknown')
        )
    
    # Basic required columns
    basic_columns = ['timestamp', 'asset_id', 'strategy']
    
    # Add strategy-specific required columns
    all_required = basic_columns.copy()
    if config.required_columns:
        all_required.extend(config.required_columns)
    
    # Validate input
    is_valid, missing_cols = validate_dataframe(
        signals_df, 
        all_required, 
        config.caller_name
    )
    
    if not is_valid:
        logger.warning(f"Invalid signals DataFrame: missing columns {missing_cols}")
        empty_df = pd.DataFrame(columns=all_required)
        return empty_df
    
    # Filter for specific strategy
    filtered_signals = signals_df[signals_df['strategy'] == config.strategy_name].copy()
    
    if len(filtered_signals) == 0:
        logger.warning(f"No {config.strategy_name} signals found in input data")
        empty_df = pd.DataFrame(columns=all_required)
        return empty_df
    
    # Add default values for missing columns
    if config.default_values:
        filtered_signals = ensure_columns(filtered_signals, config.default_values, config.caller_name)
    
    return filtered_signals


@validate_input_df
@log_operation
def prepare_price_data(
    prices_df: DataFrame,
    config: Optional[PriceConfig] = None,
    **kwargs
) -> DataFrame:
    """
    Validate and prepare price data for strategy use.
    
    Args:
        prices_df: DataFrame with price data
        config: PriceConfig object or None
        **kwargs: Alternative way to specify config parameters
        
    Returns:
        Validated and prepared price DataFrame
    
    Example:
        >>> prepared_prices = prepare_price_data(
        >>>     prices_df,
        >>>     PriceConfig(
        >>>         required_columns=['volume'],
        >>>         derived_columns={'bid': 'close', 'ask': 'close'},
        >>>         default_values={'volume': 100000},
        >>>         caller_name=self.name
        >>>     )
        >>> )
    """
    # Process configuration
    if config is None:
        config = PriceConfig(
            required_columns=kwargs.get('required_columns'),
            derived_columns=kwargs.get('derived_columns'),
            default_values=kwargs.get('default_values'),
            caller_name=kwargs.get('caller_name', 'unknown')
        )
    
    # Basic required columns
    basic_columns = ['timestamp', 'asset_id', 'close']
    
    # Add strategy-specific required columns
    all_required = basic_columns.copy()
    if config.required_columns:
        all_required.extend(config.required_columns)
    
    # Validate input
    is_valid, missing_cols = validate_dataframe(
        prices_df, 
        all_required, 
        config.caller_name
    )
    
    if not is_valid:
        # Try to recover if 'close' is missing but derived columns source exists
        recoverable = False
        if 'close' in missing_cols and config.derived_columns and 'close' in config.derived_columns:
            source_col = config.derived_columns['close']
            if source_col in prices_df.columns:
                prices_df = prices_df.copy()
                prices_df['close'] = prices_df[source_col]
                missing_cols.remove('close')
                recoverable = True
                logger.info(f"Derived 'close' column from '{source_col}'")
        
        if not recoverable:
            logger.warning(f"Invalid prices DataFrame: missing columns {missing_cols}")
            empty_df = pd.DataFrame(columns=all_required)
            return empty_df
    
    # Make a copy to avoid modifying the original
    result = prices_df.copy()
    
    # Apply defaults and derive columns
    if config.default_values:
        result = ensure_columns(result, config.default_values, config.caller_name)
    
    if config.derived_columns:
        for target_col, source_col in config.derived_columns.items():
            if target_col not in result.columns and source_col and source_col in result.columns:
                if target_col == 'bid' and source_col == 'close':
                    result[target_col] = result[source_col] * 0.9999
                    logger.info(f"Created 'bid' column from 'close' with -0.01% adjustment")
                elif target_col == 'ask' and source_col == 'close':
                    result[target_col] = result[source_col] * 1.0001
                    logger.info(f"Created 'ask' column from 'close' with +0.01% adjustment")
                else:
                    result[target_col] = result[source_col]
                    logger.info(f"Created '{target_col}' column from '{source_col}'")
    
    # Calculate common derived fields
    if all(col in result.columns for col in ['bid', 'ask']) and 'mid_price' not in result.columns:
        result['mid_price'] = (result['bid'] + result['ask']) / 2
        logger.info("Calculated 'mid_price' as average of bid and ask")
    
    if all(col in result.columns for col in ['bid', 'ask']) and 'spread' not in result.columns:
        result['spread'] = result['ask'] - result['bid']
        logger.info("Calculated 'spread' as ask minus bid")
    
    # Convert timestamp to datetime
    result = convert_datetime_columns(result)
    
    # Sort by timestamp
    result = sort_by_time(result)
    
    return result


@validate_input_df
@log_operation
def standardize_positions(
    positions_df: DataFrame,
    position_column: str = 'position',
    weight_column: str = 'target_weight',
    add_missing_columns: bool = True,
    caller_name: str = "unknown",
    standard_columns: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Standardize position DataFrame format across strategies.
    
    Args:
        positions_df: DataFrame with positions
        position_column: Column containing position sizes
        weight_column: Column to store target weight
        add_missing_columns: Whether to add missing standard columns with defaults
        caller_name: Name of the caller for logging purposes
        standard_columns: Optional dictionary of standard columns to add
        
    Returns:
        Standardized positions DataFrame
    """
    # Check required columns
    required_columns = ['timestamp', 'asset_id', position_column]
    is_valid, missing_cols = validate_dataframe(
        positions_df,
        required_columns,
        caller_name
    )
    
    if not is_valid:
        logger.warning(f"Invalid positions DataFrame: missing columns {missing_cols}")
        return pd.DataFrame(columns=['timestamp', 'asset_id', 'position', 'target_weight'])
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Rename position column if needed
    if position_column != 'position' and position_column in result.columns:
        result['position'] = result[position_column]
    
    # Add target_weight if missing
    if weight_column not in result.columns:
        # Default weight calculation
        result[weight_column] = result['position']
        logger.info(f"Added '{weight_column}' column based on position")
    
    # Ensure standard columns
    if add_missing_columns:
        columns_to_add = {
            'exchange_id': 'default',
            'strategy': caller_name
        }
        
        if standard_columns:
            columns_to_add.update(standard_columns)
            
        result = ensure_columns(result, columns_to_add, caller_name)
    
    # Ensure position and weight are numeric
    for col in ['position', weight_column]:
        if col in result.columns and not pd.api.types.is_numeric_dtype(result[col]):
            try:
                result[col] = pd.to_numeric(result[col])
                logger.info(f"Converted '{col}' column to numeric")
            except:
                logger.warning(f"Could not convert '{col}' column to numeric")
    
    # Convert timestamp to datetime
    result = convert_datetime_columns(result)
    
    return result


@validate_input_df
@log_operation
def calculate_volatility(
    prices_df: DataFrame,
    config: Optional[VolatilityConfig] = None,
    **kwargs
) -> DataFrame:
    """
    Calculate rolling volatility for assets in a price DataFrame.
    
    Args:
        prices_df: DataFrame with price data
        config: VolatilityConfig object or None
        **kwargs: Alternative way to specify config parameters
        
    Returns:
        DataFrame with volatility values added
    """
    # Process configuration
    if config is None:
        config = VolatilityConfig(
            lookback=kwargs.get('lookback', 20),
            min_periods=kwargs.get('min_periods', 5),
            asset_column=kwargs.get('asset_column', 'asset_id'),
            time_column=kwargs.get('time_column', 'timestamp'),
            price_column=kwargs.get('price_column', 'close'),
            vol_column=kwargs.get('vol_column', 'volatility'),
            annualization_factor=kwargs.get('annualization_factor', 252)
        )
    
    # Check required columns
    required_columns = [config.time_column, config.asset_column, config.price_column]
    is_valid, missing_cols = validate_dataframe(
        prices_df,
        required_columns,
        "calculate_volatility"
    )
    
    if not is_valid:
        logger.warning(f"Invalid prices DataFrame: missing columns {missing_cols}")
        return prices_df
    
    # Convert timestamp to datetime
    result = convert_datetime_columns(prices_df, [config.time_column])
    
    # Calculate rolling volatility by asset
    vol_values = []
    
    for asset_id, asset_data in result.groupby(config.asset_column):
        # Sort by time
        asset_data = sort_by_time(asset_data, config.time_column)
        
        # Calculate returns
        asset_data['return'] = asset_data[config.price_column].pct_change()
        
        # Calculate rolling volatility
        asset_data[config.vol_column] = asset_data['return'].rolling(
            window=config.lookback, min_periods=config.min_periods
        ).std() * np.sqrt(config.annualization_factor)
        
        # Drop the temporary returns column
        asset_data = asset_data.drop('return', axis=1)
        
        # Add to results
        vol_values.append(asset_data)
    
    if not vol_values:
        return result
    
    # Combine results
    result = pd.concat(vol_values, ignore_index=True)
    
    # Fill missing volatility values with mean
    if config.vol_column in result.columns:
        mean_vol = result[config.vol_column].mean()
        result[config.vol_column] = result[config.vol_column].fillna(mean_vol)
    
    return result


@validate_input_df
@log_operation
def identify_regimes(
    signals_df: DataFrame,
    regime_thresholds: Dict[str, float] = None,
    columns: Dict[str, str] = None
) -> DataFrame:
    """
    Identify market regimes from signal probabilities.
    
    Args:
        signals_df: DataFrame with regime signals
        regime_thresholds: Dictionary mapping regime names to threshold probabilities
        columns: Dictionary mapping column roles to actual column names
        
    Returns:
        DataFrame with identified regimes
    """
    # Default column mappings
    default_columns = {
        'time': 'timestamp',
        'asset': 'asset_id',
        'strategy': 'strategy',
        'regime': 'regime',
        'probability': 'probability'
    }
    
    # Use provided columns or defaults
    col_map = default_columns.copy()
    if columns:
        col_map.update(columns)
    
    # Default thresholds if not provided
    if regime_thresholds is None:
        regime_thresholds = {
            'trending': 0.6,
            'mean_reverting': 0.6,
            'high_volatility': 0.7,
            'low_volatility': 0.7,
            'normal': 0.5
        }
    
    # Check required columns
    required_columns = list(col_map.values())
    is_valid, missing_cols = validate_dataframe(
        signals_df,
        required_columns,
        "identify_regimes"
    )
    
    if not is_valid:
        logger.warning(f"Invalid signals DataFrame: missing columns {missing_cols}")
        return pd.DataFrame()
    
    # Filter for regime-related strategies
    regime_signals = signals_df[
        (signals_df[col_map['strategy']] == 'regime_hmm') | 
        (signals_df[col_map['strategy']] == 'regime_cp')
    ].copy()
    
    if len(regime_signals) == 0:
        logger.warning("No regime signals found in input data")
        return pd.DataFrame()
    
    # Initialize results list
    regime_classifications = []
    
    # Process signals by timestamp and asset
    for (timestamp, asset_id), group in regime_signals.groupby([col_map['time'], col_map['asset']]):
        # Extract regime probabilities
        regime_probs = {}
        
        for _, row in group.iterrows():
            regime = row[col_map['regime']]
            if regime in regime_probs:
                # Use higher probability if regime detected by multiple methods
                regime_probs[regime] = max(regime_probs[regime], row[col_map['probability']])
            else:
                regime_probs[regime] = row[col_map['probability']]
        
        # Determine dominant regime
        if regime_probs:
            dominant_regime, dominant_prob = _determine_dominant_regime(regime_probs, regime_thresholds)
            
            # Add to results
            regime_classifications.append({
                col_map['time']: timestamp,
                col_map['asset']: asset_id,
                col_map['regime']: dominant_regime,
                col_map['probability']: dominant_prob,
                'all_probs': regime_probs  # Store all probabilities for reference
            })
    
    # Convert to DataFrame
    regime_df = pd.DataFrame(regime_classifications)
    
    if len(regime_df) == 0:
        logger.warning("No regime classifications generated")
        return pd.DataFrame()
    
    # Get regime distribution stats
    if col_map['regime'] in regime_df.columns:
        regime_distribution = regime_df.groupby(col_map['regime']).size()
        logger.info(f"Identified regimes: {regime_distribution.to_dict()}")
    
    return regime_df


# ===============================================================================
# Helper Functions
# ===============================================================================

def _determine_dominant_regime(
    regime_probs: Dict[str, float], 
    regime_thresholds: Dict[str, float]
) -> Tuple[str, float]:
    """
    Determine the dominant regime from probability dictionary.
    
    Args:
        regime_probs: Dictionary of regime probabilities
        regime_thresholds: Dictionary of threshold values for each regime
        
    Returns:
        Tuple of (dominant_regime, probability)
    """
    # Find regime with highest probability exceeding its threshold
    dominant_regime = None
    max_prob_diff = -float('inf')
    
    for regime, prob in regime_probs.items():
        threshold = regime_thresholds.get(regime, 0.5)
        prob_diff = prob - threshold
        
        if prob_diff > 0 and prob_diff > max_prob_diff:
            dominant_regime = regime
            max_prob_diff = prob_diff
    
    # If no regime exceeds threshold, default to 'normal'
    if dominant_regime is None:
        dominant_regime = 'normal'
        dominant_prob = regime_probs.get('normal', 0.5)
    else:
        dominant_prob = regime_probs[dominant_regime]
    
    return dominant_regime, dominant_prob


def calculate_metrics(
    positions_df: DataFrame,
    prices_df: DataFrame,
    metrics: List[str] = None,
    column_map: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Calculate various metrics for positions execution.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with prices
        metrics: List of metrics to calculate (defaults to all)
        column_map: Mapping of standard column names to actual column names
        
    Returns:
        Dictionary with calculated metrics
    """
    # Default metrics to calculate
    all_metrics = ['vwap', 'slippage', 'execution_count', 'turnover', 'implementation_shortfall']
    metrics_to_calculate = metrics or all_metrics
    
    # Default column mappings
    default_columns = {
        'time': 'timestamp',
        'asset': 'asset_id',
        'position': 'position',
        'price': 'close'
    }
    
    # Use provided mapping or defaults
    cols = default_columns.copy()
    if column_map:
        cols.update(column_map)
    
    # Ensure DataFrames have required columns
    for df, required, name in [(positions_df, [cols['time'], cols['asset'], cols['position']], 'positions'),
                              (prices_df, [cols['time'], cols['asset'], cols['price']], 'prices')]:
        is_valid, missing = validate_dataframe(df, required, "calculate_metrics")
        if not is_valid:
            logger.warning(f"Invalid {name} DataFrame for metrics calculation")
            return {}
    
    # Empty base dictionary with zeros for all requested metrics
    result = {metric: 0.0 for metric in metrics_to_calculate}
    result['valid'] = False
    
    try:
        # Ensure timestamp is datetime
        positions = convert_datetime_columns(positions_df, [cols['time']])
        prices = convert_datetime_columns(prices_df, [cols['time']])
        
        # Merge positions with prices
        merged = pd.merge(
            positions,
            prices[[cols['time'], cols['asset'], cols['price']]],
            on=[cols['time'], cols['asset']],
            how='left'
        )
        
        if merged.empty:
            logger.warning("No matching data after merging positions and prices")
            return result
        
        # Filter non-zero positions (actual trades)
        trades = merged[merged[cols['position']] != 0].copy()
        
        if trades.empty:
            logger.warning("No trades found for metrics calculation")
            return result
        
        # Mark result as valid since we have data
        result['valid'] = True
        
        # Calculate notional value
        trades['notional'] = trades[cols['position']].abs() * trades[cols['price']]
        result['total_notional'] = float(trades['notional'].sum())
        
        # Calculate VWAP
        if 'vwap' in metrics_to_calculate and 'execution_price' in trades.columns:
            trades['trade_value'] = trades[cols['position']].abs() * trades['execution_price']
            vwap = trades['trade_value'].sum() / trades[cols['position']].abs().sum()
            result['vwap'] = float(vwap)
        
        # Calculate slippage
        if 'slippage' in metrics_to_calculate and 'execution_price' in trades.columns:
            calculate_slippage(trades, cols, result)
        
        # Count executions
        if 'execution_count' in metrics_to_calculate:
            result['execution_count'] = len(trades)
        
        # Calculate turnover
        if 'turnover' in metrics_to_calculate:
            result['turnover'] = float(trades['notional'].sum() * 2)  # Both sides of trades
        
        # Calculate implementation shortfall
        if 'implementation_shortfall' in metrics_to_calculate and 'target_price' in trades.columns:
            calculate_implementation_shortfall(trades, cols, result)
            
        return result
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return result


def calculate_slippage(trades: DataFrame, cols: Dict[str, str], result: Dict[str, Any]) -> None:
    """Calculate slippage metrics and update result dictionary."""
    # For buys: (execution_price - reference_price) / reference_price * 10000
    # For sells: (reference_price - execution_price) / reference_price * 10000
    buy_mask = trades[cols['position']] > 0
    sell_mask = trades[cols['position']] < 0
    
    trades['slippage_bps'] = 0.0
    
    if buy_mask.any():
        trades.loc[buy_mask, 'slippage_bps'] = (
            (trades.loc[buy_mask, 'execution_price'] - trades.loc[buy_mask, cols['price']]) / 
            trades.loc[buy_mask, cols['price']] * 10000
        )
    
    if sell_mask.any():
        trades.loc[sell_mask, 'slippage_bps'] = (
            (trades.loc[sell_mask, cols['price']] - trades.loc[sell_mask, 'execution_price']) / 
            trades.loc[sell_mask, cols['price']] * 10000
        )
    
    # Weight by notional value for average
    result['slippage_bps'] = float((trades['slippage_bps'] * trades['notional']).sum() / trades['notional'].sum())


def calculate_implementation_shortfall(trades: DataFrame, cols: Dict[str, str], result: Dict[str, Any]) -> None:
    """Calculate implementation shortfall metrics and update result dictionary."""
    buy_mask = trades[cols['position']] > 0
    sell_mask = trades[cols['position']] < 0
    
    trades['shortfall_bps'] = 0.0
    
    if buy_mask.any():
        trades.loc[buy_mask, 'shortfall_bps'] = (
            (trades.loc[buy_mask, 'execution_price'] - trades.loc[buy_mask, 'target_price']) / 
            trades.loc[buy_mask, 'target_price'] * 10000
        )
    
    if sell_mask.any():
        trades.loc[sell_mask, 'shortfall_bps'] = (
            (trades.loc[sell_mask, 'target_price'] - trades.loc[sell_mask, 'execution_price']) / 
            trades.loc[sell_mask, 'target_price'] * 10000
        )
    
    # Weight by notional value for average
    result['implementation_shortfall_bps'] = float(
        (trades['shortfall_bps'] * trades['notional']).sum() / trades['notional'].sum()
    )


# Maintain compatibility with existing codebase that might import these
from quant_research.backtest.utils import normalize_positions