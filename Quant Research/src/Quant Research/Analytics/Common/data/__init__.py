"""
Data Utilities

This module provides fundamental data handling utilities for time series analysis.
It serves as the foundation for all data operations across the library, focusing on:

1. Core DataFrame operations (validation, index handling)
2. Time series processing (filtering, alignment, resampling)
3. Financial calculations (returns, metrics)
4. Data cleaning utilities (outlier detection, normalization)
5. Feature engineering (creating derived features)
6. Cross-validation for time series data

All other modules should build upon these core utilities rather than reimplementing
similar functionality.
"""

# Standard library imports
import logging

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.data")

# Re-export all functions from submodules to maintain backward compatibility
from .core_operations import (
    ensure_datetime_index,
    validate_ohlc,
    detect_frequency,
    validate_missing_data
)

from .time_series import (
    filter_time_range,
    resample_data,
    align_dataframes
)

from .financial_metrics import (
    calculate_returns,
    calculate_cumulative_returns,
    calculate_drawdowns,
    calculate_momentum,
    calculate_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    calculate_adx
)

from .cleaning import (
    detect_outliers,
    handle_outliers,
    normalize_data,
    calculate_zscore
)

from .feature_engineering import (
    add_lagged_features,
    add_difference_features,
    add_rolling_features
)

from .cross_validation import (
    time_series_split,
    expanding_window_split,
    walk_forward_split
)

# Define public API
__all__ = [
    # Core operations
    'ensure_datetime_index', 'validate_ohlc', 'detect_frequency', 'validate_missing_data',
    
    # Time series
    'filter_time_range', 'resample_data', 'align_dataframes',
    
    # Financial metrics
    'calculate_returns', 'calculate_cumulative_returns', 'calculate_drawdowns',
    'calculate_momentum', 'calculate_volatility', 'parkinson_volatility', 
    'garman_klass_volatility', 'calculate_adx',
    
    # Data cleaning
    'detect_outliers', 'handle_outliers', 'normalize_data', 'calculate_zscore',
    
    # Feature engineering
    'add_lagged_features', 'add_difference_features', 'add_rolling_features',
    
    # Cross-validation
    'time_series_split', 'expanding_window_split', 'walk_forward_split'
]