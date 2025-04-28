"""
Data Cleaning Utilities

This module provides functions for cleaning and normalizing time series data,
including outlier detection, handling missing values, and normalization.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.data.cleaning")

def detect_outliers(
    series: pd.Series,
    method: str = 'zscore',
    threshold: float = 3.0,
    window: Optional[int] = None
) -> pd.Series:
    """
    Detect outliers in a time series.
    
    Args:
        series: Input time series
        method: Detection method ('zscore', 'iqr', 'mad')
        threshold: Threshold for outlier detection
        window: Window size for rolling detection (None for global)
        
    Returns:
        Boolean Series with True for outliers
        
    Raises:
        ValueError: If an invalid method is specified
    """
    if method == 'zscore':
        if window is None:
            # Global z-score
            mean = series.mean()
            std = series.std()
            z_scores = (series - mean) / std
            return z_scores.abs() > threshold
        else:
            # Rolling z-score
            z_scores = calculate_zscore(series, window=window)
            return z_scores.abs() > threshold
    
    elif method == 'iqr':
        if window is None:
            # Global IQR
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return (series < lower_bound) | (series > upper_bound)
        else:
            # Rolling IQR
            outliers = pd.Series(False, index=series.index)
            for i in range(window, len(series) + 1):
                window_data = series.iloc[i - window:i]
                q1 = window_data.quantile(0.25)
                q3 = window_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                current_idx = series.index[i - 1]
                current_value = series.iloc[i - 1]
                outliers.loc[current_idx] = (current_value < lower_bound) or (current_value > upper_bound)
            
            return outliers
    
    elif method == 'mad':
        if window is None:
            # Global Median Absolute Deviation
            median = series.median()
            mad = (series - median).abs().median()
            return (series - median).abs() > threshold * mad
        else:
            # Rolling MAD
            outliers = pd.Series(False, index=series.index)
            for i in range(window, len(series) + 1):
                window_data = series.iloc[i - window:i]
                median = window_data.median()
                mad = (window_data - median).abs().median()
                
                current_idx = series.index[i - 1]
                current_value = series.iloc[i - 1]
                outliers.loc[current_idx] = (current_value - median).abs() > threshold * mad
            
            return outliers
    
    else:
        raise ValueError(f"Invalid outlier detection method: {method}. Use 'zscore', 'iqr', or 'mad'")


def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'zscore',
    threshold: float = 3.0,
    treatment: str = 'winsorize',
    window: Optional[int] = None
) -> pd.DataFrame:
    """
    Detect and handle outliers in specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (defaults to all numeric columns)
        method: Detection method ('zscore', 'iqr', 'mad')
        threshold: Threshold for outlier detection
        treatment: Treatment method ('winsorize', 'mean', 'median', 'drop', 'none')
        window: Window size for rolling detection (None for global)
        
    Returns:
        DataFrame with outliers handled
        
    Raises:
        ValueError: If an invalid treatment method is specified
    """
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Default to numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Verify that specified columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Process each column
    outlier_counts = {}
    for col in columns:
        # Skip columns with all NaN
        if result[col].isna().all():
            logger.warning(f"Skipping column '{col}': all values are NaN")
            continue
        
        # Detect outliers
        is_outlier = detect_outliers(
            result[col].dropna(),
            method=method,
            threshold=threshold,
            window=window
        )
        
        # Count outliers
        outlier_count = is_outlier.sum()
        outlier_counts[col] = outlier_count
        
        if outlier_count == 0:
            continue
        
        # Get outlier indices
        outlier_indices = is_outlier[is_outlier].index
        
        # Handle outliers according to treatment method
        if treatment == 'winsorize':
            if window is None:
                # Global winsorization
                lower_bound = result[col].quantile(0.01)
                upper_bound = result[col].quantile(0.99)
                result.loc[result[col] < lower_bound, col] = lower_bound
                result.loc[result[col] > upper_bound, col] = upper_bound
            else:
                for idx in outlier_indices:
                    # Find the window containing this index
                    pos = result.index.get_loc(idx)
                    window_start = max(0, pos - window + 1)
                    window_end = pos + 1
                    
                    # Get the data in this window
                    window_data = result.iloc[window_start:window_end][col]
                    
                    # Calculate winsorization bounds
                    lower_bound = window_data.quantile(0.01)
                    upper_bound = window_data.quantile(0.99)
                    
                    # Apply winsorization
                    if result.loc[idx, col] < lower_bound:
                        result.loc[idx, col] = lower_bound
                    elif result.loc[idx, col] > upper_bound:
                        result.loc[idx, col] = upper_bound
        
        elif treatment == 'mean':
            if window is None:
                # Replace with global mean
                mean_value = result[col].mean()
                result.loc[outlier_indices, col] = mean_value
            else:
                for idx in outlier_indices:
                    # Find the window containing this index
                    pos = result.index.get_loc(idx)
                    window_start = max(0, pos - window + 1)
                    window_end = pos + 1
                    
                    # Calculate mean excluding the outlier
                    window_values = result.iloc[window_start:window_end][col]
                    window_values = window_values[window_values.index != idx]
                    mean_value = window_values.mean()
                    
                    # Replace outlier with mean
                    result.loc[idx, col] = mean_value
        
        elif treatment == 'median':
            if window is None:
                # Replace with global median
                median_value = result[col].median()
                result.loc[outlier_indices, col] = median_value
            else:
                for idx in outlier_indices:
                    # Find the window containing this index
                    pos = result.index.get_loc(idx)
                    window_start = max(0, pos - window + 1)
                    window_end = pos + 1
                    
                    # Calculate median excluding the outlier
                    window_values = result.iloc[window_start:window_end][col]
                    window_values = window_values[window_values.index != idx]
                    median_value = window_values.median()
                    
                    # Replace outlier with median
                    result.loc[idx, col] = median_value
        
        elif treatment == 'drop':
            # Drop rows with outliers
            result = result.drop(outlier_indices)
        
        elif treatment == 'none':
            # Just detect, don't treat
            continue
        
        else:
            raise ValueError(
                f"Invalid outlier treatment method: {treatment}. Use "
                f"'winsorize', 'mean', 'median', 'drop', or 'none'."
            )
    
    # Log outlier counts
    if any(count > 0 for count in outlier_counts.values()):
        outlier_info = ", ".join([f"{col}: {count}" for col, count in outlier_counts.items() if count > 0])
        logger.info(f"Outliers detected and {treatment}d - {outlier_info}")
    
    return result


def normalize_data(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'standard',
    feature_range: Tuple[float, float] = (0, 1),
    window: Optional[int] = None,
    output_scaler: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]:
    """
    Normalize or scale data.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize (defaults to all numeric columns)
        method: Scaling method ('standard', 'minmax', 'robust', 'log', 'box-cox')
        feature_range: Range for MinMaxScaler
        window: Window size for rolling normalization (None for global)
        output_scaler: Whether to return the fitted scaler
        
    Returns:
        Normalized DataFrame or tuple of (DataFrame, scaler)
        
    Raises:
        ValueError: If an invalid method is specified
    """
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Default to numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Verify that specified columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # For global scaling
    if window is None:
        # Create the appropriate scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            scaler = RobustScaler()
        elif method in ['log', 'box-cox']:
            # For these methods we don't use sklearn scalers
            scaler = None
        else:
            raise ValueError(
                f"Invalid normalization method: {method}. Use "
                f"'standard', 'minmax', 'robust', 'log', or 'box-cox'."
            )
        
        # Apply scaling
        if method in ['standard', 'minmax', 'robust']:
            # Extract columns to scale
            data_to_scale = result[columns].values
            
            # Fit and transform
            scaled_data = scaler.fit_transform(data_to_scale)
            
            # Update DataFrame
            result[columns] = scaled_data
            
        elif method == 'log':
            # Check for non-positive values
            for col in columns:
                if (result[col] <= 0).any():
                    min_val = result[col].min()
                    if min_val <= 0:
                        # Shift to make all values positive
                        shift = abs(min_val) + 1e-6
                        logger.warning(
                            f"Column '{col}' contains non-positive values. "
                            f"Shifting by {shift} for log transformation."
                        )
                        result[col] = result[col] + shift
                
                # Apply log transformation
                result[col] = np.log(result[col])
            
        elif method == 'box-cox':
            from scipy import stats
            
            # Apply Box-Cox transformation
            for col in columns:
                # Check for non-positive values
                if (result[col] <= 0).any():
                    min_val = result[col].min()
                    if min_val <= 0:
                        # Shift to make all values positive
                        shift = abs(min_val) + 1e-6
                        logger.warning(
                            f"Column '{col}' contains non-positive values. "
                            f"Shifting by {shift} for Box-Cox transformation."
                        )
                        result[col] = result[col] + shift
                
                # Apply Box-Cox transformation
                transformed_data, lambda_param = stats.boxcox(result[col].values)
                result[col] = transformed_data
    
    # For rolling window scaling
    else:
        # Only implemented for basic methods
        if method not in ['standard', 'minmax']:
            raise ValueError(
                f"Rolling normalization only supports 'standard' and 'minmax' methods, got '{method}'"
            )
        
        for col in columns:
            # For each point, normalize based on the preceding window
            for i in range(window, len(result) + 1):
                window_data = result.iloc[i - window:i][col].values.reshape(-1, 1)
                
                if method == 'standard':
                    scaler = StandardScaler().fit(window_data)
                else:  # minmax
                    scaler = MinMaxScaler(feature_range=feature_range).fit(window_data)
                
                # Transform only the current point
                current_idx = result.index[i - 1]
                current_value = result.loc[current_idx, col]
                result.loc[current_idx, col] = scaler.transform([[current_value]])[0][0]
        
        # No scaler to return for rolling window
        scaler = None
    
    if output_scaler:
        return result, scaler
    else:
        return result


def calculate_zscore(
    series: pd.Series, 
    window: int = 60,
    method: str = 'rolling',
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate z-score for a time series.
    
    Args:
        series: Time series to calculate z-score for
        window: Rolling window size
        method: Method for calculation ('rolling', 'ewm', 'expanding', 'regime_adjusted')
        min_periods: Minimum number of observations required
        
    Returns:
        Series of z-scores
        
    Raises:
        ValueError: If an invalid method is specified
    """
    if min_periods is None:
        min_periods = window // 2
        
    # Calculate means and standard deviations
    if method == 'rolling':
        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()
        
        # Replace zero standard deviations with NaN to avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        z_score = (series - rolling_mean) / rolling_std
    
    elif method == 'ewm':
        ewma_mean = series.ewm(span=window, min_periods=min_periods).mean()
        ewma_std = series.ewm(span=window, min_periods=min_periods).std()
        
        # Replace zero standard deviations with NaN to avoid division by zero
        ewma_std = ewma_std.replace(0, np.nan)
        
        z_score = (series - ewma_mean) / ewma_std
    
    elif method == 'expanding':
        expanding_mean = series.expanding(min_periods=min_periods).mean()
        expanding_std = series.expanding(min_periods=min_periods).std()
        
        # Replace zero standard deviations with NaN to avoid division by zero
        expanding_std = expanding_std.replace(0, np.nan)
        
        z_score = (series - expanding_mean) / expanding_std
    
    elif method == 'regime_adjusted':
        # First calculate standard rolling z-score
        z_score = calculate_zscore(series, window, 'rolling', min_periods)
        
        # Calculate volatility of volatility (meta-volatility)
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()
        vol_window = window * 3  # Longer window for regime detection
        vol_of_vol = rolling_std.rolling(window=vol_window).std() / rolling_std.rolling(window=vol_window).mean()
        
        # Calculate scaling factor (higher volatility periods get downscaled)
        scaling = 1.0 / np.maximum(1.0, vol_of_vol / vol_of_vol.rolling(window=vol_window).median())
        
        # Apply scaling to original z-score
        z_score = z_score * scaling
        
    else:
        raise ValueError(f"Invalid z-score method: {method}. Use 'rolling', 'ewm', 'expanding', or 'regime_adjusted'")
    
    return z_score