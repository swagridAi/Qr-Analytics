"""
Time Series Processing

This module provides functions for manipulating and transforming time series data,
including filtering, resampling, and alignment operations.
"""

# Standard library imports
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.data.time_series")

def filter_time_range(
    df: pd.DataFrame,
    start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    inclusive: str = 'both'
) -> pd.DataFrame:
    """
    Filter DataFrame to a specific time range.
    
    Args:
        df: Input DataFrame with datetime index
        start_date: Start date for the filter
        end_date: End date for the filter
        inclusive: Whether to include start/end in the range ('both', 'left', 'right', 'neither')
        
    Returns:
        Filtered DataFrame
        
    Raises:
        TypeError: If DataFrame doesn't have a datetime index
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a datetime index for time filtering")
    
    # Convert string dates to timestamps
    if start_date is not None and isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if end_date is not None and isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    
    # Default to entire range if nothing specified
    if start_date is None and end_date is None:
        return df
    
    # Create the mask based on the specified range
    if start_date is not None and end_date is not None:
        if inclusive == 'both':
            mask = (df.index >= start_date) & (df.index <= end_date)
        elif inclusive == 'left':
            mask = (df.index >= start_date) & (df.index < end_date)
        elif inclusive == 'right':
            mask = (df.index > start_date) & (df.index <= end_date)
        else:  # 'neither'
            mask = (df.index > start_date) & (df.index < end_date)
    elif start_date is not None:
        mask = df.index >= start_date
    else:  # end_date is not None
        mask = df.index <= end_date
    
    # Apply the filter
    filtered_df = df[mask]
    
    # Log the filtering results
    logger.debug(
        f"Filtered from {len(df)} to {len(filtered_df)} rows "
        f"({100 * len(filtered_df) / len(df):.1f}% retained)"
    )
    
    return filtered_df


def resample_data(
    df: pd.DataFrame, 
    freq: str,
    agg_dict: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Resample time series data to a specified frequency.
    
    Args:
        df: Input DataFrame with datetime index
        freq: Target frequency (e.g., '1D', '1H', '15T')
        agg_dict: Dictionary of column:aggregation mappings
        
    Returns:
        Resampled DataFrame
        
    Raises:
        TypeError: If DataFrame doesn't have a datetime index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a datetime index for resampling")
    
    # Default aggregation for OHLCV data
    if agg_dict is None:
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    # Filter to only include columns that exist in the DataFrame
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    # Ensure we have at least one column to aggregate
    if not agg_dict:
        raise ValueError(
            "No valid columns for aggregation. Provide an agg_dict with "
            "mappings for columns that exist in the DataFrame."
        )
    
    # Perform the resampling
    resampled = df.resample(freq).agg(agg_dict)
    
    # Log the resampling results
    logger.debug(
        f"Resampled from {len(df)} to {len(resampled)} rows "
        f"(frequency: {freq})"
    )
    
    return resampled


def align_dataframes(
    dfs: List[pd.DataFrame],
    method: str = 'outer',
    fill_method: Optional[str] = None,
    freq: Optional[str] = None
) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to a common time index.
    
    Args:
        dfs: List of DataFrames with datetime indices
        method: Join method ('outer', 'inner', 'left', 'right')
        fill_method: Method for filling missing values (None, 'ffill', 'bfill', 'nearest')
        freq: Optional frequency to resample all DataFrames to
        
    Returns:
        List of aligned DataFrames
        
    Raises:
        TypeError: If any DataFrame doesn't have a datetime index
    """
    # Validate inputs
    for i, df in enumerate(dfs):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"DataFrame {i} must have a datetime index for alignment")
    
    # Resample to common frequency if specified
    if freq is not None:
        resampled_dfs = []
        for df in dfs:
            # Detect columns and appropriate aggregation methods
            cols = df.columns
            agg_dict = {}
            
            for col in cols:
                if any(price_kw in col.lower() for price_kw in ['open', 'first', 'start']):
                    agg_dict[col] = 'first'
                elif any(price_kw in col.lower() for price_kw in ['high', 'max']):
                    agg_dict[col] = 'max'
                elif any(price_kw in col.lower() for price_kw in ['low', 'min']):
                    agg_dict[col] = 'min'
                elif any(price_kw in col.lower() for price_kw in ['close', 'last', 'end']):
                    agg_dict[col] = 'last'
                elif any(vol_kw in col.lower() for vol_kw in ['volume', 'qty', 'amount', 'size']):
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'last'  # Default to last
                
            resampled_dfs.append(df.resample(freq).agg(agg_dict))
        
        dfs = resampled_dfs
    
    # Determine the reference index based on the join method
    if method == 'outer':
        # Union of all indices
        ref_index = dfs[0].index
        for df in dfs[1:]:
            ref_index = ref_index.union(df.index)
    elif method == 'inner':
        # Intersection of all indices
        ref_index = dfs[0].index
        for df in dfs[1:]:
            ref_index = ref_index.intersection(df.index)
    elif method == 'left':
        # Use the first DataFrame's index
        ref_index = dfs[0].index
    elif method == 'right':
        # Use the last DataFrame's index
        ref_index = dfs[-1].index
    else:
        raise ValueError(f"Invalid join method: {method}")
    
    # Reindex all DataFrames to the reference index
    aligned_dfs = []
    for df in dfs:
        aligned_df = df.reindex(ref_index)
        
        # Apply fill method if specified
        if fill_method == 'ffill':
            aligned_df = aligned_df.ffill()
        elif fill_method == 'bfill':
            aligned_df = aligned_df.bfill()
        elif fill_method == 'nearest':
            aligned_df = aligned_df.interpolate(method='nearest')
        
        aligned_dfs.append(aligned_df)
    
    return aligned_dfs