"""
Time Series Cross-Validation

This module provides functions for creating proper train-test splits for time series data,
ensuring that temporal dependencies are respected to prevent data leakage.
"""

# Standard library imports
import logging
from typing import List, Optional, Tuple

# Third-party imports
import pandas as pd

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.data.cross_validation")

def time_series_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_pct: float = 0.8,
    valid_pct: float = 0.1,
    gap: int = 0,
    min_train_size: Optional[int] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create time series train/validation/test splits.
    
    Args:
        df: Input DataFrame with datetime index
        n_splits: Number of splits to create
        train_pct: Percentage of data for training in each split
        valid_pct: Percentage of data for validation in each split
        gap: Number of samples to skip between train and validation/test
        min_train_size: Minimum size of training set
        
    Returns:
        List of (train, validation, test) DataFrame tuples
        
    Raises:
        TypeError: If DataFrame doesn't have a datetime index
        ValueError: If invalid split percentages are provided
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a datetime index for time series split")
    
    # Validate percentages
    if train_pct + valid_pct >= 1.0:
        raise ValueError(
            f"Sum of train_pct ({train_pct}) and valid_pct ({valid_pct}) "
            f"must be less than 1.0 to allow for a test set"
        )
    
    if train_pct <= 0 or valid_pct <= 0 or (1 - train_pct - valid_pct) <= 0:
        raise ValueError("All split percentages must be positive")
    
    # Calculate default min_train_size
    if min_train_size is None:
        min_train_size = int(len(df) * train_pct / n_splits)
    
    # Create splits
    splits = []
    
    # Size of each increment
    incr_size = (len(df) - min_train_size) // (n_splits - 1) if n_splits > 1 else 0
    
    for i in range(n_splits):
        # Calculate split indices
        train_end = min_train_size + i * incr_size
        valid_start = train_end + gap
        valid_end = valid_start + int(len(df) * valid_pct)
        test_start = valid_end + gap
        test_end = min(test_start + int(len(df) * (1 - train_pct - valid_pct)), len(df))
        
        # Skip if not enough data for all splits
        if test_end <= test_start or valid_end <= valid_start:
            logger.warning(f"Skipping split {i+1}/{n_splits}: insufficient data")
            continue
        
        # Create splits
        train = df.iloc[:train_end]
        valid = df.iloc[valid_start:valid_end]
        test = df.iloc[test_start:test_end]
        
        splits.append((train, valid, test))
    
    logger.info(f"Created {len(splits)} time series splits")
    
    return splits


def expanding_window_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    gap: int = 0,
    min_train_size: Optional[int] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create expanding window time series splits (train set grows with each split).
    
    Args:
        df: Input DataFrame with datetime index
        n_splits: Number of splits to create
        test_size: Proportion of data for testing
        valid_size: Proportion of data for validation
        gap: Number of samples to skip between train and validation/test
        min_train_size: Minimum size of training set
        
    Returns:
        List of (train, validation, test) DataFrame tuples
        
    Raises:
        TypeError: If DataFrame doesn't have a datetime index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a datetime index for expanding window split")
    
    # Set default min_train_size
    if min_train_size is None:
        min_train_size = int(len(df) * 0.3)
    
    # Total size of validation and test portions
    eval_size = test_size + valid_size
    
    # Calculate size of each step
    full_size = len(df)
    eval_portion = full_size * eval_size  # Size of combined validation and test data
    step_size = (full_size - eval_portion - min_train_size) / (n_splits - 1) if n_splits > 1 else 0
    
    # Create splits
    splits = []
    
    for i in range(n_splits):
        # Calculate split indices
        train_end = min_train_size + int(i * step_size)
        valid_start = train_end + gap
        valid_end = valid_start + int(valid_size * full_size)
        test_start = valid_end + gap
        test_end = min(test_start + int(test_size * full_size), full_size)
        
        # Skip if not enough data for all splits
        if test_end <= test_start or valid_end <= valid_start:
            logger.warning(f"Skipping split {i+1}/{n_splits}: insufficient data")
            continue
        
        # Create splits
        train = df.iloc[:train_end]
        valid = df.iloc[valid_start:valid_end]
        test = df.iloc[test_start:test_end]
        
        splits.append((train, valid, test))
    
    logger.info(f"Created {len(splits)} expanding window splits")
    
    return splits


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    valid_size: Optional[int] = None,
    step_size: Optional[int] = None,
    gap: int = 0
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward time series splits with sliding windows.
    
    Args:
        df: Input DataFrame with datetime index
        n_splits: Number of splits to create
        train_size: Number of samples in training set (default is 70% of data)
        test_size: Number of samples in test set (default is 15% of data)
        valid_size: Number of samples in validation set (default is 15% of data)
        step_size: Number of samples to slide window forward (default is test_size)
        gap: Number of samples to skip between train/validation/test
        
    Returns:
        List of (train, validation, test) DataFrame tuples
        
    Raises:
        TypeError: If DataFrame doesn't have a datetime index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a datetime index for walk forward split")
    
    # Set defaults
    if train_size is None:
        train_size = int(len(df) * 0.7)
    
    if test_size is None:
        test_size = int(len(df) * 0.15)
    
    if valid_size is None:
        valid_size = int(len(df) * 0.15)
    
    if step_size is None:
        step_size = test_size
    
    # Ensure there's enough data
    if train_size + valid_size + test_size + 2 * gap > len(df):
        raise ValueError(
            f"Insufficient data for specified split sizes: "
            f"train_size={train_size}, valid_size={valid_size}, test_size={test_size}, gap={gap}"
        )
    
    # Create splits
    splits = []
    
    for i in range(n_splits):
        # Calculate start/end indices
        train_start = i * step_size
        train_end = train_start + train_size
        
        # If we've reached the end of the data, stop
        if train_end + valid_size + test_size + 2 * gap > len(df):
            break
        
        valid_start = train_end + gap
        valid_end = valid_start + valid_size
        test_start = valid_end + gap
        test_end = test_start + test_size
        
        # Create split
        train = df.iloc[train_start:train_end]
        valid = df.iloc[valid_start:valid_end]
        test = df.iloc[test_start:test_end]
        
        splits.append((train, valid, test))
    
    logger.info(f"Created {len(splits)} walk-forward splits")
    
    return splits