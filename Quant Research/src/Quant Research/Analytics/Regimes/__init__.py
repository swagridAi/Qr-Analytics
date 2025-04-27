"""
Regime detection module for market state identification.

This package provides tools for identifying distinct market regimes
using various statistical methods like Hidden Markov Models and
change point detection.
"""

from typing import Dict, Any, List, Optional

import pandas as pd

from quant_research.analytics.regimes.hmm import (
    HMMRegimeDetector,
    generate_signal as hmm_generate_signal
)
from quant_research.analytics.regimes.change_point import (
    ChangePointDetector,
    generate_signal as cp_generate_signal,
    online_detection
)
from quant_research.analytics.regimes.base import state_analysis


def generate_signal(
    df: pd.DataFrame,
    method: str = "hmm",
    n_states: Optional[int] = None,
    n_bkps: Optional[int] = None,
    features: List[str] = ["returns", "volatility"],
    **kwargs
) -> pd.DataFrame:
    """
    Generate market regime signals using the specified method.
    
    This is the main entry point for the regimes module.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with market data
    method : str
        Detection method: 'hmm' or 'change_point'
    n_states : Optional[int]
        Number of regimes to detect (for HMM)
    n_bkps : Optional[int]
        Number of breakpoints to detect (for change_point)
    features : List[str]
        Features to use for regime detection
    **kwargs
        Additional parameters for specific algorithms
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regime states and additional metrics
        
    Examples
    --------
    >>> import yfinance as yf
    >>> data = yf.download("SPY", start="2020-01-01", end="2022-01-01")
    >>> # Using HMM
    >>> hmm_result = generate_signal(data, method="hmm", n_states=3)
    >>> # Using change point detection
    >>> cp_result = generate_signal(data, method="change_point", n_bkps=10)
    """
    if method.lower() == "hmm":
        if n_states is not None:
            kwargs["n_states"] = n_states
        return hmm_generate_signal(df, features=features, **kwargs)
    
    elif method.lower() in ["change_point", "changepoint", "cp"]:
        if n_bkps is not None:
            kwargs["n_bkps"] = n_bkps
        return cp_generate_signal(df, features=features, **kwargs)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hmm' or 'change_point'.")


__all__ = [
    "generate_signal",
    "HMMRegimeDetector",
    "ChangePointDetector",
    "online_detection",
    "state_analysis",
]