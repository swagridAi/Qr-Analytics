"""
Base module for market regime detection algorithms.

This module provides common functionality for both HMM-based and
change point-based regime detection methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from quant_research.core.models import Signal


logger = logging.getLogger(__name__)


def prepare_features(
    df: pd.DataFrame,
    features: List[str] = ["returns", "volatility"],
    window: int = 20,
    add_derived: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Prepare and normalize features for regime detection algorithms.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with price/return data
    features : List[str]
        List of column names to use as features
    window : int
        Window size for calculating derived features like volatility
    add_derived : bool
        Whether to add derived features if not already present
        
    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler]
        Processed feature dataframe and fitted scaler
    """
    feature_df = df.copy()
    
    # Add derived features if not present and requested
    if add_derived:
        if "returns" not in feature_df.columns and "close" in feature_df.columns:
            feature_df["returns"] = feature_df["close"].pct_change()
            
        if "volatility" not in feature_df.columns and "returns" in feature_df.columns:
            feature_df["volatility"] = feature_df["returns"].rolling(window=window).std()
            
        if "volume_change" not in feature_df.columns and "volume" in feature_df.columns:
            feature_df["volume_change"] = feature_df["volume"].pct_change()
    
    # Select only requested features
    feature_subset = [f for f in features if f in feature_df.columns]
    
    if not feature_subset:
        raise ValueError(f"None of the requested features {features} found in dataframe")
    
    # Create feature matrix
    X = feature_df[feature_subset].copy()
    
    # Drop NaN values (from rolling calculations)
    X = X.dropna()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=X.columns
    )
    
    return X_scaled, scaler


def calculate_regime_metrics(
    states: np.ndarray,
    index: pd.Index,
    probabilities: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Calculate additional regime metrics from the predicted states.
    
    Parameters
    ----------
    states : np.ndarray
        Array of predicted regime states
    index : pd.Index
        Index from original dataframe
    probabilities : Optional[np.ndarray]
        Array of state probabilities (for HMM)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regime metrics
    """
    # Create DataFrame with state
    regime_df = pd.DataFrame({
        "regime_state": states,
    }, index=index)
    
    # Add probabilities if available
    if probabilities is not None:
        regime_df["regime_probability"] = [p for p in probabilities]
        regime_df["dominant_probability"] = np.max(probabilities, axis=1)
    
    # Calculate regime duration (consecutive periods in same state)
    regime_df["state_changed"] = regime_df["regime_state"].diff().ne(0).astype(int)
    regime_df["regime_duration"] = regime_df.groupby(
        regime_df["state_changed"].cumsum()
    )["regime_state"].transform("count")
    
    # Drop intermediate column
    regime_df = regime_df.drop(columns=["state_changed"])
    
    return regime_df


def state_analysis(
    df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> Dict[int, Dict[str, float]]:
    """
    Analyze characteristics of each regime state.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original market data
    regime_df : pd.DataFrame
        DataFrame with regime states
        
    Returns
    -------
    Dict[int, Dict[str, float]]
        Dictionary with statistics for each regime
    """
    # Combine data
    analysis_df = df.join(regime_df[["regime_state"]], how="inner")
    
    # Calculate metrics per regime
    result = {}
    
    for state in analysis_df["regime_state"].unique():
        state_data = analysis_df[analysis_df["regime_state"] == state]
        
        # Calculate statistics
        if "returns" in analysis_df.columns:
            returns = state_data["returns"]
            volatility = returns.std() * np.sqrt(252)  # Annualized
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            volatility = np.nan
            sharpe = np.nan
            
        result[int(state)] = {
            "count": len(state_data),
            "pct_of_total": len(state_data) / len(analysis_df),
            "volatility_annualized": volatility,
            "sharpe_ratio": sharpe,
            "avg_duration": state_data["regime_duration"].mean() if "regime_duration" in state_data.columns else np.nan,
            "max_duration": state_data["regime_duration"].max() if "regime_duration" in state_data.columns else np.nan,
        }
        
        # Add additional metrics if columns exist
        for col in ["volume", "close"]:
            if col in state_data.columns:
                result[int(state)][f"{col}_mean"] = state_data[col].mean()
                result[int(state)][f"{col}_std"] = state_data[col].std()
    
    return result


class RegimeDetectorBase(ABC):
    """
    Base class for regime detection algorithms.
    
    This abstract class defines the common interface for all
    regime detection implementations.
    """
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, **kwargs) -> Any:
        """
        Fit the regime detection model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        **kwargs
            Additional parameters for the specific algorithm
            
        Returns
        -------
        Any
            Fitted model
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict regimes for the given data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Array of regime states and optionally state probabilities
        """
        pass
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, states: np.ndarray, 
                         index: pd.Index, **kwargs) -> List[Signal]:
        """
        Generate signals based on detected regimes.
        
        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe
        states : np.ndarray
            Detected regime states
        index : pd.Index
            Index from dataframe
        **kwargs
            Additional parameters
            
        Returns
        -------
        List[Signal]
            List of generated signals
        """
        pass
    
    def generate_signal(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate market regime signals.
        
        This is the main entry point for all regime detection algorithms.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with market data
        **kwargs
            Additional parameters for specific algorithms
            
        Returns
        -------
        pd.DataFrame
            DataFrame with regime states and additional metrics
        """
        # Prepare features
        X, scaler = prepare_features(
            df,
            features=kwargs.get("features", ["returns", "volatility"]),
            window=kwargs.get("window", 20),
            add_derived=kwargs.get("add_derived_features", True),
        )
        
        # Fit model
        model = self.fit(X, **kwargs)
        
        # Predict regimes
        states, probabilities = self.predict(X)
        
        # Calculate regime metrics
        regime_df = calculate_regime_metrics(states, X.index, probabilities)
        
        # Generate signals
        signals = self.generate_signals(df, states, X.index, **kwargs)
        
        logger.info(f"Generated {len(signals)} regime signals")
        
        # Merge back with original data
        result = df.join(regime_df, how="left")
        
        # Add model metadata
        for key, value in self.get_metadata().items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                result[key] = value
        
        return result
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the fitted model.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with model metadata
        """
        pass