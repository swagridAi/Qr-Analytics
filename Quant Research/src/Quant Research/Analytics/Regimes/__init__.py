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