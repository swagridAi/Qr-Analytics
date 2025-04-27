"""
Hidden Markov Model implementation for market regime detection.

This module uses hmmlearn to identify distinct market regimes based on
returns, volatility, and other optional features.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from hmmlearn import hmm

from quant_research.core.models import Signal
from quant_research.analytics.regimes.base import (
    RegimeDetectorBase, 
    prepare_features,
    state_analysis
)


logger = logging.getLogger(__name__)


class HMMRegimeDetector(RegimeDetectorBase):
    """
    Regime detector using Hidden Markov Models.
    
    This implementation uses hmmlearn's GaussianHMM to detect
    different market regimes based on multivariate time series data.
    """
    
    def __init__(self):
        self.model = None
        self.n_states = None
        self.covariance_type = None
        self.random_state = None
    
    def fit(self, X: pd.DataFrame, **kwargs) -> hmm.GaussianHMM:
        """
        Fit a Hidden Markov Model on the provided feature matrix.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (standardized)
        **kwargs
            Additional parameters including:
            - n_states: Number of hidden states to detect
            - covariance_type: Type of covariance parameter
            - n_iter: Number of iterations for EM algorithm
            - random_state: Random seed for reproducibility
            
        Returns
        -------
        hmm.GaussianHMM
            Trained HMM model
        """
        # Extract parameters
        self.n_states = kwargs.get("n_states", 3)
        self.covariance_type = kwargs.get("covariance_type", "full")
        n_iter = kwargs.get("n_iter", 100)
        self.random_state = kwargs.get("random_state", 42)
        
        # Convert DataFrame to numpy array if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Initialize and train HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=n_iter,
            random_state=self.random_state,
        )
        
        self.model.fit(X_array)
        
        logger.info(f"HMM training completed with score: {self.model.score(X_array):.2f}")
        
        return self.model
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regimes for the given data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Array of regime states and state probabilities
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert DataFrame to numpy array if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Predict states and probabilities
        states = self.model.predict(X_array)
        state_proba = self.model.predict_proba(X_array)
        
        return states, state_proba
    
    def generate_signals(
        self, 
        df: pd.DataFrame, 
        states: np.ndarray,
        index: pd.Index,
        **kwargs
    ) -> List[Signal]:
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
        if self.model is None:
            raise ValueError("Model must be fitted before generating signals")
        
        signals = []
        
        # Create state series for easier analysis
        state_series = pd.Series(states, index=index)
        
        for state in range(self.n_states):
            # Create signal when entering a new regime
            state_entries = (state_series == state) & (state_series.shift(1) != state)
            
            for idx in index[state_entries]:
                # Get probability for this state at this timestamp
                proba_idx = np.where(index == idx)[0][0]
                if proba_idx < len(states):
                    # Create signal
                    signals.append(
                        Signal(
                            timestamp=idx,
                            signal_type=f"regime_state_{state}",
                            value=1.0,
                            confidence=float(self.model.predict_proba(df.loc[idx:idx].values)[0][state]),
                            metadata={
                                "regime_duration_expected": float(1 / (1 - self.model.transmat_[state, state])),
                                "source": "hmm",
                            }
                        )
                    )
        
        return signals
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the fitted HMM model.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with model metadata
        """
        if self.model is None:
            return {
                "model_type": "hmm",
                "fitted": False
            }
        
        return {
            "model_type": "hmm",
            "fitted": True,
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "log_likelihood": float(self.model.score(self.model.monitor_.history[-1])),
            "n_iter": len(self.model.monitor_.history),
            "transition_matrix": str(self.model.transmat_),
        }


def generate_signal(
    df: pd.DataFrame,
    n_states: int = 3,
    features: List[str] = ["returns", "volatility"],
    window: int = 20,
    covariance_type: str = "full",
    add_derived_features: bool = True,
    n_iter: int = 100,
    random_state: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate market regime signals using Hidden Markov Model.
    
    This is a convenience function that wraps the HMMRegimeDetector class.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with market data
    n_states : int
        Number of regimes to detect
    features : List[str]
        Features to use for regime detection
    window : int
        Lookback window for derived features
    covariance_type : str
        Covariance type for HMM (full, tied, diagonal, spherical)
    add_derived_features : bool
        Whether to add derived features from raw data
    n_iter : int
        Number of iterations for HMM training
    random_state : int
        Random seed for reproducibility
    **kwargs
        Additional parameters passed to HMM
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regime states and probabilities
    """
    detector = HMMRegimeDetector()
    
    return detector.generate_signal(
        df=df,
        features=features,
        window=window,
        n_states=n_states,
        covariance_type=covariance_type,
        add_derived_features=add_derived_features,
        n_iter=n_iter,
        random_state=random_state,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download some test data
    data = yf.download("SPY", start="2020-01-01", end="2022-12-31")
    
    # Generate regime signals
    result = generate_signal(data, n_states=3)
    
    # Analyze regimes
    analysis = state_analysis(data, result)
    
    print("Regime Analysis:")
    for state, metrics in analysis.items():
        print(f"State {state}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")