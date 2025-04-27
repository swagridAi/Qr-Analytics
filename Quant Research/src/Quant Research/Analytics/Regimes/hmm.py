"""
Hidden Markov Model implementation for market regime detection.

This module uses hmmlearn to identify distinct market regimes based on
returns, volatility, and other optional features. It extends the common
SignalGenerator base class and integrates with the analytics pipeline.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from quant_research.core.models import Signal
from quant_research.analytics.common.base import SignalGenerator, SignalGeneratorParams
from quant_research.analytics.common.data_prep import (
    add_technical_features, 
    ensure_datetime_index,
    calculate_returns
)
from quant_research.analytics.common.validation import (
    validate_numeric_param,
    validate_string_param,
    validate_list_param,
    validate_inputs
)
from quant_research.analytics.common.statistics import calculate_moments


logger = logging.getLogger(__name__)


@dataclass
class HMMParams(SignalGeneratorParams):
    """Parameters for HMM regime detection."""
    n_states: int = 3
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42
    features: List[str] = None
    window: int = 20
    add_derived_features: bool = True
    
    def __post_init__(self):
        """Set default values for optional fields."""
        super().__post_init__()
        if self.features is None:
            self.features = ["returns", "volatility"]


class HMMRegimeDetector(SignalGenerator):
    """
    Regime detector using Hidden Markov Models.
    
    This implementation uses hmmlearn's GaussianHMM to detect
    different market regimes based on multivariate time series data.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the HMM regime detector.
        
        Args:
            **kwargs: Parameters for regime detection
        """
        super().__init__(**kwargs)
        self.model = None
        self.params = self.validate_params(HMMParams, kwargs)
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Prepare features for HMM analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (feature_df, scaler)
        """
        # Ensure datetime index
        df = ensure_datetime_index(df)
        
        # Add derived features if needed
        feature_df = df.copy()
        
        if self.params.add_derived_features:
            # Add returns if not present
            if "returns" not in feature_df.columns and "close" in feature_df.columns:
                feature_df["returns"] = calculate_returns(
                    feature_df, 
                    column="close", 
                    return_type="log"
                )
            
            # Add technical features using common utility
            feature_df = add_technical_features(
                feature_df,
                price_col="close",
                window=self.params.window,
                include=self.params.features
            )
        
        # Select requested features
        feature_cols = [f for f in self.params.features if f in feature_df.columns]
        
        if not feature_cols:
            raise ValueError(
                f"None of the requested features {self.params.features} found in DataFrame"
            )
        
        # Extract feature data and drop NaNs
        X = feature_df[feature_cols].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        return X_scaled, scaler
    
    def _generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation-specific signal generation.
        
        Args:
            df: Preprocessed input DataFrame
            
        Returns:
            DataFrame with generated signals
        """
        # Prepare features
        X, _ = self._prepare_features(df)
        
        # Fit the model
        self._fit_model(X)
        
        # Generate predictions
        states, probabilities = self._predict(X)
        
        # Calculate regime metrics
        regime_df = self._calculate_regime_metrics(states, X.index, probabilities)
        
        # Join with original data (using left join to keep all original rows)
        result = df.join(regime_df, how='left')
        
        # Add metadata
        meta = self.get_metadata()
        for key, value in meta.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                result[key] = value
        
        return result
    
    def _fit_model(self, X: pd.DataFrame) -> hmm.GaussianHMM:
        """
        Fit the HMM model to the data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Fitted model
        """
        # Convert to numpy array
        X_array = X.values
        
        # Initialize model
        self.model = hmm.GaussianHMM(
            n_components=self.params.n_states,
            covariance_type=self.params.covariance_type,
            n_iter=self.params.n_iter,
            random_state=self.params.random_state
        )
        
        # Fit model
        self.model.fit(X_array)
        
        self.logger.info(f"HMM training completed with score: {self.model.score(X_array):.2f}")
        
        return self.model
    
    def _predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regimes using the fitted model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple of (states, probabilities)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to numpy array
        X_array = X.values
        
        # Predict states and probabilities
        states = self.model.predict(X_array)
        probabilities = self.model.predict_proba(X_array)
        
        return states, probabilities
    
    def _calculate_regime_metrics(
        self,
        states: np.ndarray,
        index: pd.Index,
        probabilities: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate additional regime metrics.
        
        Args:
            states: Array of predicted states
            index: DataFrame index
            probabilities: State probabilities
            
        Returns:
            DataFrame with regime metrics
        """
        # Create DataFrame with regime state
        regime_df = pd.DataFrame({
            "regime_state": states
        }, index=index)
        
        # Add probabilities
        regime_df["regime_probability"] = list(probabilities)
        regime_df["dominant_probability"] = np.max(probabilities, axis=1)
        
        # Calculate regime duration
        regime_df["state_changed"] = regime_df["regime_state"].diff().ne(0).astype(int)
        regime_df["regime_duration"] = regime_df.groupby(
            regime_df["state_changed"].cumsum()
        )["regime_state"].transform("count")
        
        # Remove temporary columns
        regime_df = regime_df.drop(columns=["state_changed"])
        
        return regime_df
    
    def generate_signals(self, df: pd.DataFrame, states: np.ndarray, 
                         index: pd.Index, **kwargs) -> List[Signal]:
        """
        Generate signals based on detected regimes.
        
        Args:
            df: Original DataFrame
            states: Detected regime states
            index: DataFrame index
            **kwargs: Additional parameters
            
        Returns:
            List of Signal objects
        """
        if self.model is None:
            raise ValueError("Model must be fitted before generating signals")
        
        signals = []
        
        # Create state series for easier analysis
        state_series = pd.Series(states, index=index)
        
        for state in range(self.params.n_states):
            # Create signal when entering a new regime
            state_entries = (state_series == state) & (state_series.shift(1) != state)
            
            for idx in index[state_entries]:
                # Get probability for this state at this timestamp
                try:
                    # Use predict_proba if we're looking at a single point
                    if idx in df.index:
                        point_data = df.loc[[idx]]
                        features, _ = self._prepare_features(point_data)
                        if not features.empty:
                            proba = self.model.predict_proba(features.values)[0][state]
                        else:
                            proba = np.nan
                    else:
                        proba = np.nan
                    
                    if np.isnan(proba):
                        # Fall back to the probability from the main calculation
                        proba_idx = np.where(index == idx)[0][0]
                        if proba_idx < len(probabilities):
                            proba = probabilities[proba_idx][state]
                except:
                    # If we can't get the probability, use a default
                    proba = 0.8  # Reasonable default
                
                # Create signal
                signals.append(
                    Signal(
                        timestamp=idx,
                        signal_type=f"regime_state_{state}",
                        value=1.0,
                        confidence=float(proba),
                        metadata={
                            "regime_duration_expected": float(1 / (1 - self.model.transmat_[state, state])),
                            "source": "hmm",
                        }
                    )
                )
        
        return signals
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the fitted model.
        
        Returns:
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
            "n_states": self.params.n_states,
            "covariance_type": self.params.covariance_type,
            "log_likelihood": float(self.model.score(self.model.monitor_.history[-1])),
            "n_iter": len(self.model.monitor_.history),
            "transition_matrix": str(self.model.transmat_),
        }


@validate_inputs(
    n_states=lambda x: validate_numeric_param(x, "n_states", min_value=2, integer_only=True),
    covariance_type=lambda x: validate_string_param(x, "covariance_type", 
                                                   allowed_values=["full", "tied", "diagonal", "spherical"])
)
def generate_signal(
    df: pd.DataFrame,
    n_states: int = 3,
    features: List[str] = None,
    window: int = 20,
    covariance_type: str = "full",
    add_derived_features: bool = True,
    n_iter: int = 100,
    random_state: int = 42,
    **kwargs
) -> pd.DataFrame:
    """
    Generate market regime signals using Hidden Markov Model.
    
    This is a convenience function that wraps the HMMRegimeDetector class.
    
    Args:
        df: Input dataframe with market data
        n_states: Number of regimes to detect
        features: Features to use for regime detection
        window: Lookback window for derived features
        covariance_type: Covariance type for HMM
        add_derived_features: Whether to add derived features
        n_iter: Number of iterations for HMM training
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with regime states and probabilities
    """
    # Set default features if not provided
    if features is None:
        features = ["returns", "volatility"]
    
    # Create detector
    detector = HMMRegimeDetector(
        n_states=n_states,
        features=features,
        window=window,
        covariance_type=covariance_type,
        add_derived_features=add_derived_features,
        n_iter=n_iter,
        random_state=random_state,
        **kwargs
    )
    
    # Generate signals
    return detector.generate_signal(df)


# Function for backwards compatibility with existing code
def state_analysis(df: pd.DataFrame, regime_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """
    Analyze characteristics of each regime state.
    
    Args:
        df: Original market data
        regime_df: DataFrame with regime states
        
    Returns:
        Dictionary with statistics for each regime
    """
    # Combine data
    analysis_df = df.join(regime_df[["regime_state"]], how="inner")
    
    # Calculate metrics per regime
    result = {}
    
    for state in analysis_df["regime_state"].unique():
        state_data = analysis_df[analysis_df["regime_state"] == state]
        
        # Calculate statistics using the common moments calculation function
        metrics = {}
        
        # Calculate return statistics if available
        if "returns" in analysis_df.columns:
            returns = state_data["returns"]
            return_stats = calculate_moments(returns, annualize=True, periods_per_year=252)
            metrics["volatility_annualized"] = return_stats["std"]
            metrics["sharpe_ratio"] = return_stats["mean"] / return_stats["std"] if return_stats["std"] > 0 else 0
        else:
            metrics["volatility_annualized"] = np.nan
            metrics["sharpe_ratio"] = np.nan
            
        # Add regime metrics
        metrics.update({
            "count": len(state_data),
            "pct_of_total": len(state_data) / len(analysis_df),
            "avg_duration": state_data["regime_duration"].mean() if "regime_duration" in state_data.columns else np.nan,
            "max_duration": state_data["regime_duration"].max() if "regime_duration" in state_data.columns else np.nan,
        })
        
        # Add column statistics for key columns
        for col in ["volume", "close"]:
            if col in state_data.columns:
                col_metrics = calculate_moments(state_data[col])
                metrics[f"{col}_mean"] = col_metrics["mean"]
                metrics[f"{col}_std"] = col_metrics["std"]
        
        result[int(state)] = metrics
    
    return result


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