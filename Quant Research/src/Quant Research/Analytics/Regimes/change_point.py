"""
Change point detection for market regime identification.

This module uses the ruptures library to detect abrupt changes in time series
that indicate shifts between market regimes.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import ruptures as rpt

from quant_research.core.models import Signal
from quant_research.analytics.common.base import SignalGenerator
from quant_research.analytics.common.data_prep import prepare_features, calculate_returns
from quant_research.analytics.common.validation import validate_numeric_param, validate_dataframe
from quant_research.analytics.common.statistics import calculate_moments
from quant_research.analytics.regimes.base import RegimeDetectorBase, calculate_regime_metrics

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Data Structures
# --------------------------------------------------------------------------------------

@dataclass
class ChangePointParams:
    """Parameters for change point detection and signal generation."""
    # Detection algorithm parameters
    method: str = "pelt"            # Detection algorithm ('pelt', 'window', 'binseg', 'dynp', 'bottomup')
    model: str = "rbf"              # Cost model ('l1', 'l2', 'rbf', 'linear', 'normal', 'ar')
    min_size: int = 20              # Minimum segment length
    penalty: Optional[float] = None # Penalty value (higher = fewer changes)
    n_bkps: Optional[int] = 10      # Number of breakpoints to detect
    jump: int = 5                   # Jump value for faster computation
    
    # Signal generation parameters
    signal_expiry: int = 5          # How long the signal remains valid
    
    # Input processing parameters
    features: List[str] = None      # Features to use (default: ["returns", "volatility"])
    window: int = 20                # Window for feature calculation
    
    def __post_init__(self):
        """Initialize default lists."""
        if self.features is None:
            self.features = ["returns", "volatility"]
        
        # Validate parameters
        self.method = self.method.lower()
        self.model = self.model.lower()
        self.min_size = validate_numeric_param(self.min_size, "min_size", min_value=2)
        self.jump = validate_numeric_param(self.jump, "jump", min_value=1)
        if self.n_bkps is not None:
            self.n_bkps = validate_numeric_param(self.n_bkps, "n_bkps", min_value=1)


class ChangePointDetector(RegimeDetectorBase, SignalGenerator):
    """
    Regime detector using change point detection algorithms.
    
    This implementation uses the ruptures library to detect abrupt
    changes in time series data that indicate regime shifts.
    """
    
    def __init__(self, params: Optional[ChangePointParams] = None):
        """
        Initialize the change point detector.
        
        Args:
            params: Parameters for detection and signal generation
        """
        SignalGenerator.__init__(self)
        self.params = params or ChangePointParams()
        self.method = self.params.method
        self.model = self.params.model
        self.min_size = self.params.min_size
        self.penalty = self.params.penalty
        self.n_bkps = self.params.n_bkps
        self.change_points = None
        self.algorithm = None
        self.segment_costs = None
    
    def fit(self, X: pd.DataFrame, **kwargs) -> Any:
        """
        Fit a change point detection model on the provided data.
        
        Args:
            X: Feature matrix (standardized)
            **kwargs: Additional parameters
            
        Returns:
            Fitted algorithm object or detected change points
        """
        # Override with parameters from kwargs if provided
        method = kwargs.get("method", self.method)
        model = kwargs.get("model", self.model)
        min_size = kwargs.get("min_size", self.min_size)
        penalty = kwargs.get("penalty", self.penalty)
        n_bkps = kwargs.get("n_bkps", self.n_bkps)
        jump = kwargs.get("jump", self.params.jump)
        
        # Convert DataFrame to numpy array if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Validate input data
        if len(X_array) < min_size * 2:
            logger.warning(f"Insufficient data for change point detection: {len(X_array)} < {min_size * 2}")
            self.change_points = []
            return None
        
        # Select and configure algorithm
        if method == "pelt":
            self.algorithm = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(X_array)
            # For PELT, either penalty or n_bkps must be provided
            if penalty is not None:
                self.change_points = self.algorithm.predict(pen=penalty)
            elif n_bkps is not None:
                self.change_points = self.algorithm.predict(n_bkps=n_bkps)
            else:
                # Default penalty based on BIC criterion
                penalty = 2 * X_array.shape[1] * np.log(X_array.shape[0])
                self.change_points = self.algorithm.predict(pen=penalty)
                
        elif method == "window":
            self.algorithm = rpt.Window(width=min_size, model=model).fit(X_array)
            self.change_points = self.algorithm.predict(pen=penalty) if penalty is not None else self.algorithm.predict(n_bkps=n_bkps)
            
        elif method == "binseg":
            self.algorithm = rpt.Binseg(model=model, min_size=min_size, jump=jump).fit(X_array)
            self.change_points = self.algorithm.predict(pen=penalty) if penalty is not None else self.algorithm.predict(n_bkps=n_bkps)
            
        elif method == "bottomup":
            self.algorithm = rpt.BottomUp(model=model, min_size=min_size, jump=jump).fit(X_array)
            self.change_points = self.algorithm.predict(pen=penalty) if penalty is not None else self.algorithm.predict(n_bkps=n_bkps)
            
        elif method == "dynp":
            self.algorithm = rpt.Dynp(model=model, min_size=min_size, jump=jump).fit(X_array)
            self.change_points = self.algorithm.predict(n_bkps=n_bkps)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # The last value is always the length of the array, which is not a true change point
        if self.change_points and self.change_points[-1] == len(X_array):
            self.change_points = self.change_points[:-1]
        
        logger.info(f"Detected {len(self.change_points)} change points using {method} method")
        
        # Calculate costs for each segment as a measure of quality
        cost_func = rpt.costs.cost_factory(model=model)
        
        start_idx = 0
        self.segment_costs = []
        all_points = self.change_points + [len(X_array)]
        
        for end_idx in all_points:
            if end_idx > start_idx:
                segment = X_array[start_idx:end_idx]
                cost = cost_func(segment)
                self.segment_costs.append(cost)
                start_idx = end_idx
        
        return self.algorithm
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, None]:
        """
        Generate regime states from change points.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of regime states and None (no probabilities for change points)
        """
        if self.change_points is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert change points to regime states
        states = np.zeros(len(X), dtype=int)
        
        # Start with regime 0
        current_regime = 0
        
        # Set regime for each segment
        start_idx = 0
        for change_point in self.change_points:
            if change_point < len(states):
                states[start_idx:change_point] = current_regime
                start_idx = change_point
                current_regime += 1
        
        # Set the last segment
        states[start_idx:] = current_regime
        
        return states, None
    
    def generate_signals(
        self, 
        df: pd.DataFrame, 
        states: np.ndarray,
        index: pd.Index,
        **kwargs
    ) -> List[Signal]:
        """
        Generate signals based on detected regimes.
        
        Args:
            df: Original dataframe
            states: Detected regime states
            index: Index from dataframe
            **kwargs: Additional parameters
            
        Returns:
            List of Signal objects
        """
        if self.change_points is None:
            raise ValueError("Model must be fitted before generating signals")
        
        signals = []
        
        # Create change point indices
        change_point_indices = [index[cp] for cp in self.change_points if cp < len(index)]
        
        # Create signals at change points
        for i, cp_timestamp in enumerate(change_point_indices):
            # Get regime states before and after change point
            idx_pos = index.get_loc(cp_timestamp)
            
            # Make sure we're not at the edge
            if idx_pos > 0 and idx_pos < len(states) - 1:
                prev_regime = int(states[idx_pos - 1])
                new_regime = int(states[idx_pos])
                
                # Create signal
                signals.append(
                    Signal(
                        timestamp=cp_timestamp,
                        signal_type=f"regime_change",
                        value=float(new_regime),
                        confidence=1.0,  # Change points are deterministic
                        metadata={
                            "previous_regime": prev_regime,
                            "new_regime": new_regime,
                            "source": f"change_point_{self.method}",
                        }
                    )
                )
        
        return signals
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the fitted change point model.
        
        Returns:
            Dictionary with model metadata
        """
        if self.change_points is None:
            return {
                "model_type": "change_point",
                "fitted": False
            }
        
        return {
            "model_type": "change_point",
            "fitted": True,
            "method": self.method,
            "cost_model": self.model,
            "n_regimes": len(self.change_points) + 1,
            "min_size": self.min_size,
            "penalty": self.penalty,
            "n_bkps": self.n_bkps,
            "avg_segment_cost": np.mean(self.segment_costs) if self.segment_costs else None,
        }
    
    def _generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of the SignalGenerator interface to generate signals.
        
        Args:
            df: Input DataFrame with market data
            
        Returns:
            DataFrame with signal data
        """
        # Validate input data
        df, errors = validate_dataframe(df, min_rows=self.params.min_size * 2)
        if errors:
            logger.warning(f"Data validation warnings: {errors}")
        
        # Prepare features using common functionality
        features, scaler = prepare_features(
            df,
            features=self.params.features,
            window=self.params.window,
            add_derived=True
        )
        
        # Fit the model
        self.fit(features)
        
        # Predict regimes
        states, _ = self.predict(features)
        
        # Calculate regime metrics
        regime_df = calculate_regime_metrics(states, features.index)
        
        # Generate signals
        signals = self.generate_signals(df, states, features.index)
        
        # Merge regime data with original data
        result = df.join(regime_df, how="left")
        
        # Add model metadata
        for key, value in self.get_metadata().items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                result[key] = value
        
        # Add change points as boolean column
        is_change_point = pd.Series(False, index=result.index)
        
        if self.change_points is not None:
            change_point_indices = [features.index[cp] for cp in self.change_points if cp < len(features.index)]
            is_change_point.loc[change_point_indices] = True
        
        result["is_change_point"] = is_change_point
        
        return result


def generate_signal(
    df: pd.DataFrame,
    method: str = "pelt",
    model: str = "rbf",
    features: List[str] = None,
    window: int = 20,
    min_size: int = 20,
    penalty: Optional[float] = None,
    n_bkps: Optional[int] = 10,
    jump: int = 5,
    **kwargs
) -> pd.DataFrame:
    """
    Generate market regime signals using change point detection.
    
    This is a convenience function that wraps the ChangePointDetector class.
    
    Args:
        df: Input dataframe with market data
        method: Detection method: 'pelt', 'window', 'binseg', 'dynp', 'bottomup'
        model: Cost model: 'l1', 'l2', 'rbf', 'linear', 'normal', 'ar'
        features: Features to use for regime detection
        window: Lookback window for derived features
        min_size: Minimum segment length
        penalty: Penalty value (higher = fewer changes)
        n_bkps: Number of breakpoints to detect (alternative to penalty)
        jump: Jump value for faster computation
        **kwargs: Additional parameters for detector
        
    Returns:
        DataFrame with regime states and change points
    """
    # Create parameter object
    params = ChangePointParams(
        method=method,
        model=model,
        min_size=min_size,
        penalty=penalty,
        n_bkps=n_bkps,
        jump=jump,
        features=features if features is not None else ["returns", "volatility"],
        window=window
    )
    
    # Create detector
    detector = ChangePointDetector(params)
    
    # Generate signals
    result = detector.generate_signal(df)
    
    return result


def online_detection(
    df: pd.DataFrame,
    window_size: int = 100,
    model: str = "rbf",
    features: List[str] = None,
    threshold: float = 5.0,
    **kwargs
) -> pd.DataFrame:
    """
    Perform online change point detection for streaming data.
    
    Args:
        df: Input dataframe with market data
        window_size: Size of the sliding window
        model: Cost model: 'l1', 'l2', 'rbf', 'linear', 'normal'
        features: Features to use for regime detection
        threshold: Threshold for detecting a change
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with detected change points
    """
    # Use default features if none provided
    if features is None:
        features = ["returns", "volatility"]
    
    # Prepare features
    X, _ = prepare_features(df, features=features, window=window_size//5)
    X_array = X.values
    
    # Initialize detector
    detector = rpt.Window(width=window_size, model=model)
    
    # Online detection
    online_changes = []
    scores = []
    
    for i in range(window_size, len(X_array)):
        # Extract window
        current_window = X_array[i-window_size:i]
        
        # Compute score
        score = detector.score(current_window)
        scores.append(score)
        
        # Detect change if score exceeds threshold
        if score > threshold:
            online_changes.append(i)
    
    # Create result
    result = df.copy()
    
    # Add change point indicator
    is_change = pd.Series(False, index=result.index)
    change_indices = [X.index[cp] for cp in online_changes if cp < len(X.index)]
    is_change.loc[change_indices] = True
    result["is_change_point"] = is_change
    
    # Add scores
    result["change_score"] = np.nan
    score_indices = X.index[window_size:min(len(X), len(scores) + window_size)]
    result.loc[score_indices, "change_score"] = scores
    
    return result


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    from quant_research.analytics.regimes.base import state_analysis
    
    # Download some test data
    data = yf.download("SPY", start="2020-01-01", end="2022-12-31")
    
    # Generate regime signals using change point detection
    result = generate_signal(
        data, 
        method="pelt",
        model="rbf",
        n_bkps=10,  # Detect 10 regime changes
    )
    
    # Analyze regimes
    analysis = state_analysis(data, result)
    
    print("Regime Analysis:")
    for state, metrics in analysis.items():
        print(f"State {state}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    # Example of online detection for streaming data
    online_result = online_detection(
        data.iloc[:200],  # Use subset for example
        window_size=50,
        threshold=3.0,
    )
    
    print(f"Online detection found {online_result['is_change_point'].sum()} change points")