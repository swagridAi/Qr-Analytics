"""
Change point detection for market regime identification.

This module uses the ruptures library to detect abrupt changes in time series
that indicate shifts between market regimes.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import ruptures as rpt

from quant_research.core.models import Signal
from quant_research.analytics.regimes.base import (
    RegimeDetectorBase, 
    prepare_features,
    state_analysis
)


logger = logging.getLogger(__name__)


class ChangePointDetector(RegimeDetectorBase):
    """
    Regime detector using change point detection algorithms.
    
    This implementation uses the ruptures library to detect abrupt
    changes in time series data that indicate regime shifts.
    """
    
    def __init__(self):
        self.method = None
        self.model = None
        self.min_size = None
        self.penalty = None
        self.n_bkps = None
        self.change_points = None
        self.algorithm = None
        self.segment_costs = None
    
    def fit(self, X: pd.DataFrame, **kwargs) -> Any:
        """
        Fit a change point detection model on the provided data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (standardized)
        **kwargs
            Additional parameters including:
            - method: Detection algorithm ('pelt', 'window', etc.)
            - model: Cost model ('l1', 'l2', 'rbf', etc.)
            - min_size: Minimum segment length
            - penalty: Penalty value for model complexity
            - n_bkps: Number of breakpoints to detect
            
        Returns
        -------
        Any
            Fitted algorithm object or detected change points
        """
        # Extract parameters
        self.method = kwargs.get("method", "pelt")
        self.model = kwargs.get("model", "rbf")
        self.min_size = kwargs.get("min_size", 20)
        self.penalty = kwargs.get("penalty")
        self.n_bkps = kwargs.get("n_bkps", 10)
        jump = kwargs.get("jump", 5)
        
        # Convert DataFrame to numpy array if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Select and configure algorithm
        if self.method == "pelt":
            self.algorithm = rpt.Pelt(model=self.model, min_size=self.min_size, jump=jump).fit(X_array)
            # For PELT, either penalty or n_bkps must be provided
            if self.penalty is not None:
                self.change_points = self.algorithm.predict(pen=self.penalty)
            elif self.n_bkps is not None:
                self.change_points = self.algorithm.predict(n_bkps=self.n_bkps)
            else:
                # Default penalty based on BIC criterion
                self.penalty = 2 * X_array.shape[1] * np.log(X_array.shape[0])
                self.change_points = self.algorithm.predict(pen=self.penalty)
                
        elif self.method == "window":
            self.algorithm = rpt.Window(width=self.min_size, model=self.model).fit(X_array)
            self.change_points = self.algorithm.predict(pen=self.penalty) if self.penalty is not None else self.algorithm.predict(n_bkps=self.n_bkps)
            
        elif self.method == "binseg":
            self.algorithm = rpt.Binseg(model=self.model, min_size=self.min_size, jump=jump).fit(X_array)
            self.change_points = self.algorithm.predict(pen=self.penalty) if self.penalty is not None else self.algorithm.predict(n_bkps=self.n_bkps)
            
        elif self.method == "bottomup":
            self.algorithm = rpt.BottomUp(model=self.model, min_size=self.min_size, jump=jump).fit(X_array)
            self.change_points = self.algorithm.predict(pen=self.penalty) if self.penalty is not None else self.algorithm.predict(n_bkps=self.n_bkps)
            
        elif self.method == "dynp":
            self.algorithm = rpt.Dynp(model=self.model, min_size=self.min_size, jump=jump).fit(X_array)
            self.change_points = self.algorithm.predict(n_bkps=self.n_bkps)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # The last value is always the length of the array, which is not a true change point
        if self.change_points and self.change_points[-1] == len(X_array):
            self.change_points = self.change_points[:-1]
        
        logger.info(f"Detected {len(self.change_points)} change points using {self.method} method")
        
        # Calculate costs for each segment as a measure of quality
        cost_func = rpt.costs.cost_factory(model=self.model)
        
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
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        Tuple[np.ndarray, None]
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
        
        Returns
        -------
        Dict[str, Any]
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


def generate_signal(
    df: pd.DataFrame,
    method: str = "pelt",
    model: str = "rbf",
    features: List[str] = ["returns", "volatility"],
    window: int = 20,
    min_size: int = 20,
    penalty: Optional[float] = None,
    n_bkps: Optional[int] = 10,
    jump: int = 5,
    add_derived_features: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate market regime signals using change point detection.
    
    This is a convenience function that wraps the ChangePointDetector class.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with market data
    method : str
        Detection method: 'pelt', 'window', 'binseg', 'dynp', 'bottomup'
    model : str
        Cost model: 'l1', 'l2', 'rbf', 'linear', 'normal', 'ar'
    features : List[str]
        Features to use for regime detection
    window : int
        Lookback window for derived features
    min_size : int
        Minimum segment length
    penalty : Optional[float]
        Penalty value (higher = fewer changes)
    n_bkps : Optional[int]
        Number of breakpoints to detect (alternative to penalty)
    jump : int
        Jump value for faster computation
    add_derived_features : bool
        Whether to add derived features from raw data
    **kwargs
        Additional parameters for ruptures algorithms
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regime states and change points
    """
    detector = ChangePointDetector()
    
    result = detector.generate_signal(
        df=df,
        features=features,
        window=window,
        method=method,
        model=model,
        min_size=min_size,
        penalty=penalty,
        n_bkps=n_bkps,
        jump=jump,
        add_derived_features=add_derived_features,
        **kwargs
    )
    
    # Add change points as boolean column
    is_change_point = pd.Series(False, index=result.index)
    
    if detector.change_points is not None:
        # Get feature dataframe to map indices
        X, _ = prepare_features(
            df,
            features=features,
            window=window,
            add_derived=add_derived_features,
        )
        
        change_point_indices = [X.index[cp] for cp in detector.change_points if cp < len(X.index)]
        is_change_point.loc[change_point_indices] = True
    
    result["is_change_point"] = is_change_point
    
    return result


def online_detection(
    df: pd.DataFrame,
    window_size: int = 100,
    model: str = "rbf",
    features: List[str] = ["returns", "volatility"],
    threshold: float = 5.0,
    **kwargs
) -> pd.DataFrame:
    """
    Perform online change point detection for streaming data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with market data
    window_size : int
        Size of the sliding window
    model : str
        Cost model: 'l1', 'l2', 'rbf', 'linear', 'normal'
    features : List[str]
        Features to use for regime detection
    threshold : float
        Threshold for detecting a change
    **kwargs
        Additional parameters
        
    Returns
    -------
    pd.DataFrame
        DataFrame with detected change points
    """
    # Prepare features
    X, _ = prepare_features(df, features=features)
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