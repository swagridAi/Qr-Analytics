"""
GARCH Volatility Models

This module implements GARCH-type models for volatility forecasting as part of
the quantitative research analytics pipeline.

The main function to use is generate_signal which accepts price data and returns
volatility forecasts in the standard signal format.

Example:
    >>> from quant_research.analytics.volatility import garch
    >>> signals = garch.generate_signal(price_df, p=1, q=1, dist='normal')
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

from arch import arch_model
from arch.univariate import GARCH, EGARCH, GJR_GARCH

# Import utilities and models
from quant_research.analytics.volatility.utils import (
    calculate_returns, prepare_dataframe, calculate_confidence_interval,
    annualize_volatility, format_signals, save_signals
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants and Default Parameters
# -----------------------------------------------------------------------------

# Model type mapping
MODEL_TYPES = {
    'garch': GARCH,
    'egarch': EGARCH,
    'gjr_garch': GJR_GARCH
}

# Default parameters
DEFAULT_PARAMS = {
    'model_type': 'garch',  # One of 'garch', 'egarch', 'gjr_garch'
    'p': 1,                # GARCH lag order
    'q': 1,                # ARCH lag order
    'dist': 'normal',      # Error distribution ('normal', 't', 'skewt')
    'window': 252,         # Rolling window size
    'horizon': 5,          # Forecast horizon in days
    'return_type': 'log',  # Either 'log' or 'pct' for return calculation
    'target_column': 'close',  # Column to use for pricing
    'min_periods': 20,     # Minimum periods for fitting
    'alpha': 0.05,         # Significance level for CI
    'use_mean': True,      # Include mean in model
    'vol_scaling': True,   # Apply annualization scaling
    'trading_days': 252,   # Trading days per year for annualization
}

# -----------------------------------------------------------------------------
# Core GARCH Model Functions
# -----------------------------------------------------------------------------

def fit_garch_model(returns: pd.Series, 
                   model_type: str = 'garch',
                   p: int = 1, 
                   q: int = 1, 
                   dist: str = 'normal', 
                   use_mean: bool = True) -> Any:
    """
    Fit a GARCH model to return data.
    
    Args:
        returns: Series with return data
        model_type: Type of GARCH model
        p: GARCH lag order
        q: ARCH lag order
        dist: Error distribution
        use_mean: Whether to include mean in model
        
    Returns:
        Fitted arch model
    """
    # Handle missing values
    returns = returns.dropna()
    
    if len(returns) == 0:
        raise ValueError("No valid returns data after dropping NaN values")
    
    # Create model
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}. Available models: {list(MODEL_TYPES.keys())}")
    
    # Fit model
    model = arch_model(
        returns * 100,  # Scale for numerical stability
        vol=model_type,
        p=p, 
        q=q, 
        dist=dist,
        mean='Constant' if use_mean else 'Zero'
    )
    
    try:
        result = model.fit(disp='off', show_warning=False)
        return result
    except Exception as e:
        logger.warning(f"GARCH model fitting failed: {e}")
        raise RuntimeError(f"Failed to fit GARCH model: {e}")


def forecast_volatility(model_result: Any, 
                       horizon: int = 5, 
                       alpha: float = 0.05) -> Dict[str, pd.Series]:
    """
    Generate volatility forecasts from fitted model.
    
    Args:
        model_result: Fitted arch model result
        horizon: Forecast horizon
        alpha: Significance level for CI
        
    Returns:
        Dictionary with volatility forecast and confidence intervals
    """
    try:
        forecast = model_result.forecast(horizon=horizon, reindex=False)
        
        # Extract conditional variance forecasts
        variance = forecast.variance.iloc[-1]
        
        # Convert variance to volatility (std dev)
        volatility = np.sqrt(variance) / 100  # Undo the scaling
        
        # Calculate confidence intervals
        z_score = np.abs(np.percentile(np.random.normal(0, 1, 10000), 100 * (1 - alpha/2)))
        lower_ci = volatility - z_score * volatility * 0.25  # Approximate std error
        upper_ci = volatility + z_score * volatility * 0.25
        
        # Create forecast dictionary
        result = {
            'volatility': volatility,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        }
        
        return result
    
    except Exception as e:
        logger.warning(f"Volatility forecasting failed: {e}")
        raise RuntimeError(f"Failed to forecast volatility: {e}")


def find_optimal_params(returns: pd.Series, 
                       model_types: List[str] = None, 
                       p_range: range = range(1, 3), 
                       q_range: range = range(1, 3),
                       dists: List[str] = None) -> Dict[str, Any]:
    """
    Find optimal GARCH parameters using BIC.
    
    Args:
        returns: Series with return data
        model_types: List of model types to try
        p_range: Range of p values to try
        q_range: Range of q values to try
        dists: List of distributions to try
        
    Returns:
        Dictionary with optimal parameters
    """
    if model_types is None:
        model_types = ['garch']
    
    if dists is None:
        dists = ['normal', 't']
    
    best_bic = np.inf
    best_params = {}
    
    for model_type in model_types:
        for p in p_range:
            for q in q_range:
                for dist in dists:
                    try:
                        model = arch_model(
                            returns * 100,
                            vol=model_type,
                            p=p,
                            q=q,
                            dist=dist,
                            rescale=True
                        )
                        result = model.fit(disp='off', show_warning=False)
                        
                        if result.bic < best_bic:
                            best_bic = result.bic
                            best_params = {
                                'model_type': model_type,
                                'p': p,
                                'q': q,
                                'dist': dist
                            }
                    
                    except Exception as e:
                        logger.debug(f"Failed to fit model with {model_type}, p={p}, q={q}, dist={dist}: {e}")
                        continue
    
    if not best_params:
        logger.warning("Could not find optimal parameters, using defaults")
        best_params = {
            'model_type': 'garch',
            'p': 1,
            'q': 1,
            'dist': 'normal'
        }
    
    return best_params


# -----------------------------------------------------------------------------
# Rolling Forecast Implementation
# -----------------------------------------------------------------------------

def rolling_garch_forecast(df: pd.DataFrame, 
                          params: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate rolling GARCH forecasts.
    
    Args:
        df: DataFrame with price data
        params: Dictionary with model parameters
        
    Returns:
        DataFrame with volatility forecasts
    """
    # Extract parameters with defaults
    p = params.get('p', DEFAULT_PARAMS['p'])
    q = params.get('q', DEFAULT_PARAMS['q'])
    window = params.get('window', DEFAULT_PARAMS['window'])
    horizon = params.get('horizon', DEFAULT_PARAMS['horizon'])
    return_type = params.get('return_type', DEFAULT_PARAMS['return_type'])
    target_column = params.get('target_column', DEFAULT_PARAMS['target_column'])
    model_type = params.get('model_type', DEFAULT_PARAMS['model_type'])
    dist = params.get('dist', DEFAULT_PARAMS['dist'])
    alpha = params.get('alpha', DEFAULT_PARAMS['alpha'])
    use_mean = params.get('use_mean', DEFAULT_PARAMS['use_mean'])
    min_periods = params.get('min_periods', DEFAULT_PARAMS['min_periods'])
    
    # Calculate returns
    returns = calculate_returns(df, target_column, return_type)
    
    # Initialize results
    volatility = pd.Series(index=df.index, dtype=float)
    lower_ci = pd.Series(index=df.index, dtype=float)
    upper_ci = pd.Series(index=df.index, dtype=float)
    
    # Rolling forecast
    for i in range(window, len(df)):
        try:
            # Get window
            window_returns = returns.iloc[i-window:i]
            
            if len(window_returns.dropna()) < min_periods:
                continue
                
            # Fit model
            model_result = fit_garch_model(
                window_returns, 
                model_type=model_type,
                p=p, 
                q=q, 
                dist=dist,
                use_mean=use_mean
            )
            
            # Generate forecast
            forecast = forecast_volatility(
                model_result, 
                horizon=horizon, 
                alpha=alpha
            )
            
            # Store results (use the last day forecast)
            current_idx = df.index[i]
            volatility[current_idx] = forecast['volatility'].iloc[-1] if isinstance(forecast['volatility'], pd.Series) else forecast['volatility']
            lower_ci[current_idx] = forecast['lower_ci'].iloc[-1] if isinstance(forecast['lower_ci'], pd.Series) else forecast['lower_ci']
            upper_ci[current_idx] = forecast['upper_ci'].iloc[-1] if isinstance(forecast['upper_ci'], pd.Series) else forecast['upper_ci']
            
        except Exception as e:
            logger.warning(f"Failed to generate forecast for window ending at {df.index[i]}: {e}")
            continue
    
    # Annualize volatility (if daily data) and apply scaling if needed
    if params.get('vol_scaling', DEFAULT_PARAMS['vol_scaling']):
        trading_days = params.get('trading_days', DEFAULT_PARAMS['trading_days'])
        volatility = annualize_volatility(volatility, df, trading_days)
        lower_ci = annualize_volatility(lower_ci, df, trading_days)
        upper_ci = annualize_volatility(upper_ci, df, trading_days)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'volatility': volatility,
        'volatility_lower': lower_ci,
        'volatility_upper': upper_ci
    })
    
    return result


# -----------------------------------------------------------------------------
# Public API Functions
# -----------------------------------------------------------------------------

def generate_signal(df: pd.DataFrame, **params) -> Union[pd.DataFrame, List[Signal]]:
    """
    Generate volatility signals from price data.
    
    This is the main function that should be called by the analytics pipeline.
    
    Args:
        df: DataFrame with price data (must have at least timestamp and close columns)
        **params: Dictionary with model parameters
            - model_type: Type of GARCH model
            - p: GARCH lag order
            - q: ARCH lag order
            - dist: Error distribution
            - window: Rolling window size
            - horizon: Forecast horizon
            - return_type: Return calculation method
            - target_column: Column to use for pricing
            - min_periods: Minimum periods for fitting
            - alpha: Significance level for CI
            - use_mean: Include mean in model
            - vol_scaling: Apply annualization scaling
            - optimize_params: Whether to find optimal parameters
            - output_file: Path to save the signals (optional)
        
    Returns:
        DataFrame with volatility signals or list of Signal objects
    """
    logger.info(f"Generating GARCH volatility signals with parameters: {params}")
    
    # Prepare DataFrame
    target_column = params.get('target_column', DEFAULT_PARAMS['target_column'])
    df = prepare_dataframe(df, target_column)
    
    # Merge parameters with defaults
    model_params = DEFAULT_PARAMS.copy()
    model_params.update(params)
    
    # Optionally optimize parameters
    if model_params.get('optimize_params', False):
        returns = calculate_returns(
            df, 
            model_params['target_column'], 
            model_params['return_type']
        )
        optimal_params = find_optimal_params(
            returns.dropna(),
            model_types=model_params.get('model_types', ['garch', 'gjr_garch']),
            p_range=range(1, model_params.get('max_p', 3)),
            q_range=range(1, model_params.get('max_q', 3)),
            dists=model_params.get('dists', ['normal', 't'])
        )
        
        logger.info(f"Optimal parameters found: {optimal_params}")
        model_params.update(optimal_params)
    
    # Generate forecasts
    forecasts = rolling_garch_forecast(df, model_params)
    
    # Format as signals
    model_name = f"{model_params['model_type'].upper()}({model_params['p']},{model_params['q']})"
    signals = format_signals(forecasts, df, model_params, 'volatility_forecast', model_name)
    
    # Save and/or return as objects
    return save_signals(signals, model_params, model_params.get('as_objects', False))


# -----------------------------------------------------------------------------
# Convenience Functions for Specific Models
# -----------------------------------------------------------------------------

def generate_garch_signal(df: pd.DataFrame, **params) -> Union[pd.DataFrame, List[Signal]]:
    """
    Generate standard GARCH volatility signals.
    
    Args:
        df: DataFrame with price data
        **params: Model parameters
        
    Returns:
        Volatility signals
    """
    params['model_type'] = 'garch'
    return generate_signal(df, **params)


def generate_egarch_signal(df: pd.DataFrame, **params) -> Union[pd.DataFrame, List[Signal]]:
    """
    Generate EGARCH volatility signals for asymmetric volatility.
    
    Args:
        df: DataFrame with price data
        **params: Model parameters
        
    Returns:
        Volatility signals
    """
    params['model_type'] = 'egarch'
    return generate_signal(df, **params)


def generate_gjr_garch_signal(df: pd.DataFrame, **params) -> Union[pd.DataFrame, List[Signal]]:
    """
    Generate GJR-GARCH volatility signals for leverage effects.
    
    Args:
        df: DataFrame with price data
        **params: Model parameters
        
    Returns:
        Volatility signals
    """
    params['model_type'] = 'gjr_garch'
    return generate_signal(df, **params)


# -----------------------------------------------------------------------------
# Command-line Interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate GARCH volatility signals")
    parser.add_argument("--input", required=True, help="Input price data file (CSV or Parquet)")
    parser.add_argument("--output", required=True, help="Output signals file (Parquet)")
    parser.add_argument("--model-type", default="garch", choices=["garch", "egarch", "gjr_garch"], 
                       help="GARCH model type")
    parser.add_argument("--p", type=int, default=1, help="GARCH lag order")
    parser.add_argument("--q", type=int, default=1, help="ARCH lag order")
    parser.add_argument("--window", type=int, default=252, help="Rolling window size")
    parser.add_argument("--optimize", action="store_true", help="Optimize model parameters")
    
    args = parser.parse_args()
    
    # Load data
    if args.input.endswith('.csv'):
        data = pd.read_csv(args.input, parse_dates=['timestamp']).set_index('timestamp')
    else:
        data = pd.read_parquet(args.input)
    
    # Generate signals
    signals = generate_signal(
        data, 
        model_type=args.model_type,
        p=args.p,
        q=args.q,
        window=args.window,
        optimize_params=args.optimize,
        output_file=args.output
    )
    
    print(f"Generated {len(signals)} volatility signals")