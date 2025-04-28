"""
Correlation and Cointegration Analysis

This module provides functions for analyzing relationships between time series,
including various correlation measures and cointegration tests.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, coint
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.statistics.correlation")

def calculate_correlation(
    x: Union[pd.Series, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    method: str = 'pearson',
    min_periods: Optional[int] = None
) -> float:
    """
    Calculate correlation between two series.
    
    Args:
        x: First time series
        y: Second time series
        method: Correlation method ('pearson', 'spearman', 'kendall')
        min_periods: Minimum number of valid observations
        
    Returns:
        Correlation coefficient
        
    Raises:
        ValueError: If an invalid method is specified
    """
    # Convert numpy arrays to pandas Series if necessary
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    # Align series
    x, y = x.align(y, join='inner')
    
    # Check we have enough data
    if len(x) == 0:
        logger.warning("No overlapping data for correlation calculation")
        return np.nan
    
    # Default min_periods if not specified
    if min_periods is None:
        min_periods = min(10, len(x) // 2)
    
    # Calculate correlation
    if method == 'pearson':
        return x.corr(y, method='pearson', min_periods=min_periods)
    elif method == 'spearman':
        return x.corr(y, method='spearman', min_periods=min_periods)
    elif method == 'kendall':
        return x.corr(y, method='kendall', min_periods=min_periods)
    else:
        raise ValueError(f"Invalid correlation method: {method}. Use 'pearson', 'spearman', or 'kendall'")


def rolling_correlation(
    x: pd.Series,
    y: pd.Series,
    window: int = 60,
    method: str = 'pearson',
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate rolling correlation between two series.
    
    Args:
        x: First time series
        y: Second time series
        window: Rolling window size
        method: Correlation method ('pearson', 'spearman', 'kendall')
        min_periods: Minimum number of valid observations
        
    Returns:
        Series with rolling correlation values
        
    Raises:
        ValueError: If an invalid method is specified
    """
    # Align series
    x, y = x.align(y, join='inner')
    
    # Default min_periods if not specified
    if min_periods is None:
        min_periods = min(10, window // 2)
    
    # Create DataFrame with both series
    df = pd.DataFrame({'x': x, 'y': y})
    
    # Calculate rolling correlation
    if method == 'pearson':
        return df['x'].rolling(window=window, min_periods=min_periods).corr(df['y'])
    elif method == 'spearman':
        # Calculate rank series
        x_rank = x.rank()
        y_rank = y.rank()
        df_rank = pd.DataFrame({'x_rank': x_rank, 'y_rank': y_rank})
        return df_rank['x_rank'].rolling(window=window, min_periods=min_periods).corr(df_rank['y_rank'])
    elif method == 'kendall':
        # For Kendall, we compute for each window manually
        result = pd.Series(index=df.index, dtype=float)
        for i in range(len(df) - window + 1):
            if i + window > len(df):
                break
            window_data = df.iloc[i:i+window]
            if len(window_data.dropna()) >= min_periods:
                tau, _ = stats.kendalltau(window_data['x'].dropna(), window_data['y'].dropna())
                result.iloc[i+window-1] = tau
        return result
    else:
        raise ValueError(f"Invalid correlation method: {method}. Use 'pearson', 'spearman', or 'kendall'")


def cross_correlation(
    x: pd.Series,
    y: pd.Series,
    max_lags: int = 10,
    normalize: bool = True
) -> Tuple[pd.Series, int]:
    """
    Calculate cross-correlation function (CCF) to find lead-lag relationships.
    
    Args:
        x: First time series
        y: Second time series
        max_lags: Maximum number of lags to compute
        normalize: Whether to normalize (output in [-1, 1])
        
    Returns:
        Series with CCF values and lag with maximum correlation
        
    Notes:
        - Positive lag: x leads y
        - Negative lag: y leads x
    """
    # Align series
    x, y = x.align(y, join='inner')
    
    # Calculate cross-correlation
    ccf_values = {}
    
    for lag in range(-max_lags, max_lags + 1):
        if lag < 0:
            # y is shifted |lag| periods forward (y leads)
            corr = x.corr(y.shift(lag))
        elif lag > 0:
            # x is shifted lag periods forward (x leads)
            corr = x.shift(-lag).corr(y)
        else:
            # Contemporaneous
            corr = x.corr(y)
        
        ccf_values[lag] = corr
    
    # Create Series with CCF values
    ccf_series = pd.Series(ccf_values)
    
    # Find lag with maximum correlation
    max_corr_lag = ccf_series.apply(abs).idxmax()
    
    return ccf_series, max_corr_lag


def test_cointegration(
    x: pd.Series,
    y: pd.Series,
    method: str = 'johansen',
    regression_method: str = 'ols',
    max_lags: int = None,
    trend: str = 'c',
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Test for cointegration between two time series.
    
    Args:
        x: First time series
        y: Second time series
        method: Test method ('engle-granger', 'johansen')
        regression_method: Regression method for Engle-Granger ('ols', 'ts')
        max_lags: Maximum lags for ADF test (None for automatic)
        trend: Type of trend ('c', 'ct', 'ctt', 'nc')
        significance_level: Significance level for hypothesis testing
        
    Returns:
        Dictionary with test results including:
            - is_cointegrated: Boolean indicating cointegration
            - p_value: P-value of the test
            - test_statistic: Test statistic
            - critical_values: Critical values
            - hedge_ratio: Hedge ratio (beta) for pair trading
            - half_life: Half-life of mean reversion (in periods)
            - spread: Cointegrated residual series
            
    Raises:
        ValueError: If an invalid method is specified
    """
    # Align series
    x, y = x.align(y, join='inner')
    
    # Check we have enough data
    if len(x) < 30:  # Arbitrary minimum for decent cointegration test
        logger.warning("Insufficient data for cointegration test (minimum 30 points)")
        return {
            'is_cointegrated': False,
            'p_value': np.nan,
            'test_statistic': np.nan,
            'critical_values': {},
            'hedge_ratio': np.nan,
            'half_life': np.nan,
            'spread': pd.Series(dtype=float)
        }
    
    # Create DataFrame for the regression
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    x = df['x']
    y = df['y']
    
    # Perform cointegration test
    if method == 'engle-granger':
        # Engle-Granger test (two-step approach)
        
        # Step 1: Estimate cointegrating relationship
        if regression_method == 'ols':
            # OLS regression
            model = sm.OLS(y, sm.add_constant(x))
            results = model.fit()
            const = results.params[0]
            hedge_ratio = results.params[1]
            spread = y - (const + hedge_ratio * x)
        else:  # ts (theil-sen)
            # Theil-Sen estimator (more robust to outliers)
            slope, intercept = stats.theilslopes(y, x)
            hedge_ratio = slope
            const = intercept
            spread = y - (const + hedge_ratio * x)
        
        # Step 2: Test for stationarity of residuals
        adf_result = adfuller(spread, maxlag=max_lags, regression=trend)
        test_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        
        is_cointegrated = p_value < significance_level
        
    elif method == 'johansen':
        # Johansen test (system approach)
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        
        # Prepare data for Johansen test
        data = pd.concat([x, y], axis=1).dropna()
        
        # Perform Johansen cointegration test
        try:
            # Determine lag order
            if max_lags is None:
                max_lags = int(np.ceil(12 * (len(data) / 100) ** (1 / 4)))
            
            # Johansen test
            result = coint_johansen(data, det_order=0, k_ar_diff=max_lags)
            
            # Extract results (first eigenvalue for two series)
            test_statistic = result.lr1[0]
            
            # Get critical values
            # 90%, 95%, and 99% critical values for trace test
            critical_values = {
                '90%': result.cvt[0, 0],
                '95%': result.cvt[0, 1],
                '99%': result.cvt[0, 2]
            }
            
            # Determine if cointegrated
            is_cointegrated = test_statistic > critical_values[f'{int((1-significance_level)*100)}%']
            
            # Calculate p-value (approximate via interpolation)
            # For simplicity, we'll use critical values to approximate
            if test_statistic > critical_values['99%']:
                p_value = 0.01
            elif test_statistic > critical_values['95%']:
                p_value = 0.05
            elif test_statistic > critical_values['90%']:
                p_value = 0.1
            else:
                p_value = 0.2  # Rough approximation
            
            # Get cointegrating vector
            if is_cointegrated:
                # The cointegrating vector is the eigenvector for the first eigenvalue
                coint_vector = result.evec[:, 0]
                hedge_ratio = -coint_vector[1] / coint_vector[0]
                const = 0  # Johansen without constant term
                spread = y - hedge_ratio * x
            else:
                hedge_ratio = np.nan
                const = np.nan
                spread = pd.Series(np.nan, index=x.index)
            
        except Exception as e:
            logger.warning(f"Johansen test failed: {e}")
            is_cointegrated = False
            test_statistic = np.nan
            p_value = np.nan
            critical_values = {}
            hedge_ratio = np.nan
            const = np.nan
            spread = pd.Series(np.nan, index=x.index)
            
    else:
        raise ValueError(f"Invalid cointegration test method: {method}. Use 'engle-granger' or 'johansen'")
    
    # Calculate half-life of mean reversion
    half_life = np.nan
    if is_cointegrated:
        # Regress change in spread on lag of spread
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        # Drop NaN values
        valid_data = pd.DataFrame({'diff': spread_diff, 'lag': spread_lag}).dropna()
        
        if len(valid_data) > 0:
            # Calculate half-life via AR(1) model
            model = sm.OLS(valid_data['diff'], valid_data['lag'])
            results = model.fit()
            
            # Coefficient should be negative for mean reversion
            if results.params[0] < 0:
                half_life = np.log(2) / abs(results.params[0])
            else:
                half_life = np.inf  # Not mean-reverting
    
    return {
        'is_cointegrated': is_cointegrated,
        'p_value': p_value,
        'test_statistic': test_statistic,
        'critical_values': critical_values,
        'hedge_ratio': hedge_ratio,
        'half_life': half_life,
        'spread': spread
    }


def granger_causality_test(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 5,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Test for Granger causality between two time series.
    
    Args:
        x: First time series
        y: Second time series
        max_lag: Maximum number of lags to test
        significance_level: Significance level for hypothesis testing
        
    Returns:
        Dictionary with test results including:
            - x_causes_y: Boolean indicating if x Granger-causes y
            - y_causes_x: Boolean indicating if y Granger-causes x
            - x_to_y_p_values: P-values for x causing y at different lags
            - y_to_x_p_values: P-values for y causing x at different lags
            - optimal_lag: Optimal lag based on information criteria
    """
    # Align series
    x, y = x.align(y, join='inner')
    
    # Create DataFrame for VAR model
    data = pd.DataFrame({'x': x, 'y': y}).dropna()
    
    # Drop rows with NaN values
    data = data.dropna()
    
    # Create results dictionary
    results = {
        'x_causes_y': False,
        'y_causes_x': False,
        'x_to_y_p_values': {},
        'y_to_x_p_values': {},
        'optimal_lag': 1
    }
    
    # Try different lag orders
    min_lag = min(12, max_lag)
    x_to_y_significant = False
    y_to_x_significant = False
    
    try:
        # Determine optimal lag using information criteria
        for lag in range(1, min_lag + 1):
            # Fit VAR model
            model = VAR(data)
            try:
                res = model.fit(lag)
                
                # Test Granger causality
                try:
                    # Test if x Granger-causes y
                    test_x_y = res.test_causality(caused='y', causing='x')
                    p_val_x_y = test_x_y['pvalue']
                    results['x_to_y_p_values'][lag] = p_val_x_y
                    
                    # Test if y Granger-causes x
                    test_y_x = res.test_causality(caused='x', causing='y')
                    p_val_y_x = test_y_x['pvalue']
                    results['y_to_x_p_values'][lag] = p_val_y_x
                    
                    # Check for statistical significance
                    if p_val_x_y < significance_level:
                        x_to_y_significant = True
                    
                    if p_val_y_x < significance_level:
                        y_to_x_significant = True
                
                except Exception as e:
                    logger.warning(f"Granger causality test failed for lag {lag}: {e}")
                    continue
            
            except Exception as e:
                logger.warning(f"Failed to fit VAR model for lag {lag}: {e}")
                continue
        
        # Find optimal lag using AIC
        model = VAR(data)
        lag_order = model.select_order(maxlags=min_lag)
        results['optimal_lag'] = lag_order.aic
        
        # Set final results
        results['x_causes_y'] = x_to_y_significant
        results['y_causes_x'] = y_to_x_significant
        
    except Exception as e:
        logger.warning(f"Granger causality test failed: {e}")
    
    return results