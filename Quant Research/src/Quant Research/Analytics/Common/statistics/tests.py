"""
Statistical Tests

This module provides functions for various statistical tests commonly used
in financial time series analysis, including stationarity, normality,
autocorrelation, and heteroskedasticity tests.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
import statsmodels.api as sm

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.statistics.tests")

def test_stationarity(
    series: pd.Series,
    test_type: str = 'adf',
    regression: str = 'c',
    max_lags: Optional[int] = None,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Test time series for stationarity.
    
    Args:
        series: Time series to test
        test_type: Type of test ('adf', 'kpss', 'both')
        regression: Regression type for ADF ('c', 'ct', 'ctt', 'nc')
        max_lags: Maximum lags for test (None for automatic)
        significance_level: Significance level for hypothesis testing
        
    Returns:
        Dictionary with test results
        
    Raises:
        ValueError: If an invalid test type is specified
    """
    # Remove NaN values
    series_clean = series.dropna()
    
    if len(series_clean) < 20:  # Arbitrary minimum for decent stationarity test
        logger.warning("Insufficient data for stationarity test (minimum 20 points)")
        return {
            'is_stationary': None,
            'adf_statistic': np.nan,
            'adf_p_value': np.nan,
            'kpss_statistic': np.nan,
            'kpss_p_value': np.nan,
            'critical_values': {}
        }
    
    results = {
        'is_stationary': None,
        'adf_statistic': None,
        'adf_p_value': None,
        'kpss_statistic': None,
        'kpss_p_value': None,
        'critical_values': {}
    }
    
    # Run appropriate test(s)
    if test_type in ['adf', 'both']:
        # Augmented Dickey-Fuller test
        # Null hypothesis: series has a unit root (non-stationary)
        try:
            adf_result = adfuller(series_clean, regression=regression, maxlag=max_lags)
            results['adf_statistic'] = adf_result[0]
            results['adf_p_value'] = adf_result[1]
            results['critical_values']['adf'] = adf_result[4]
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
    
    if test_type in ['kpss', 'both']:
        # KPSS test
        # Null hypothesis: series is stationary
        try:
            kpss_result = kpss(series_clean, regression=regression, nlags=max_lags)
            results['kpss_statistic'] = kpss_result[0]
            results['kpss_p_value'] = kpss_result[1]
            results['critical_values']['kpss'] = kpss_result[3]
        except Exception as e:
            logger.warning(f"KPSS test failed: {e}")
    
    # Determine stationarity based on test results
    if test_type == 'adf':
        # For ADF, reject null hypothesis (p < alpha) means stationary
        results['is_stationary'] = results['adf_p_value'] < significance_level
    elif test_type == 'kpss':
        # For KPSS, fail to reject null hypothesis (p > alpha) means stationary
        results['is_stationary'] = results['kpss_p_value'] >= significance_level
    elif test_type == 'both':
        # Require both tests to agree for more conservative estimate
        if results['adf_p_value'] is not None and results['kpss_p_value'] is not None:
            adf_stationary = results['adf_p_value'] < significance_level
            kpss_stationary = results['kpss_p_value'] >= significance_level
            results['is_stationary'] = adf_stationary and kpss_stationary
    else:
        raise ValueError(f"Invalid test type: {test_type}. Use 'adf', 'kpss', or 'both'")
    
    return results


def test_normality(
    series: pd.Series,
    test_type: str = 'shapiro',
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Test if data follows a normal distribution.
    
    Args:
        series: Data to test
        test_type: Type of test ('shapiro', 'ks', 'jarque_bera', 'anderson')
        significance_level: Significance level for hypothesis testing
        
    Returns:
        Dictionary with test results
        
    Raises:
        ValueError: If an invalid test type is specified
    """
    # Remove NaN values
    series_clean = series.dropna()
    
    results = {
        'is_normal': None,
        'statistic': None,
        'p_value': None,
        'critical_values': {}
    }
    
    # Standardize data (important for some tests)
    data = (series_clean - series_clean.mean()) / series_clean.std()
    
    # Run appropriate test
    if test_type == 'shapiro':
        # Shapiro-Wilk test
        # Null hypothesis: data comes from a normal distribution
        try:
            stat, p_value = stats.shapiro(data)
            results['statistic'] = stat
            results['p_value'] = p_value
            results['is_normal'] = p_value >= significance_level
        except Exception as e:
            logger.warning(f"Shapiro-Wilk test failed: {e}")
            
    elif test_type == 'ks':
        # Kolmogorov-Smirnov test
        # Null hypothesis: data comes from a normal distribution
        try:
            stat, p_value = stats.kstest(data, 'norm')
            results['statistic'] = stat
            results['p_value'] = p_value
            results['is_normal'] = p_value >= significance_level
        except Exception as e:
            logger.warning(f"Kolmogorov-Smirnov test failed: {e}")
            
    elif test_type == 'jarque_bera':
        # Jarque-Bera test
        # Null hypothesis: data has skewness and kurtosis matching normal distribution
        try:
            stat, p_value = stats.jarque_bera(data)
            results['statistic'] = stat
            results['p_value'] = p_value
            results['is_normal'] = p_value >= significance_level
        except Exception as e:
            logger.warning(f"Jarque-Bera test failed: {e}")
            
    elif test_type == 'anderson':
        # Anderson-Darling test
        # Critical values are for specific significance levels
        try:
            result = stats.anderson(data, 'norm')
            results['statistic'] = result.statistic
            
            # Anderson-Darling test provides critical values at specific significance levels
            # Index 2 corresponds to 5% significance level
            results['critical_values'] = {
                '15%': result.critical_values[0],
                '10%': result.critical_values[1],
                '5%': result.critical_values[2],
                '2.5%': result.critical_values[3],
                '1%': result.critical_values[4]
            }
            
            # Check if statistic is less than critical value at specified significance level
            # For Anderson-Darling, if statistic > critical value, we reject normality
            critical_value = result.critical_values[2]  # 5% significance level
            results['is_normal'] = result.statistic < critical_value
        except Exception as e:
            logger.warning(f"Anderson-Darling test failed: {e}")
            
    else:
        raise ValueError(f"Invalid test type: {test_type}. Use 'shapiro', 'ks', 'jarque_bera', or 'anderson'")
    
    return results


def test_autocorrelation(
    series: pd.Series,
    max_lag: int = 20,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Test for autocorrelation in time series.
    
    Args:
        series: Time series to test
        max_lag: Maximum lag to test
        significance_level: Significance level for hypothesis testing
        
    Returns:
        Dictionary with autocorrelation test results
    """
    # Remove NaN values
    series_clean = series.dropna()
    
    # Calculate autocorrelation function
    acf_values = acf(series_clean, nlags=max_lag, fft=True)
    
    # Calculate partial autocorrelation function
    pacf_values = pacf(series_clean, nlags=max_lag, method='ols')
    
    # Calculate standard error (approximate)
    n = len(series_clean)
    se = 1.96 / np.sqrt(n)  # 95% confidence interval
    
    # Check for significant autocorrelation
    significant_lags = []
    for lag in range(1, max_lag + 1):
        if abs(acf_values[lag]) > se:
            significant_lags.append(lag)
    
    # Ljung-Box test for autocorrelation
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    # Test at multiple lags
    lb_results = acorr_ljungbox(series_clean, lags=range(1, max_lag + 1))
    
    # Extract test statistics and p-values
    if hasattr(lb_results, 'iloc'):  # DataFrame output (newer statsmodels)
        lb_stat = lb_results['lb_stat'].values
        lb_pvalue = lb_results['lb_pvalue'].values
    else:  # Tuple output (older statsmodels)
        lb_stat, lb_pvalue = lb_results
    
    # Check if series is autocorrelated
    is_autocorrelated = any(p < significance_level for p in lb_pvalue)
    
    # Compile results
    results = {
        'is_autocorrelated': is_autocorrelated,
        'significant_lags': significant_lags,
        'acf': acf_values[1:],  # Exclude lag 0 (always 1)
        'pacf': pacf_values[1:],  # Exclude lag 0
        'ljung_box_stat': lb_stat,
        'ljung_box_pvalue': lb_pvalue,
        'confidence_interval': se
    }
    
    return results


def test_heteroskedasticity(
    series: pd.Series,
    test_type: str = 'arch',
    max_lag: int = 5,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Test for heteroskedasticity (volatility clustering) in time series.
    
    Args:
        series: Time series to test
        test_type: Type of test ('arch', 'breusch_pagan', 'white')
        max_lag: Maximum lag to test for ARCH effects
        significance_level: Significance level for hypothesis testing
        
    Returns:
        Dictionary with test results
        
    Raises:
        ValueError: If an invalid test type is specified
    """
    # Remove NaN values
    series_clean = series.dropna()
    
    results = {
        'has_heteroskedasticity': None,
        'test_statistic': None,
        'p_value': None,
        'lags_tested': max_lag
    }
    
    if test_type == 'arch':
        # ARCH LM test
        # Null hypothesis: no ARCH effects
        try:
            from statsmodels.stats.diagnostic import het_arch
            
            arch_test = het_arch(series_clean, maxlag=max_lag)
            results['test_statistic'] = arch_test[0]
            results['p_value'] = arch_test[1]
            results['has_heteroskedasticity'] = arch_test[1] < significance_level
            
        except Exception as e:
            logger.warning(f"ARCH test failed: {e}")
            
    elif test_type == 'breusch_pagan':
        # Breusch-Pagan test
        # Null hypothesis: homoskedasticity
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            
            # For Breusch-Pagan, we need to set up a regression model
            # We'll use AR(1) model: y_t = a + b*y_{t-1} + e_t
            y = series_clean
            X = sm.add_constant(y.shift(1).dropna())
            y = y.iloc[1:].reset_index(drop=True)
            X = X.reset_index(drop=True)
            
            # Run OLS regression
            model = sm.OLS(y, X).fit()
            
            # Run Breusch-Pagan test
            bp_test = het_breuschpagan(model.resid, model.model.exog)
            results['test_statistic'] = bp_test[0]
            results['p_value'] = bp_test[1]
            results['has_heteroskedasticity'] = bp_test[1] < significance_level
            
        except Exception as e:
            logger.warning(f"Breusch-Pagan test failed: {e}")
            
    elif test_type == 'white':
        # White's test
        # Null hypothesis: homoskedasticity
        try:
            from statsmodels.stats.diagnostic import het_white
            
            # For White's test, we need to set up a regression model
            # We'll use AR(1) model: y_t = a + b*y_{t-1} + e_t
            y = series_clean
            X = sm.add_constant(y.shift(1).dropna())
            y = y.iloc[1:].reset_index(drop=True)
            X = X.reset_index(drop=True)
            
            # Run OLS regression
            model = sm.OLS(y, X).fit()
            
            # Run White's test
            white_test = het_white(model.resid, model.model.exog)
            results['test_statistic'] = white_test[0]
            results['p_value'] = white_test[1]
            results['has_heteroskedasticity'] = white_test[1] < significance_level
            
        except Exception as e:
            logger.warning(f"White's test failed: {e}")
            
    else:
        raise ValueError(f"Invalid test type: {test_type}. Use 'arch', 'breusch_pagan', or 'white'")
    
    return results