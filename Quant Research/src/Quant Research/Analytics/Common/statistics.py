"""
Statistical Analysis Utilities

This module provides statistical functions commonly used across all analytics modules.
It includes correlation analysis, hypothesis testing, regression analysis, and various
financial metrics.

Features:
- Correlation and cointegration analysis
- Statistical tests for time series
- Distribution analysis and fitting
- Financial performance metrics
- Hypothesis testing and regression analysis

Usage:
    ```python
    from quant_research.analytics.common.statistics import (
        calculate_correlation,
        test_stationarity,
        calculate_sharpe_ratio,
        fit_distribution
    )
    
    # Calculate correlation between two assets
    corr = calculate_correlation(asset1_returns, asset2_returns, method='pearson')
    
    # Test for stationarity
    is_stationary, p_value = test_stationarity(price_series)
    
    # Calculate Sharpe ratio
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    
    # Fit distribution to returns
    dist_params = fit_distribution(returns, dist_family='t')
    ```
"""

# Standard library imports
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Third-party imports
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize, signal
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, coint
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.statistics")

#------------------------------------------------------------------------
# Correlation and Cointegration Analysis
#------------------------------------------------------------------------

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


#------------------------------------------------------------------------
# Statistical Tests
#------------------------------------------------------------------------

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


#------------------------------------------------------------------------
# Distribution Analysis
#------------------------------------------------------------------------

def calculate_moments(
    series: pd.Series,
    annualize: bool = False,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate statistical moments of a distribution.
    
    Args:
        series: Data series to analyze
        annualize: Whether to annualize results for return data
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Dictionary with mean, variance, skewness, kurtosis and other stats
    """
    # Remove NaN values
    data = series.dropna()
    
    if len(data) == 0:
        return {
            'count': 0,
            'mean': np.nan,
            'std': np.nan,
            'variance': np.nan,
            'skewness': np.nan,
            'excess_kurtosis': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'mad': np.nan
        }
    
    # Calculate basic statistics
    count = len(data)
    mean = data.mean()
    std = data.std()
    variance = data.var()
    min_val = data.min()
    max_val = data.max()
    median = data.median()
    mad = (data - median).abs().mean()  # Median Absolute Deviation
    
    # Calculate higher moments
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)  # Excess kurtosis (normal = 0)
    
    # Annualize if requested (for return data)
    if annualize:
        mean = mean * periods_per_year
        variance = variance * periods_per_year
        std = std * np.sqrt(periods_per_year)
    
    return {
        'count': count,
        'mean': mean,
        'std': std,
        'variance': variance,
        'skewness': skewness,
        'excess_kurtosis': kurtosis,
        'min': min_val,
        'max': max_val,
        'median': median,
        'mad': mad
    }


def fit_distribution(
    data: Union[pd.Series, np.ndarray],
    dist_family: str = 'norm',
    n_samples: int = 1000,
    test_fit: bool = True
) -> Dict[str, Any]:
    """
    Fit a statistical distribution to data.
    
    Args:
        data: Data series to fit
        dist_family: Distribution family ('norm', 't', 'skewnorm', etc.)
        n_samples: Number of samples for comparing fitted vs actual
        test_fit: Whether to perform goodness-of-fit test
        
    Returns:
        Dictionary with fitted parameters and goodness-of-fit
        
    Raises:
        ValueError: If an invalid distribution family is specified
    """
    # Convert to numpy array if needed
    if isinstance(data, pd.Series):
        data = data.dropna().values
    
    # Check available distributions
    available_dists = [
        'norm', 't', 'skewnorm', 'cauchy', 'laplace',
        'logistic', 'gennorm', 'gamma', 'expon', 'lognorm'
    ]
    
    if dist_family not in available_dists:
        raise ValueError(f"Invalid distribution family: {dist_family}. Available: {available_dists}")
    
    # Get the distribution class
    dist_class = getattr(stats, dist_family)
    
    # Fit the distribution
    try:
        params = dist_class.fit(data)
        
        # Generate results
        results = {
            'distribution': dist_family,
            'params': params,
            'mean': dist_class.mean(*params),
            'variance': dist_class.var(*params),
            'skewness': dist_class.stats(*params, moments='s'),
            'kurtosis': dist_class.stats(*params, moments='k')
        }
        
        # Test goodness of fit if requested
        if test_fit:
            # Generate samples from fitted distribution
            samples = dist_class.rvs(*params, size=n_samples)
            
            # Perform KS test
            ks_stat, ks_pvalue = stats.kstest(data, dist_family, params)
            
            # Calculate log-likelihood
            loglik = np.sum(dist_class.logpdf(data, *params))
            
            # Calculate BIC and AIC
            k = len(params)
            n = len(data)
            bic = k * np.log(n) - 2 * loglik
            aic = 2 * k - 2 * loglik
            
            # Add fit statistics to results
            results.update({
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'loglikelihood': loglik,
                'aic': aic,
                'bic': bic
            })
        
        return results
    
    except Exception as e:
        logger.warning(f"Failed to fit {dist_family} distribution: {e}")
        return {'distribution': dist_family, 'error': str(e)}


def estimate_tail_risk(
    returns: pd.Series,
    method: str = 'historical',
    alpha: float = 0.05,
    window: Optional[int] = None,
    tail: str = 'left'
) -> Dict[str, float]:
    """
    Estimate tail risk measures such as Value-at-Risk (VaR) and Expected Shortfall (ES).
    
    Args:
        returns: Return series
        method: Method for estimation ('historical', 'parametric', 'ewma')
        alpha: Significance level (e.g., 0.05 for 95% VaR)
        window: Window size for rolling estimation (None for full sample)
        tail: Which tail to analyze ('left' for losses, 'right' for gains)
        
    Returns:
        Dictionary with tail risk measures
        
    Raises:
        ValueError: If an invalid method or tail is specified
    """
    # Remove NaN values
    returns_clean = returns.dropna()
    
    # Validate inputs
    if tail not in ['left', 'right']:
        raise ValueError(f"Invalid tail: {tail}. Use 'left' or 'right'")
    
    # Adjust for tail direction
    if tail == 'left':
        # Analyzing losses, so negative returns are in the left tail
        q = alpha
    else:
        # Analyzing gains, so positive returns are in the right tail
        q = 1 - alpha
    
    # Implement different VaR/ES methods
    if method == 'historical':
        # Historical simulation method
        var = returns_clean.quantile(q)
        
        if tail == 'left':
            es_values = returns_clean[returns_clean <= var]
        else:
            es_values = returns_clean[returns_clean >= var]
        
        es = es_values.mean() if len(es_values) > 0 else var
        
    elif method == 'parametric':
        # Parametric method (Gaussian approximation)
        mean = returns_clean.mean()
        std = returns_clean.std()
        
        # Calculate VaR
        z_score = stats.norm.ppf(q)
        var = mean + z_score * std
        
        # Calculate ES
        if tail == 'left':
            es = mean - std * stats.norm.pdf(z_score) / alpha
        else:
            es = mean + std * stats.norm.pdf(z_score) / alpha
        
    elif method == 'ewma':
        # Exponentially Weighted Moving Average for volatility
        if window is None:
            window = min(len(returns_clean), 60)  # Default to 60 periods
        
        # Calculate EWMA variance
        decay = 0.94  # RiskMetrics standard
        vol = np.sqrt(returns_clean.ewm(alpha=1-decay).var())
        
        # Calculate VaR
        mean = returns_clean.ewm(alpha=1-decay).mean()
        z_score = stats.norm.ppf(q)
        var = mean.iloc[-1] + z_score * vol.iloc[-1]
        
        # Calculate ES
        if tail == 'left':
            es = mean.iloc[-1] - vol.iloc[-1] * stats.norm.pdf(z_score) / alpha
        else:
            es = mean.iloc[-1] + vol.iloc[-1] * stats.norm.pdf(z_score) / alpha
        
    else:
        raise ValueError(f"Invalid method: {method}. Use 'historical', 'parametric', or 'ewma'")
    
    # Create result dictionary
    name_suffix = "VaR" if tail == 'left' else "gain_VaR"
    es_name = "ES" if tail == 'left' else "gain_ES"
    
    results = {
        f"{int((1-alpha)*100)}%_{name_suffix}": var,
        f"{int((1-alpha)*100)}%_{es_name}": es,
        'alpha': alpha,
        'method': method
    }
    
    return results


#------------------------------------------------------------------------
# Financial Performance Metrics
#------------------------------------------------------------------------

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    negative_sharpe: bool = True
) -> float:
    """
    Calculate Sharpe ratio for a return series.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        negative_sharpe: Whether to allow negative Sharpe ratios
        
    Returns:
        Sharpe ratio
    """
    # Remove NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Adjust risk-free rate to match return frequency
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns_clean - rf_per_period
    
    # Calculate mean and std of excess returns
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    
    # Calculate Sharpe ratio
    if std_excess == 0:
        return np.nan
    
    sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
    
    # Adjust negative Sharpe if requested
    if not negative_sharpe and sharpe < 0:
        return 0.0
    
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: Optional[float] = None,
    negative_sortino: bool = True
) -> float:
    """
    Calculate Sortino ratio for a return series.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        target_return: Target return (if None, use risk-free rate)
        negative_sortino: Whether to allow negative Sortino ratios
        
    Returns:
        Sortino ratio
    """
    # Remove NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Adjust risk-free rate to match return frequency
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Set target return
    if target_return is None:
        target_return = rf_per_period
    
    # Calculate excess returns
    excess_returns = returns_clean - target_return
    
    # Calculate mean excess return
    mean_excess = excess_returns.mean()
    
    # Calculate downside deviation (only consider returns below target)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        # No downside returns - perfect Sortino
        return np.inf
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return np.nan
    
    # Calculate Sortino ratio
    sortino = mean_excess / downside_deviation * np.sqrt(periods_per_year)
    
    # Adjust negative Sortino if requested
    if not negative_sortino and sortino < 0:
        return 0.0
    
    return sortino


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    max_dd_method: str = 'returns',
    window: Optional[int] = None
) -> float:
    """
    Calculate Calmar ratio for a return series.
    
    Args:
        returns: Return series
        periods_per_year: Number of periods per year
        max_dd_method: Method to calculate maximum drawdown ('returns' or 'prices')
        window: Window for max drawdown calculation (None for full sample)
        
    Returns:
        Calmar ratio
        
    Raises:
        ValueError: If an invalid max_dd_method is specified
    """
    # Remove NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Calculate annualized return
    annual_return = returns_clean.mean() * periods_per_year
    
    # Calculate maximum drawdown
    if max_dd_method == 'returns':
        # Calculate cumulative returns
        cum_returns = (1 + returns_clean).cumprod()
        
        # Limit to window if specified
        if window is not None:
            cum_returns = cum_returns.iloc[-window:]
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = (cum_returns / running_max) - 1
        
        # Get maximum drawdown
        max_dd = drawdowns.min()
        
    elif max_dd_method == 'prices':
        # Treat return series as price series directly
        prices = returns_clean
        
        # Limit to window if specified
        if window is not None:
            prices = prices.iloc[-window:]
        
        # Calculate running maximum
        running_max = prices.cummax()
        
        # Calculate drawdowns
        drawdowns = (prices / running_max) - 1
        
        # Get maximum drawdown
        max_dd = drawdowns.min()
        
    else:
        raise ValueError(f"Invalid max_dd_method: {max_dd_method}. Use 'returns' or 'prices'")
    
    # Check for zero drawdown
    if max_dd == 0:
        return np.inf
    
    # Calculate Calmar ratio
    calmar = annual_return / abs(max_dd)
    
    return calmar


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information Ratio for a return series versus a benchmark.
    
    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        periods_per_year: Number of periods per year
        
    Returns:
        Information Ratio
    """
    # Align series and remove NaNs
    returns_clean, benchmark_clean = returns.align(benchmark_returns, join='inner')
    returns_clean = returns_clean.dropna()
    benchmark_clean = benchmark_clean.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Calculate tracking error
    tracking_diff = returns_clean - benchmark_clean
    tracking_error = tracking_diff.std()
    
    if tracking_error == 0:
        return np.nan
    
    # Calculate Information Ratio
    information_ratio = tracking_diff.mean() / tracking_error * np.sqrt(periods_per_year)
    
    return information_ratio


def calculate_drawdowns(
    returns: pd.Series,
    calculate_recovery: bool = True
) -> pd.DataFrame:
    """
    Calculate drawdowns from a return series.
    
    Args:
        returns: Return series
        calculate_recovery: Whether to calculate recovery periods
        
    Returns:
        DataFrame with drawdown information
    """
    # Remove NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return pd.DataFrame()
    
    # Calculate cumulative returns
    cum_returns = (1 + returns_clean).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdowns
    drawdowns = (cum_returns / running_max) - 1
    
    # Create DataFrame for results
    result = pd.DataFrame({
        'returns': returns_clean,
        'cum_returns': cum_returns,
        'drawdown': drawdowns
    })
    
    # Find drawdown periods
    is_drawdown = result['drawdown'] < 0
    
    # Calculate underwater periods (consecutive drawdown)
    result['is_drawdown'] = is_drawdown
    result['drawdown_group'] = (result['is_drawdown'] != result['is_drawdown'].shift()).cumsum()
    
    # Calculate start and end of each drawdown
    drawdown_periods = []
    
    # Extract unique drawdown periods
    for group_id, group_df in result[result['is_drawdown']].groupby('drawdown_group'):
        # Only include actual drawdowns
        if not group_df.empty and group_df['drawdown'].min() < 0:
            start_date = group_df.index[0]
            end_date = group_df.index[-1]
            max_drawdown = group_df['drawdown'].min()
            max_drawdown_date = group_df['drawdown'].idxmin()
            
            recovery_date = None
            recovery_periods = np.nan
            
            # Calculate recovery if requested and not in the most recent drawdown
            if calculate_recovery and end_date != result.index[-1]:
                # Find when we next reach the previous peak
                peak_value = running_max.loc[start_date]
                future_df = cum_returns.loc[end_date:]
                recovery_dates = future_df[future_df >= peak_value].index
                
                if len(recovery_dates) > 0:
                    recovery_date = recovery_dates[0]
                    recovery_periods = len(result.loc[end_date:recovery_date]) - 1
            
            drawdown_periods.append({
                'start_date': start_date,
                'maxdd_date': max_drawdown_date,
                'end_date': end_date,
                'recovery_date': recovery_date,
                'max_drawdown': max_drawdown,
                'drawdown_length': len(group_df),
                'recovery_length': recovery_periods
            })
    
    drawdown_df = pd.DataFrame(drawdown_periods)
    
    if len(drawdown_df) > 0:
        # Sort by drawdown magnitude
        drawdown_df = drawdown_df.sort_values('max_drawdown')
    
    return drawdown_df


def calculate_trade_metrics(
    trades: pd.DataFrame,
    pnl_col: str = 'pnl',
    win_threshold: float = 0.0
) -> Dict[str, float]:
    """
    Calculate trading metrics from a list of trades.
    
    Args:
        trades: DataFrame with trade information
        pnl_col: Name of column with profit/loss values
        win_threshold: Threshold for considering a trade a win
        
    Returns:
        Dictionary with trading metrics
    """
    # Validate input
    if pnl_col not in trades.columns:
        raise ValueError(f"PnL column '{pnl_col}' not found in trades DataFrame")
    
    # Extract relevant data
    pnl = trades[pnl_col]
    
    # Basic metrics
    total_trades = len(trades)
    win_trades = (pnl > win_threshold).sum()
    loss_trades = (pnl <= win_threshold).sum()
    win_rate = win_trades / total_trades if total_trades > 0 else np.nan
    
    # PnL metrics
    total_pnl = pnl.sum()
    avg_pnl = pnl.mean()
    
    # Separate winning and losing trades
    wins = pnl[pnl > win_threshold]
    losses = pnl[pnl <= win_threshold]
    
    # Average win and loss
    avg_win = wins.mean() if len(wins) > 0 else np.nan
    avg_loss = losses.mean() if len(losses) > 0 else np.nan
    
    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Win/loss ratio
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    
    # Expected payoff
    expected_payoff = win_rate * avg_win + (1 - win_rate) * avg_loss if win_rate is not np.nan else np.nan
    
    # Maximum consecutive wins and losses
    consecutive_wins = 0
    max_consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    
    for p in pnl:
        if p > win_threshold:  # Win
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:  # Loss
            consecutive_wins = 0
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    
    # Compile results
    metrics = {
        'total_trades': total_trades,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'win_loss_ratio': win_loss_ratio,
        'expected_payoff': expected_payoff,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses
    }
    
    return metrics


#------------------------------------------------------------------------
# Regression and Predictive Analysis
#------------------------------------------------------------------------

def run_linear_regression(
    X: Union[pd.Series, pd.DataFrame],
    y: pd.Series,
    add_constant: bool = True,
    robust: bool = False
) -> Dict[str, Any]:
    """
    Run linear regression and return comprehensive statistics.
    
    Args:
        X: Independent variable(s) (feature)
        y: Dependent variable (target)
        add_constant: Whether to add a constant term
        robust: Whether to use robust regression (for outliers)
        
    Returns:
        Dictionary with regression results
    """
    # Convert Series to DataFrame for consistency
    if isinstance(X, pd.Series):
        X = X.to_frame()
    
    # Align data
    X_aligned, y_aligned = X.align(y, join='inner', axis=0)
    
    # Check for empty data
    if len(X_aligned) == 0 or len(y_aligned) == 0:
        logger.warning("No aligned data for regression")
        return {
            'coefficients': {},
            'r_squared': np.nan,
            'adj_r_squared': np.nan,
            'f_statistic': np.nan,
            'f_pvalue': np.nan,
            'model': None
        }
    
    # Add constant if requested
    if add_constant:
        X_aligned = sm.add_constant(X_aligned)
    
    try:
        # Fit model
        if robust:
            # Robust regression
            model = sm.RLM(y_aligned, X_aligned)
            results = model.fit()
            
            # Get parameters (some stats are not available for robust regression)
            params = results.params
            
            # Create coefficient dictionary
            coefficients = dict(zip(X_aligned.columns, params))
            
            regression_stats = {
                'coefficients': coefficients,
                'r_squared': np.nan,  # Not available for robust regression
                'adj_r_squared': np.nan,
                'f_statistic': np.nan,
                'f_pvalue': np.nan,
                'model': results
            }
            
        else:
            # OLS regression
            model = sm.OLS(y_aligned, X_aligned)
            results = model.fit()
            
            # Get parameters
            params = results.params
            pvalues = results.pvalues
            conf_int = results.conf_int()
            
            # Create coefficient dictionary with additional stats
            coefficients = {
                col: {
                    'value': params[i],
                    'p_value': pvalues[i],
                    'conf_low': conf_int[0][i],
                    'conf_high': conf_int[1][i],
                    'significant': pvalues[i] < 0.05
                }
                for i, col in enumerate(X_aligned.columns)
            }
            
            regression_stats = {
                'coefficients': coefficients,
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj,
                'f_statistic': results.fvalue,
                'f_pvalue': results.f_pvalue,
                'aic': results.aic,
                'bic': results.bic,
                'model': results
            }
        
        return regression_stats
    
    except Exception as e:
        logger.warning(f"Regression failed: {e}")
        return {
            'coefficients': {},
            'r_squared': np.nan,
            'adj_r_squared': np.nan,
            'f_statistic': np.nan,
            'f_pvalue': np.nan,
            'error': str(e),
            'model': None
        }


def calculate_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Calculate regression performance metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with performance metrics
    """
    # Convert to numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Remove NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Check for empty data
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("No data for metrics calculation")
        return {
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'explained_variance': np.nan
        }
    
    # Mean squared error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root mean squared error
    rmse = np.sqrt(mse)
    
    # Mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    if ss_total == 0:
        r2 = np.nan  # Can't calculate R² if all true values are the same
    else:
        r2 = 1 - (ss_residual / ss_total)
    
    # Explained variance
    var_y_true = np.var(y_true)
    explained_variance = 1 - (np.var(y_true - y_pred) / var_y_true) if var_y_true > 0 else np.nan
    
    # Compile metrics
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'explained_variance': explained_variance
    }
    
    return metrics


def calculate_information_coefficient(
    predicted: pd.Series,
    actual: pd.Series,
    method: str = 'rank',
    by_group: Optional[pd.Series] = None
) -> Union[float, pd.Series]:
    """
    Calculate Information Coefficient (IC) between predicted and actual values.
    
    Args:
        predicted: Predicted values (e.g., alpha signals)
        actual: Actual values (e.g., forward returns)
        method: IC calculation method ('rank' or 'pearson')
        by_group: Optional grouping variable for calculating IC by group
        
    Returns:
        Information Coefficient value or Series of values per group
        
    Raises:
        ValueError: If an invalid method is specified
    """
    # Align data
    predicted, actual = predicted.align(actual, join='inner')
    
    # Check for empty data
    if len(predicted) == 0 or len(actual) == 0:
        logger.warning("No aligned data for IC calculation")
        return np.nan
    
    # Calculate IC by group if specified
    if by_group is not None:
        # Align grouping variable
        predicted, actual, by_group = predicted.align(actual, by_group, join='inner')
        
        # Calculate IC for each group
        ic_by_group = {}
        
        for group, group_idx in by_group.groupby(by_group).groups.items():
            if len(group_idx) < 2:  # Need at least 2 points for correlation
                ic_by_group[group] = np.nan
                continue
            
            group_predicted = predicted.loc[group_idx]
            group_actual = actual.loc[group_idx]
            
            if method == 'rank':
                ic = stats.spearmanr(group_predicted, group_actual)[0]
            elif method == 'pearson':
                ic = stats.pearsonr(group_predicted, group_actual)[0]
            else:
                raise ValueError(f"Invalid IC method: {method}. Use 'rank' or 'pearson'")
            
            ic_by_group[group] = ic
        
        return pd.Series(ic_by_group)
    
    # Calculate IC for all data
    if method == 'rank':
        ic = stats.spearmanr(predicted, actual)[0]
    elif method == 'pearson':
        ic = stats.pearsonr(predicted, actual)[0]
    else:
        raise ValueError(f"Invalid IC method: {method}. Use 'rank' or 'pearson'")
    
    return ic


def bootstrap_statistic(
    data: Union[pd.Series, np.ndarray],
    statistic: Callable,
    n_samples: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence intervals for a statistic.
    
    Args:
        data: Data to bootstrap
        statistic: Function that computes the statistic
        n_samples: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with bootstrap results
    """
    # Convert to numpy array if needed
    if isinstance(data, pd.Series):
        data = data.dropna().values
    
    # Check for empty data
    if len(data) == 0:
        logger.warning("No data for bootstrap")
        return {
            'statistic': np.nan,
            'lower_bound': np.nan,
            'upper_bound': np.nan,
            'std_error': np.nan
        }
    
    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calculate the statistic on original data
    original_stat = statistic(data)
    
    # Generate bootstrap samples
    bootstrap_stats = []
    
    for _ in range(n_samples):
        # Sample with replacement
        sample = np.random.choice(data, size=len(data), replace=True)
        
        # Calculate statistic on sample
        sample_stat = statistic(sample)
        bootstrap_stats.append(sample_stat)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    # Calculate bootstrap standard error
    std_error = np.std(bootstrap_stats)
    
    # Compile results
    results = {
        'statistic': original_stat,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'std_error': std_error,
        'n_samples': n_samples,
        'confidence_level': confidence_level
    }
    
    return results