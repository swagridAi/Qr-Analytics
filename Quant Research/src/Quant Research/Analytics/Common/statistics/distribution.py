"""
Distribution Analysis

This module provides functions for analyzing the statistical distributions
of financial time series data, including fitting distributions, calculating
moments, and estimating tail risk measures.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Third-party imports
import numpy as np
import pandas as pd
import scipy.stats as stats

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.statistics.distribution")

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