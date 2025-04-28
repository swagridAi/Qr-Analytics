"""
Regression and Predictive Analysis

This module provides functions for regression analysis and predictive modeling
used in financial time series analysis, including linear regression, information
coefficient calculations, and bootstrap methods.
"""

# Standard library imports
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Type

# Third-party imports
import statsmodels.api as sm
from scipy import stats

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.statistics.regression")

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