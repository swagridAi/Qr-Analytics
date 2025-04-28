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
"""

# Import functions from submodules to maintain the public API
from .correlation import (
    calculate_correlation,
    rolling_correlation,
    cross_correlation,
    test_cointegration,
    granger_causality_test
)

from .tests import (
    test_stationarity,
    test_normality,
    test_autocorrelation,
    test_heteroskedasticity
)

from .distribution import (
    calculate_moments,
    fit_distribution,
    estimate_tail_risk
)

from .performance import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_information_ratio,
    calculate_drawdowns,
    calculate_trade_metrics
)

from .regression import (
    run_linear_regression,
    calculate_regression_metrics,
    calculate_information_coefficient,
    bootstrap_statistic
)

# Define the public API to ensure backward compatibility
__all__ = [
    # Correlation and Cointegration Analysis
    'calculate_correlation',
    'rolling_correlation',
    'cross_correlation',
    'test_cointegration',
    'granger_causality_test',
    
    # Statistical Tests
    'test_stationarity',
    'test_normality',
    'test_autocorrelation',
    'test_heteroskedasticity',
    
    # Distribution Analysis
    'calculate_moments',
    'fit_distribution',
    'estimate_tail_risk',
    
    # Financial Performance Metrics
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_information_ratio',
    'calculate_drawdowns',
    'calculate_trade_metrics',
    
    # Regression and Predictive Analysis
    'run_linear_regression',
    'calculate_regression_metrics',
    'calculate_information_coefficient',
    'bootstrap_statistic'
]