"""
Financial and statistical metrics for the Quant Research dashboard.

This module provides functions for calculating various metrics related to
financial performance, risk, portfolio characteristics, and market behavior.
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")


# ========== Performance Metrics ==========

@dataclass
class PerformanceMetrics:
    """Container for performance metric calculations."""
    
    # Return metrics
    total_return: Optional[float] = None
    annualized_return: Optional[float] = None
    cagr: Optional[float] = None  # Compound annual growth rate
    
    # Risk metrics
    volatility: Optional[float] = None
    downside_deviation: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_drawdown_duration: Optional[int] = None
    var: Optional[float] = None  # Value at Risk
    cvar: Optional[float] = None  # Conditional Value at Risk
    
    # Risk-adjusted metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    information_ratio: Optional[float] = None
    
    # Market metrics
    beta: Optional[float] = None
    alpha: Optional[float] = None
    r_squared: Optional[float] = None
    tracking_error: Optional[float] = None
    
    # Trade metrics
    trade_count: Optional[int] = None
    win_rate: Optional[float] = None
    loss_rate: Optional[float] = None
    avg_profit: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_win_loss_ratio: Optional[float] = None
    
    # Distribution metrics
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to a dictionary.
        
        Returns:
            Dictionary of metrics with formatted values
        """
        return {
            "total_return": self._format_pct(self.total_return),
            "annualized_return": self._format_pct(self.annualized_return),
            "cagr": self._format_pct(self.cagr),
            "volatility": self._format_pct(self.volatility),
            "downside_deviation": self._format_pct(self.downside_deviation),
            "max_drawdown": self._format_pct(self.max_drawdown),
            "max_drawdown_duration": self.max_drawdown_duration,
            "var": self._format_pct(self.var),
            "cvar": self._format_pct(self.cvar),
            "sharpe_ratio": self._format_float(self.sharpe_ratio),
            "sortino_ratio": self._format_float(self.sortino_ratio),
            "calmar_ratio": self._format_float(self.calmar_ratio),
            "information_ratio": self._format_float(self.information_ratio),
            "beta": self._format_float(self.beta),
            "alpha": self._format_pct(self.alpha),
            "r_squared": self._format_float(self.r_squared),
            "tracking_error": self._format_pct(self.tracking_error),
            "trade_count": self.trade_count,
            "win_rate": self._format_pct(self.win_rate),
            "loss_rate": self._format_pct(self.loss_rate),
            "avg_profit": self._format_float(self.avg_profit),
            "avg_loss": self._format_float(self.avg_loss),
            "profit_factor": self._format_float(self.profit_factor),
            "avg_win_loss_ratio": self._format_float(self.avg_win_loss_ratio),
            "skewness": self._format_float(self.skewness),
            "kurtosis": self._format_float(self.kurtosis)
        }
    
    @staticmethod
    def _format_pct(value: Optional[float]) -> Optional[str]:
        """Format a value as a percentage string."""
        if value is None:
            return None
        return f"{value * 100:.2f}%"
    
    @staticmethod
    def _format_float(value: Optional[float]) -> Optional[str]:
        """Format a float value with 2 decimal places."""
        if value is None:
            return None
        return f"{value:.2f}"


def calculate_returns(
    prices: Union[pd.Series, pd.DataFrame],
    method: str = 'simple',
    period: int = 1
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate returns from price data.
    
    Args:
        prices: Price series or DataFrame
        method: Return calculation method ('simple', 'log')
        period: Period for return calculation
        
    Returns:
        Returns series or DataFrame
    """
    if prices is None or (isinstance(prices, pd.Series) and prices.empty) or \
       (isinstance(prices, pd.DataFrame) and prices.empty):
        return prices
    
    try:
        if method.lower() == 'log':
            # Log returns
            return np.log(prices / prices.shift(period))
        else:
            # Simple returns
            return prices.pct_change(periods=period)
    except Exception as e:
        logger.error(f"Error calculating returns: {str(e)}")
        if isinstance(prices, pd.Series):
            return pd.Series(index=prices.index)
        else:
            return pd.DataFrame(index=prices.index)


def calculate_cumulative_returns(
    returns: Union[pd.Series, pd.DataFrame],
    starting_value: float = 1.0
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate cumulative returns from a return series.
    
    Args:
        returns: Return series or DataFrame
        starting_value: Initial value
        
    Returns:
        Cumulative returns series or DataFrame
    """
    if returns is None or (isinstance(returns, pd.Series) and returns.empty) or \
       (isinstance(returns, pd.DataFrame) and returns.empty):
        return returns
    
    try:
        return starting_value * ((1 + returns).cumprod())
    except Exception as e:
        logger.error(f"Error calculating cumulative returns: {str(e)}")
        if isinstance(returns, pd.Series):
            return pd.Series(index=returns.index)
        else:
            return pd.DataFrame(index=returns.index)


def calculate_total_return(
    returns: Union[pd.Series, pd.DataFrame]
) -> Union[float, pd.Series]:
    """
    Calculate total return from a return series.
    
    Args:
        returns: Return series or DataFrame
        
    Returns:
        Total return value or Series
    """
    if returns is None or (isinstance(returns, pd.Series) and returns.empty) or \
       (isinstance(returns, pd.DataFrame) and returns.empty):
        return np.nan if isinstance(returns, pd.Series) else pd.Series()
    
    try:
        return (1 + returns).prod() - 1
    except Exception as e:
        logger.error(f"Error calculating total return: {str(e)}")
        return np.nan if isinstance(returns, pd.Series) else pd.Series()


def calculate_annualized_return(
    returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: int = 252
) -> Union[float, pd.Series]:
    """
    Calculate annualized return from a return series.
    
    Args:
        returns: Return series or DataFrame
        periods_per_year: Number of return periods in a year
        
    Returns:
        Annualized return value or Series
    """
    if returns is None or (isinstance(returns, pd.Series) and returns.empty) or \
       (isinstance(returns, pd.DataFrame) and returns.empty):
        return np.nan if isinstance(returns, pd.Series) else pd.Series()
    
    try:
        total_return = calculate_total_return(returns)
        num_periods = len(returns.dropna())
        
        if num_periods == 0:
            return np.nan if isinstance(returns, pd.Series) else pd.Series()
        
        # Calculate annualized return
        if isinstance(total_return, (float, int)):
            return (1 + total_return) ** (periods_per_year / num_periods) - 1
        else:
            # Handle Series case
            return (1 + total_return) ** (periods_per_year / num_periods) - 1
    except Exception as e:
        logger.error(f"Error calculating annualized return: {str(e)}")
        return np.nan if isinstance(returns, pd.Series) else pd.Series()


def calculate_cagr(
    equity_series: pd.Series,
    annualization_factor: int = 252
) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        equity_series: Equity curve series
        annualization_factor: Periods per year
        
    Returns:
        CAGR value
    """
    if equity_series is None or equity_series.empty:
        return np.nan
    
    try:
        equity_series = equity_series.dropna()
        if len(equity_series) < 2:
            return np.nan
            
        start_value = equity_series.iloc[0]
        end_value = equity_series.iloc[-1]
        
        # Get the timespan in years
        if isinstance(equity_series.index, pd.DatetimeIndex):
            start_date = equity_series.index[0]
            end_date = equity_series.index[-1]
            years = (end_date - start_date).days / 365.25
        else:
            # Use the number of periods if not datetime index
            years = len(equity_series) / annualization_factor
        
        if years == 0 or start_value <= 0 or end_value <= 0:
            return np.nan
            
        # Calculate CAGR
        return (end_value / start_value) ** (1 / years) - 1
    except Exception as e:
        logger.error(f"Error calculating CAGR: {str(e)}")
        return np.nan


def calculate_rolling_returns(
    returns: pd.Series,
    window: int,
    annualized: bool = True,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate rolling returns over a specified window.
    
    Args:
        returns: Return series
        window: Rolling window size
        annualized: Whether to annualize the returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Rolling returns series
    """
    if returns is None or returns.empty or window <= 0:
        return pd.Series()
    
    try:
        # Calculate rolling window product of (1+r)
        rolling_return = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
        
        # Annualize if requested
        if annualized:
            rolling_return = (1 + rolling_return) ** (periods_per_year / window) - 1
            
        return rolling_return
    except Exception as e:
        logger.error(f"Error calculating rolling returns: {str(e)}")
        return pd.Series(index=returns.index)


# ========== Risk Metrics ==========

def calculate_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    annualized: bool = True,
    periods_per_year: int = 252
) -> Union[float, pd.Series]:
    """
    Calculate volatility (standard deviation) of returns.
    
    Args:
        returns: Return series or DataFrame
        annualized: Whether to annualize the volatility
        periods_per_year: Number of periods in a year
        
    Returns:
        Volatility value or Series
    """
    if returns is None or (isinstance(returns, pd.Series) and returns.empty) or \
       (isinstance(returns, pd.DataFrame) and returns.empty):
        return np.nan if isinstance(returns, pd.Series) else pd.Series()
    
    try:
        # Calculate standard deviation
        std = returns.std()
        
        # Annualize if requested
        if annualized:
            std = std * np.sqrt(periods_per_year)
            
        return std
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
        return np.nan if isinstance(returns, pd.Series) else pd.Series()


def calculate_downside_deviation(
    returns: Union[pd.Series, pd.DataFrame],
    min_acceptable_return: float = 0.0,
    annualized: bool = True,
    periods_per_year: int = 252
) -> Union[float, pd.Series]:
    """
    Calculate downside deviation of returns.
    
    Args:
        returns: Return series or DataFrame
        min_acceptable_return: Minimum acceptable return threshold
        annualized: Whether to annualize the deviation
        periods_per_year: Number of periods in a year
        
    Returns:
        Downside deviation value or Series
    """
    if returns is None or (isinstance(returns, pd.Series) and returns.empty) or \
       (isinstance(returns, pd.DataFrame) and returns.empty):
        return np.nan if isinstance(returns, pd.Series) else pd.Series()
    
    try:
        # Calculate downside returns
        downside_returns = returns[returns < min_acceptable_return] - min_acceptable_return
        
        if len(downside_returns) == 0:
            return 0.0 if isinstance(returns, pd.Series) else pd.Series(0.0, index=returns.columns)
        
        # Calculate downside deviation
        dd = np.sqrt(np.mean(downside_returns ** 2))
        
        # Annualize if requested
        if annualized:
            dd = dd * np.sqrt(periods_per_year)
            
        return dd
    except Exception as e:
        logger.error(f"Error calculating downside deviation: {str(e)}")
        return np.nan if isinstance(returns, pd.Series) else pd.Series()


def calculate_drawdowns(
    equity_series: pd.Series
) -> Tuple[pd.Series, float, int]:
    """
    Calculate drawdown series and metrics.
    
    Args:
        equity_series: Equity curve series
        
    Returns:
        Tuple of (drawdown_series, max_drawdown, max_drawdown_duration)
    """
    if equity_series is None or equity_series.empty:
        return pd.Series(), np.nan, 0
    
    try:
        # Calculate running maximum
        running_max = equity_series.cummax()
        
        # Calculate drawdown series
        drawdown_series = (equity_series / running_max - 1)
        
        # Calculate maximum drawdown
        max_drawdown = drawdown_series.min()
        
        # Calculate maximum drawdown duration
        max_drawdown_duration = 0
        
        if max_drawdown < 0:
            # Find the index of max drawdown
            max_dd_idx = drawdown_series.idxmin()
            
            # Find the last peak before the max drawdown
            prev_peak_idx = running_max.loc[:max_dd_idx].idxmax()
            
            # Find recovery point (if any)
            try:
                recovery_idx = drawdown_series.loc[max_dd_idx:].loc[
                    drawdown_series.loc[max_dd_idx:] >= 0
                ].index[0]
            except (IndexError, KeyError):
                # No recovery point yet
                recovery_idx = equity_series.index[-1]
            
            # Calculate duration in days if datetime index, otherwise in periods
            if isinstance(equity_series.index, pd.DatetimeIndex):
                max_drawdown_duration = (recovery_idx - prev_peak_idx).days
            else:
                max_drawdown_duration = equity_series.index.get_loc(recovery_idx) - \
                                        equity_series.index.get_loc(prev_peak_idx)
        
        return drawdown_series, max_drawdown, max_drawdown_duration
    
    except Exception as e:
        logger.error(f"Error calculating drawdowns: {str(e)}")
        return pd.Series(), np.nan, 0


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical',
    lookback_period: Optional[int] = None
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Return series
        confidence_level: Confidence level (0-1)
        method: VaR calculation method ('historical', 'gaussian', 'cornish_fisher')
        lookback_period: Rolling window for VaR calculation
        
    Returns:
        VaR value (as a positive number)
    """
    if returns is None or returns.empty:
        return np.nan
    
    # Remove NaN values
    clean_returns = returns.dropna()
    
    # Use lookback period if specified
    if lookback_period is not None and lookback_period > 0 and lookback_period < len(clean_returns):
        clean_returns = clean_returns[-lookback_period:]
    
    try:
        percentile = 100 * (1 - confidence_level)
        
        if method == 'gaussian':
            # Parametric VaR assuming normal distribution
            mean = clean_returns.mean()
            std = clean_returns.std()
            var = -1 * (mean + std * stats.norm.ppf(1 - confidence_level))
        
        elif method == 'cornish_fisher':
            # Modified VaR using Cornish-Fisher expansion
            mean = clean_returns.mean()
            std = clean_returns.std()
            skew = stats.skew(clean_returns)
            kurtosis = stats.kurtosis(clean_returns)
            
            z_score = stats.norm.ppf(1 - confidence_level)
            z_cf = z_score + (1/6) * (z_score**2 - 1) * skew + \
                   (1/24) * (z_score**3 - 3*z_score) * kurtosis - \
                   (1/36) * (2*z_score**3 - 5*z_score) * skew**2
            
            var = -1 * (mean + std * z_cf)
        
        else:  # Historical VaR (default)
            var = -1 * np.percentile(clean_returns, percentile)
        
        return abs(var)  # Return as positive number
        
    except Exception as e:
        logger.error(f"Error calculating VaR: {str(e)}")
        return np.nan


def calculate_expected_shortfall(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical',
    lookback_period: Optional[int] = None
) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).
    
    Args:
        returns: Return series
        confidence_level: Confidence level (0-1)
        method: VaR calculation method ('historical', 'gaussian')
        lookback_period: Rolling window for calculation
        
    Returns:
        Expected Shortfall value (as a positive number)
    """
    if returns is None or returns.empty:
        return np.nan
    
    # Remove NaN values
    clean_returns = returns.dropna()
    
    # Use lookback period if specified
    if lookback_period is not None and lookback_period > 0 and lookback_period < len(clean_returns):
        clean_returns = clean_returns[-lookback_period:]
    
    try:
        # Calculate VaR
        var = calculate_var(clean_returns, confidence_level, method, lookback_period)
        
        if method == 'gaussian':
            # Parametric ES assuming normal distribution
            mean = clean_returns.mean()
            std = clean_returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            es = -1 * (mean + std * stats.norm.pdf(z_score) / (1 - confidence_level))
        
        else:  # Historical ES (default)
            # Find returns below VaR
            threshold = -1 * var
            tail_returns = clean_returns[clean_returns <= threshold]
            
            if len(tail_returns) == 0:
                return var  # If no tail returns, ES equals VaR
                
            es = -1 * tail_returns.mean()
        
        return abs(es)  # Return as positive number
        
    except Exception as e:
        logger.error(f"Error calculating Expected Shortfall: {str(e)}")
        return np.nan


def calculate_rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> pd.Series:
    """
    Calculate rolling Value at Risk.
    
    Args:
        returns: Return series
        window: Rolling window size
        confidence_level: Confidence level (0-1)
        method: VaR calculation method
        
    Returns:
        Rolling VaR series
    """
    if returns is None or returns.empty or window <= 0:
        return pd.Series()
    
    try:
        # Create result series
        rolling_var = pd.Series(index=returns.index)
        
        # Calculate VaR for each window
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            rolling_var.iloc[i-1] = calculate_var(window_returns, confidence_level, method)
        
        return rolling_var
        
    except Exception as e:
        logger.error(f"Error calculating rolling VaR: {str(e)}")
        return pd.Series(index=returns.index)


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 30,
    annualized: bool = True,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate rolling volatility of returns.
    
    Args:
        returns: Return series
        window: Rolling window size
        annualized: Whether to annualize the volatility
        periods_per_year: Number of periods in a year
        
    Returns:
        Rolling volatility series
    """
    if returns is None or returns.empty or window <= 0:
        return pd.Series()
    
    try:
        # Calculate rolling standard deviation
        rolling_std = returns.rolling(window=window).std()
        
        # Annualize if requested
        if annualized:
            rolling_std = rolling_std * np.sqrt(periods_per_year)
            
        return rolling_std
    except Exception as e:
        logger.error(f"Error calculating rolling volatility: {str(e)}")
        return pd.Series(index=returns.index)


# ========== Risk-Adjusted Performance Metrics ==========

def calculate_sharpe_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Union[float, pd.Series]:
    """
    Calculate Sharpe ratio of returns.
    
    Args:
        returns: Return series or DataFrame
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio value or Series
    """
    if returns is None or (isinstance(returns, pd.Series) and returns.empty) or \
       (isinstance(returns, pd.DataFrame) and returns.empty):
        return np.nan if isinstance(returns, pd.Series) else pd.Series()
    
    try:
        # Calculate daily risk-free rate
        rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        
        # Calculate excess returns
        excess_returns = returns - rf_daily
        
        # Calculate mean and std of excess returns
        mean_excess = excess_returns.mean() * periods_per_year
        std_excess = excess_returns.std() * np.sqrt(periods_per_year)
        
        # Calculate Sharpe ratio
        sharpe = mean_excess / std_excess
        
        return sharpe
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {str(e)}")
        return np.nan if isinstance(returns, pd.Series) else pd.Series()


def calculate_sortino_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    risk_free_rate: float = 0.0,
    min_acceptable_return: float = 0.0,
    periods_per_year: int = 252
) -> Union[float, pd.Series]:
    """
    Calculate Sortino ratio of returns.
    
    Args:
        returns: Return series or DataFrame
        risk_free_rate: Annualized risk-free rate
        min_acceptable_return: Minimum acceptable return
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino ratio value or Series
    """
    if returns is None or (isinstance(returns, pd.Series) and returns.empty) or \
       (isinstance(returns, pd.DataFrame) and returns.empty):
        return np.nan if isinstance(returns, pd.Series) else pd.Series()
    
    try:
        # Calculate daily risk-free rate
        rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        
        # Calculate excess returns
        excess_returns = returns - rf_daily
        
        # Calculate mean excess returns
        mean_excess = excess_returns.mean() * periods_per_year
        
        # Calculate downside deviation
        dd = calculate_downside_deviation(
            returns, min_acceptable_return, True, periods_per_year
        )
        
        # Handle zero downside deviation
        if isinstance(dd, (float, int)) and dd == 0:
            return np.inf if mean_excess > 0 else -np.inf
            
        # Calculate Sortino ratio
        sortino = mean_excess / dd
        
        return sortino
    except Exception as e:
        logger.error(f"Error calculating Sortino ratio: {str(e)}")
        return np.nan if isinstance(returns, pd.Series) else pd.Series()


def calculate_calmar_ratio(
    returns: pd.Series,
    period: int = 36,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Return series
        period: Lookback period in months for max drawdown
        periods_per_year: Number of periods in a year
        
    Returns:
        Calmar ratio value
    """
    if returns is None or returns.empty:
        return np.nan
    
    try:
        # Convert period to number of return periods
        period_length = int(period * periods_per_year / 12)
        
        # Use full history if period is longer than available data
        if period_length >= len(returns):
            period_length = len(returns)
        
        # Get relevant return slice
        return_slice = returns[-period_length:]
        
        # Calculate annualized return
        ann_return = calculate_annualized_return(return_slice, periods_per_year)
        
        # Calculate drawdowns
        equity_curve = calculate_cumulative_returns(return_slice)
        _, max_dd, _ = calculate_drawdowns(equity_curve)
        
        # Avoid division by zero
        if max_dd == 0:
            return np.inf if ann_return > 0 else -np.inf
        
        # Calculate Calmar ratio
        calmar = ann_return / abs(max_dd)
        
        return calmar
    except Exception as e:
        logger.error(f"Error calculating Calmar ratio: {str(e)}")
        return np.nan


def calculate_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Omega ratio.
    
    Args:
        returns: Return series
        threshold: Return threshold
        periods_per_year: Number of periods in a year
        
    Returns:
        Omega ratio value
    """
    if returns is None or returns.empty:
        return np.nan
    
    try:
        # Adjust threshold to match return frequency
        threshold_adj = (1 + threshold) ** (1 / periods_per_year) - 1
        
        # Classify returns as gains or losses relative to threshold
        returns_less_thresh = returns - threshold_adj
        gains = returns_less_thresh[returns_less_thresh > 0].sum()
        losses = -1 * returns_less_thresh[returns_less_thresh < 0].sum()
        
        # Avoid division by zero
        if losses == 0:
            return np.inf if gains > 0 else 0
        
        # Calculate Omega ratio
        omega = gains / losses
        
        return omega
    except Exception as e:
        logger.error(f"Error calculating Omega ratio: {str(e)}")
        return np.nan


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information ratio.
    
    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        periods_per_year: Number of periods in a year
        
    Returns:
        Information ratio value
    """
    if returns is None or returns.empty or benchmark_returns is None or benchmark_returns.empty:
        return np.nan
    
    try:
        # Align time indices
        returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
        
        if len(returns) == 0:
            return np.nan
        
        # Calculate tracking error
        tracking_difference = returns - benchmark_returns
        tracking_error = tracking_difference.std() * np.sqrt(periods_per_year)
        
        # Calculate excess returns
        excess_return = returns.mean() - benchmark_returns.mean()
        annualized_excess = excess_return * periods_per_year
        
        # Avoid division by zero
        if tracking_error == 0:
            return np.inf if annualized_excess > 0 else -np.inf
        
        # Calculate Information ratio
        information_ratio = annualized_excess / tracking_error
        
        return information_ratio
    except Exception as e:
        logger.error(f"Error calculating Information ratio: {str(e)}")
        return np.nan


# ========== Market Metrics ==========

def calculate_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate beta of returns to a benchmark.
    
    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        
    Returns:
        Beta value
    """
    if returns is None or returns.empty or benchmark_returns is None or benchmark_returns.empty:
        return np.nan
    
    try:
        # Align time indices
        returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
        
        if len(returns) == 0:
            return np.nan
        
        # Calculate covariance
        covariance = returns.cov(benchmark_returns)
        
        # Calculate variance of benchmark
        benchmark_variance = benchmark_returns.var()
        
        # Avoid division by zero
        if benchmark_variance == 0:
            return np.nan
        
        # Calculate beta
        beta = covariance / benchmark_variance
        
        return beta
    except Exception as e:
        logger.error(f"Error calculating beta: {str(e)}")
        return np.nan


def calculate_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Jensen's alpha.
    
    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Alpha value (annualized)
    """
    if returns is None or returns.empty or benchmark_returns is None or benchmark_returns.empty:
        return np.nan
    
    try:
        # Align time indices
        returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
        
        if len(returns) == 0:
            return np.nan
        
        # Calculate beta
        beta = calculate_beta(returns, benchmark_returns)
        
        # Calculate daily risk-free rate
        rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        
        # Calculate alpha
        alpha = returns.mean() - rf_daily - beta * (benchmark_returns.mean() - rf_daily)
        
        # Annualize alpha
        alpha_annualized = alpha * periods_per_year
        
        return alpha_annualized
    except Exception as e:
        logger.error(f"Error calculating alpha: {str(e)}")
        return np.nan


def calculate_r_squared(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        
    Returns:
        R-squared value
    """
    if returns is None or returns.empty or benchmark_returns is None or benchmark_returns.empty:
        return np.nan
    
    try:
        # Align time indices
        returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
        
        if len(returns) == 0:
            return np.nan
        
        # Calculate correlation coefficient
        correlation = returns.corr(benchmark_returns)
        
        # Calculate R-squared
        r_squared = correlation ** 2
        
        return r_squared
    except Exception as e:
        logger.error(f"Error calculating R-squared: {str(e)}")
        return np.nan


def calculate_tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    annualized: bool = True,
    periods_per_year: int = 252
) -> float:
    """
    Calculate tracking error.
    
    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        annualized: Whether to annualize the result
        periods_per_year: Number of periods in a year
        
    Returns:
        Tracking error value
    """
    if returns is None or returns.empty or benchmark_returns is None or benchmark_returns.empty:
        return np.nan
    
    try:
        # Align time indices
        returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
        
        if len(returns) == 0:
            return np.nan
        
        # Calculate tracking difference
        tracking_difference = returns - benchmark_returns
        
        # Calculate tracking error
        tracking_error = tracking_difference.std()
        
        # Annualize if requested
        if annualized:
            tracking_error = tracking_error * np.sqrt(periods_per_year)
            
        return tracking_error
    except Exception as e:
        logger.error(f"Error calculating tracking error: {str(e)}")
        return np.nan


def calculate_up_down_capture(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> Tuple[float, float]:
    """
    Calculate up/down market capture ratios.
    
    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        
    Returns:
        Tuple of (up_capture, down_capture)
    """
    if returns is None or returns.empty or benchmark_returns is None or benchmark_returns.empty:
        return np.nan, np.nan
    
    try:
        # Align time indices
        returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
        
        if len(returns) == 0:
            return np.nan, np.nan
        
        # Separate up and down market periods
        up_market = benchmark_returns > 0
        down_market = benchmark_returns < 0
        
        # Calculate capture ratios
        up_capture = (returns[up_market].mean() / benchmark_returns[up_market].mean()) \
                     if up_market.any() else np.nan
        down_capture = (returns[down_market].mean() / benchmark_returns[down_market].mean()) \
                     if down_market.any() else np.nan
        
        return up_capture, down_capture
    except Exception as e:
        logger.error(f"Error calculating up/down capture: {str(e)}")
        return np.nan, np.nan


# ========== Trade Metrics ==========

def calculate_trade_statistics(
    trades: pd.DataFrame,
    price_column: str = 'price',
    size_column: str = 'size',
    pnl_column: Optional[str] = 'pnl',
    type_column: str = 'type',
    entry_types: List[str] = ['ENTRY', 'BUY'],
    exit_types: List[str] = ['EXIT', 'SELL']
) -> dict:
    """
    Calculate trade statistics.
    
    Args:
        trades: DataFrame of trades
        price_column: Name of price column
        size_column: Name of size column
        pnl_column: Name of PnL column (if provided)
        type_column: Name of trade type column
        entry_types: List of entry trade types
        exit_types: List of exit trade types
        
    Returns:
        Dictionary of trade statistics
    """
    if trades is None or trades.empty:
        return {
            'trade_count': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': np.nan,
            'loss_rate': np.nan,
            'avg_profit': np.nan,
            'avg_loss': np.nan,
            'max_profit': np.nan,
            'max_loss': np.nan,
            'profit_factor': np.nan,
            'avg_win_loss_ratio': np.nan,
            'total_pnl': np.nan
        }
    
    try:
        # Check if we need to calculate PnL
        if pnl_column not in trades.columns:
            # Check if we have enough info to calculate PnL
            if price_column in trades.columns and size_column in trades.columns and type_column in trades.columns:
                # Match entries with exits
                entries = trades[trades[type_column].isin(entry_types)]
                exits = trades[trades[type_column].isin(exit_types)]
                
                if len(entries) > 0 and len(exits) > 0:
                    # This is a simplified approach - in practice, matching can be more complex
                    # Assume entries and exits alternate
                    trades_with_pnl = []
                    
                    if len(entries) == len(exits):
                        for i in range(len(entries)):
                            entry = entries.iloc[i]
                            exit = exits.iloc[i]
                            
                            # Calculate PnL
                            entry_value = entry[price_column] * entry[size_column]
                            exit_value = exit[price_column] * exit[size_column]
                            pnl = exit_value - entry_value
                            
                            # Add PnL to exit trade
                            exit_with_pnl = exit.copy()
                            exit_with_pnl['pnl'] = pnl
                            trades_with_pnl.append(exit_with_pnl)
                    
                    if trades_with_pnl:
                        # Create DataFrame with PnL
                        trades_pnl = pd.DataFrame(trades_with_pnl)
                        pnl_column = 'pnl'
                    else:
                        # Can't calculate PnL
                        return {
                            'trade_count': len(trades),
                            'win_count': np.nan,
                            'loss_count': np.nan,
                            'win_rate': np.nan,
                            'loss_rate': np.nan,
                            'avg_profit': np.nan,
                            'avg_loss': np.nan,
                            'max_profit': np.nan,
                            'max_loss': np.nan,
                            'profit_factor': np.nan,
                            'avg_win_loss_ratio': np.nan,
                            'total_pnl': np.nan
                        }
                else:
                    # No entries or exits
                    return {
                        'trade_count': len(trades),
                        'win_count': np.nan,
                        'loss_count': np.nan,
                        'win_rate': np.nan,
                        'loss_rate': np.nan,
                        'avg_profit': np.nan,
                        'avg_loss': np.nan,
                        'max_profit': np.nan,
                        'max_loss': np.nan,
                        'profit_factor': np.nan,
                        'avg_win_loss_ratio': np.nan,
                        'total_pnl': np.nan
                    }
            else:
                # Not enough info to calculate PnL
                return {
                    'trade_count': len(trades),
                    'win_count': np.nan,
                    'loss_count': np.nan,
                    'win_rate': np.nan,
                    'loss_rate': np.nan,
                    'avg_profit': np.nan,
                    'avg_loss': np.nan,
                    'max_profit': np.nan,
                    'max_loss': np.nan,
                    'profit_factor': np.nan,
                    'avg_win_loss_ratio': np.nan,
                    'total_pnl': np.nan
                }
        else:
            # Use provided PnL column
            trades_pnl = trades
        
        # Calculate trade metrics
        trade_count = len(trades_pnl)
        
        # Separate winning and losing trades
        winning_trades = trades_pnl[trades_pnl[pnl_column] > 0]
        losing_trades = trades_pnl[trades_pnl[pnl_column] < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        # Calculate rates
        win_rate = win_count / trade_count if trade_count > 0 else np.nan
        loss_rate = loss_count / trade_count if trade_count > 0 else np.nan
        
        # Calculate profits and losses
        avg_profit = winning_trades[pnl_column].mean() if win_count > 0 else np.nan
        avg_loss = losing_trades[pnl_column].mean() if loss_count > 0 else np.nan
        max_profit = winning_trades[pnl_column].max() if win_count > 0 else np.nan
        max_loss = losing_trades[pnl_column].min() if loss_count > 0 else np.nan
        
        # Calculate profit factor
        total_profit = winning_trades[pnl_column].sum() if win_count > 0 else 0
        total_loss = abs(losing_trades[pnl_column].sum()) if loss_count > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf
        
        # Calculate win/loss ratio
        avg_win_loss_ratio = abs(avg_profit / avg_loss) if avg_loss is not None and avg_loss != 0 else np.inf
        
        # Calculate total PnL
        total_pnl = trades_pnl[pnl_column].sum()
        
        # Return metrics
        return {
            'trade_count': trade_count,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'avg_win_loss_ratio': avg_win_loss_ratio,
            'total_pnl': total_pnl
        }
    
    except Exception as e:
        logger.error(f"Error calculating trade statistics: {str(e)}")
        return {
            'trade_count': len(trades),
            'win_count': np.nan,
            'loss_count': np.nan,
            'win_rate': np.nan,
            'loss_rate': np.nan,
            'avg_profit': np.nan,
            'avg_loss': np.nan,
            'max_profit': np.nan,
            'max_loss': np.nan,
            'profit_factor': np.nan,
            'avg_win_loss_ratio': np.nan,
            'total_pnl': np.nan
        }


def calculate_trade_durations(
    trades: pd.DataFrame,
    timestamp_column: str = 'timestamp',
    type_column: str = 'type',
    entry_types: List[str] = ['ENTRY', 'BUY'],
    exit_types: List[str] = ['EXIT', 'SELL'],
    symbol_column: Optional[str] = 'symbol'
) -> pd.Series:
    """
    Calculate durations of trades.
    
    Args:
        trades: DataFrame of trades
        timestamp_column: Name of timestamp column
        type_column: Name of trade type column
        entry_types: List of entry trade types
        exit_types: List of exit trade types
        symbol_column: Name of symbol column for grouping
        
    Returns:
        Series of trade durations
    """
    if trades is None or trades.empty:
        return pd.Series()
    
    try:
        # Check if we have required columns
        required_columns = [timestamp_column, type_column]
        if symbol_column is not None:
            required_columns.append(symbol_column)
            
        if not all(col in trades.columns for col in required_columns):
            logger.error("Missing required columns for trade durations calculation")
            return pd.Series()
        
        # Filter to entry and exit trades
        entries = trades[trades[type_column].isin(entry_types)]
        exits = trades[trades[type_column].isin(exit_types)]
        
        if entries.empty or exits.empty:
            logger.warning("No entry or exit trades found")
            return pd.Series()
        
        # Sort by timestamp
        entries = entries.sort_values(timestamp_column)
        exits = exits.sort_values(timestamp_column)
        
        # Group by symbol if provided
        durations = []
        
        if symbol_column is not None:
            # Calculate durations for each symbol
            for symbol in entries[symbol_column].unique():
                symbol_entries = entries[entries[symbol_column] == symbol]
                symbol_exits = exits[exits[symbol_column] == symbol]
                
                if len(symbol_entries) > 0 and len(symbol_exits) > 0:
                    # Match entries with exits
                    # Simplified approach - match each entry with the next exit
                    for i, entry in symbol_entries.iterrows():
                        entry_time = entry[timestamp_column]
                        
                        # Find the next exit after this entry
                        next_exits = symbol_exits[symbol_exits[timestamp_column] > entry_time]
                        
                        if not next_exits.empty:
                            exit_time = next_exits.iloc[0][timestamp_column]
                            duration = exit_time - entry_time
                            durations.append(duration.total_seconds())
        else:
            # No symbol grouping - assume trades are in sequence
            if len(entries) == len(exits):
                for i in range(len(entries)):
                    entry_time = entries.iloc[i][timestamp_column]
                    exit_time = exits.iloc[i][timestamp_column]
                    
                    if exit_time > entry_time:
                        duration = exit_time - entry_time
                        durations.append(duration.total_seconds())
        
        return pd.Series(durations)
    
    except Exception as e:
        logger.error(f"Error calculating trade durations: {str(e)}")
        return pd.Series()


# ========== Portfolio Metrics ==========

def calculate_portfolio_allocation(
    positions: pd.DataFrame,
    value_column: str = 'value',
    symbol_column: str = 'symbol',
    side_column: Optional[str] = 'side',
    groupby_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate portfolio allocation.
    
    Args:
        positions: DataFrame of positions
        value_column: Name of position value column
        symbol_column: Name of symbol column
        side_column: Name of side column (for long/short breakdown)
        groupby_column: Optional column to group by (e.g., sector)
        
    Returns:
        DataFrame with allocation percentages
    """
    if positions is None or positions.empty:
        return pd.DataFrame()
    
    try:
        # Check if we have required columns
        required_columns = [value_column, symbol_column]
        if side_column is not None:
            required_columns.append(side_column)
        if groupby_column is not None:
            required_columns.append(groupby_column)
            
        if not all(col in positions.columns for col in required_columns):
            logger.error("Missing required columns for portfolio allocation calculation")
            return pd.DataFrame()
        
        # Calculate absolute values
        positions = positions.copy()
        positions['abs_value'] = positions[value_column].abs()
        
        # Calculate total portfolio value
        total_value = positions['abs_value'].sum()
        
        if total_value == 0:
            logger.warning("Portfolio total value is zero")
            return pd.DataFrame()
        
        # Determine grouping column
        group_col = groupby_column if groupby_column else symbol_column
        
        # Group by the selected column
        allocation = positions.groupby(group_col)['abs_value'].sum().reset_index()
        
        # Calculate percentage
        allocation['percentage'] = allocation['abs_value'] / total_value * 100
        
        # Sort by allocation percentage
        allocation = allocation.sort_values('percentage', ascending=False)
        
        # Add long/short breakdown if side column provided
        if side_column in positions.columns:
            # Calculate long allocation
            long_positions = positions[positions[side_column].str.upper() == 'LONG']
            long_allocation = long_positions.groupby(group_col)['abs_value'].sum().reset_index()
            long_allocation.columns = [group_col, 'long_value']
            
            # Calculate short allocation
            short_positions = positions[positions[side_column].str.upper() == 'SHORT']
            short_allocation = short_positions.groupby(group_col)['abs_value'].sum().reset_index()
            short_allocation.columns = [group_col, 'short_value']
            
            # Merge allocations
            allocation = pd.merge(allocation, long_allocation, on=group_col, how='left').fillna(0)
            allocation = pd.merge(allocation, short_allocation, on=group_col, how='left').fillna(0)
            
            # Calculate percentages
            allocation['long_percentage'] = allocation['long_value'] / total_value * 100
            allocation['short_percentage'] = allocation['short_value'] / total_value * 100
        
        return allocation
    
    except Exception as e:
        logger.error(f"Error calculating portfolio allocation: {str(e)}")
        return pd.DataFrame()


def calculate_concentration_metrics(
    weights: pd.Series
) -> Dict[str, float]:
    """
    Calculate portfolio concentration metrics.
    
    Args:
        weights: Series of position weights (should sum to 1)
        
    Returns:
        Dictionary of concentration metrics
    """
    if weights is None or weights.empty:
        return {
            'herfindahl_index': np.nan,
            'gini_coefficient': np.nan,
            'top1_concentration': np.nan,
            'top5_concentration': np.nan,
            'top10_concentration': np.nan,
            'effective_n': np.nan
        }
    
    try:
        # Normalize weights if they don't sum to 1
        weights_sum = weights.sum()
        if weights_sum != 1.0:
            weights = weights / weights_sum
        
        # Sort weights in descending order
        sorted_weights = weights.sort_values(ascending=False)
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        hhi = (weights ** 2).sum()
        
        # Calculate Gini coefficient
        cum_weights = np.cumsum(sorted_weights)
        lorenz_curve = np.insert(cum_weights, 0, 0)
        n = len(lorenz_curve)
        x = np.linspace(0, 1, n)
        gini = 1 - 2 * np.trapz(lorenz_curve, x)
        
        # Calculate top-N concentration
        top1 = sorted_weights.iloc[0] if len(sorted_weights) >= 1 else np.nan
        top5 = sorted_weights.iloc[:5].sum() if len(sorted_weights) >= 5 else sorted_weights.sum()
        top10 = sorted_weights.iloc[:10].sum() if len(sorted_weights) >= 10 else sorted_weights.sum()
        
        # Calculate effective N (inverse of HHI)
        effective_n = 1 / hhi
        
        return {
            'herfindahl_index': hhi,
            'gini_coefficient': gini,
            'top1_concentration': top1,
            'top5_concentration': top5,
            'top10_concentration': top10,
            'effective_n': effective_n
        }
    
    except Exception as e:
        logger.error(f"Error calculating concentration metrics: {str(e)}")
        return {
            'herfindahl_index': np.nan,
            'gini_coefficient': np.nan,
            'top1_concentration': np.nan,
            'top5_concentration': np.nan,
            'top10_concentration': np.nan,
            'effective_n': np.nan
        }


def calculate_portfolio_exposure(
    positions: pd.DataFrame,
    value_column: str = 'value',
    side_column: str = 'side',
    equity_column: Optional[str] = None
) -> Dict[str, float]:
    """
    Calculate portfolio exposure metrics.
    
    Args:
        positions: DataFrame of positions
        value_column: Name of position value column
        side_column: Name of side column
        equity_column: Name of equity column for leverage calculation
        
    Returns:
        Dictionary of exposure metrics
    """
    if positions is None or positions.empty:
        return {
            'gross_exposure': np.nan,
            'net_exposure': np.nan,
            'long_exposure': np.nan,
            'short_exposure': np.nan,
            'leverage': np.nan
        }
    
    try:
        # Check if we have required columns
        required_columns = [value_column, side_column]
        if not all(col in positions.columns for col in required_columns):
            logger.error("Missing required columns for portfolio exposure calculation")
            return {
                'gross_exposure': np.nan,
                'net_exposure': np.nan,
                'long_exposure': np.nan,
                'short_exposure': np.nan,
                'leverage': np.nan
            }
        
        # Calculate long and short exposure
        long_positions = positions[positions[side_column].str.upper() == 'LONG']
        short_positions = positions[positions[side_column].str.upper() == 'SHORT']
        
        long_exposure = long_positions[value_column].sum()
        short_exposure = short_positions[value_column].abs().sum()
        
        # Calculate gross and net exposure
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        
        # Calculate leverage if equity column provided
        leverage = np.nan
        if equity_column in positions.columns and positions[equity_column].sum() > 0:
            equity = positions[equity_column].sum()
            leverage = gross_exposure / equity
        
        return {
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'leverage': leverage
        }
    
    except Exception as e:
        logger.error(f"Error calculating portfolio exposure: {str(e)}")
        return {
            'gross_exposure': np.nan,
            'net_exposure': np.nan,
            'long_exposure': np.nan,
            'short_exposure': np.nan,
            'leverage': np.nan
        }


def calculate_positions_summary(
    positions: pd.DataFrame,
    value_column: str = 'value',
    price_column: Optional[str] = 'price',
    size_column: Optional[str] = 'position_size',
    side_column: Optional[str] = 'side',
    side_values: Tuple[str, str] = ('LONG', 'SHORT')
) -> Dict[str, Any]:
    """
    Calculate summary statistics for positions.
    
    Args:
        positions: DataFrame of positions
        value_column: Name of position value column
        price_column: Optional name of price column
        size_column: Optional name of position size column
        side_column: Optional name of side column
        side_values: Tuple of (long_value, short_value) for side column
        
    Returns:
        Dictionary of position summary statistics
    """
    if positions is None or positions.empty:
        return {
            'total_positions': 0,
            'total_value': 0.0,
            'avg_position_size': np.nan,
            'max_position_size': np.nan,
            'min_position_size': np.nan,
            'long_positions': np.nan,
            'short_positions': np.nan,
            'long_value': np.nan,
            'short_value': np.nan
        }
    
    try:
        # Check if we have required columns
        if value_column not in positions.columns:
            logger.error(f"Missing required column: {value_column}")
            return {
                'total_positions': len(positions),
                'total_value': np.nan,
                'avg_position_size': np.nan,
                'max_position_size': np.nan,
                'min_position_size': np.nan,
                'long_positions': np.nan,
                'short_positions': np.nan,
                'long_value': np.nan,
                'short_value': np.nan
            }
        
        # Calculate basic summary statistics
        total_positions = len(positions)
        total_value = positions[value_column].abs().sum()
        
        # Calculate position sizes if available
        if size_column in positions.columns:
            avg_position_size = positions[size_column].abs().mean()
            max_position_size = positions[size_column].abs().max()
            min_position_size = positions[size_column].abs().min()
        else:
            avg_position_size = total_value / total_positions
            max_position_size = positions[value_column].abs().max()
            min_position_size = positions[value_column].abs().min()
        
        # Calculate long/short breakdown if side column available
        if side_column in positions.columns:
            long_value = side_values[0]
            short_value = side_values[1]
            
            long_positions = positions[positions[side_column] == long_value]
            short_positions = positions[positions[side_column] == short_value]
            
            long_count = len(long_positions)
            short_count = len(short_positions)
            
            long_value = long_positions[value_column].sum()
            short_value = short_positions[value_column].abs().sum()
        else:
            long_count = np.nan
            short_count = np.nan
            long_value = np.nan
            short_value = np.nan
        
        return {
            'total_positions': total_positions,
            'total_value': total_value,
            'avg_position_size': avg_position_size,
            'max_position_size': max_position_size,
            'min_position_size': min_position_size,
            'long_positions': long_count,
            'short_positions': short_count,
            'long_value': long_value,
            'short_value': short_value
        }
    
    except Exception as e:
        logger.error(f"Error calculating positions summary: {str(e)}")
        return {
            'total_positions': len(positions),
            'total_value': np.nan,
            'avg_position_size': np.nan,
            'max_position_size': np.nan,
            'min_position_size': np.nan,
            'long_positions': np.nan,
            'short_positions': np.nan,
            'long_value': np.nan,
            'short_value': np.nan
        }


# ========== Convenience Functions ==========

def calculate_all_performance_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    trades: Optional[pd.DataFrame] = None
) -> PerformanceMetrics:
    """
    Calculate all performance metrics in one function.
    
    Args:
        returns: Return series
        benchmark_returns: Optional benchmark return series
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year
        trades: Optional DataFrame of trades
        
    Returns:
        PerformanceMetrics object with all calculated metrics
    """
    if returns is None or returns.empty:
        return PerformanceMetrics()
    
    try:
        metrics = PerformanceMetrics()
        
        # Clean returns series
        returns = returns.dropna()
        
        # Calculate equity curve
        equity_curve = calculate_cumulative_returns(returns)
        
        # Return metrics
        metrics.total_return = calculate_total_return(returns)
        metrics.annualized_return = calculate_annualized_return(returns, periods_per_year)
        metrics.cagr = calculate_cagr(equity_curve, periods_per_year)
        
        # Risk metrics
        metrics.volatility = calculate_volatility(returns, True, periods_per_year)
        metrics.downside_deviation = calculate_downside_deviation(returns, 0.0, True, periods_per_year)
        
        drawdown_series, max_dd, max_dd_duration = calculate_drawdowns(equity_curve)
        metrics.max_drawdown = max_dd
        metrics.max_drawdown_duration = max_dd_duration
        
        metrics.var = calculate_var(returns, 0.95, 'historical')
        metrics.cvar = calculate_expected_shortfall(returns, 0.95, 'historical')
        
        # Risk-adjusted metrics
        metrics.sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        metrics.sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate, 0.0, periods_per_year)
        metrics.calmar_ratio = calculate_calmar_ratio(returns, 36, periods_per_year)
        
        # Market metrics if benchmark provided
        if benchmark_returns is not None and not benchmark_returns.empty:
            metrics.beta = calculate_beta(returns, benchmark_returns)
            metrics.alpha = calculate_alpha(returns, benchmark_returns, risk_free_rate, periods_per_year)
            metrics.r_squared = calculate_r_squared(returns, benchmark_returns)
            metrics.tracking_error = calculate_tracking_error(returns, benchmark_returns, True, periods_per_year)
            metrics.information_ratio = calculate_information_ratio(returns, benchmark_returns, periods_per_year)
        
        # Trade metrics if trades provided
        if trades is not None and not trades.empty:
            trade_stats = calculate_trade_statistics(trades)
            metrics.trade_count = trade_stats['trade_count']
            metrics.win_rate = trade_stats['win_rate']
            metrics.loss_rate = trade_stats['loss_rate']
            metrics.avg_profit = trade_stats['avg_profit']
            metrics.avg_loss = trade_stats['avg_loss']
            metrics.profit_factor = trade_stats['profit_factor']
            metrics.avg_win_loss_ratio = trade_stats['avg_win_loss_ratio']
        
        # Distribution metrics
        if len(returns) > 2:
            metrics.skewness = returns.skew()
            metrics.kurtosis = returns.kurtosis()
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating all performance metrics: {str(e)}")
        return PerformanceMetrics()