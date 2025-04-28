"""
Financial Performance Metrics

This module provides functions for calculating financial performance metrics
such as Sharpe ratio, Sortino ratio, drawdowns, and other risk-adjusted
return measures.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.statistics.performance")

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