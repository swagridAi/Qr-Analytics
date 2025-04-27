"""
Performance view component for the Quant Research dashboard.

This module provides a comprehensive performance visualization component
with equity curves, drawdowns, and performance metrics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Union, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PerformancePeriod(str, Enum):
    """Time periods for performance visualization."""
    ALL = "all"
    MONTH_1 = "1m"
    MONTH_3 = "3m"
    MONTH_6 = "6m"
    YEAR_1 = "1y"
    YTD = "ytd"
    
    @classmethod
    def get_offset(cls, period: 'PerformancePeriod') -> pd.DateOffset:
        """Get the pandas DateOffset for a period."""
        if period == cls.MONTH_1:
            return pd.DateOffset(months=1)
        elif period == cls.MONTH_3:
            return pd.DateOffset(months=3)
        elif period == cls.MONTH_6:
            return pd.DateOffset(months=6)
        elif period == cls.YEAR_1:
            return pd.DateOffset(years=1)
        elif period == cls.YTD:
            return pd.DateOffset(year=pd.Timestamp.now().year, month=1, day=1)
        else:  # ALL
            return pd.DateOffset(years=100)  # Large enough to include all data


class TradeAnnotationType(str, Enum):
    """Types of trade annotations on equity curve."""
    NONE = "none"           # No trade annotations
    MARKERS = "markers"     # Simple markers for entries/exits
    DETAILED = "detailed"   # Detailed markers with hover text
    TRANSFERS = "transfers" # Lines connecting entry and exit
    

@dataclass
class PerformanceViewConfig:
    """Configuration settings for performance visualization."""
    
    # Display settings
    chart_height: int = 700
    chart_width: Optional[int] = None
    theme: str = "white"  # "white" or "dark"
    
    # Subplot settings
    show_drawdown: bool = True
    show_underwater: bool = True
    show_returns: bool = False
    show_histogram: bool = False
    
    # Data settings
    period: PerformancePeriod = PerformancePeriod.ALL
    risk_free_rate: float = 0.02  # Annual, for Sharpe/Sortino calculation
    baseline_return: float = 0.0  # For Jensen's alpha calculation
    rebase_to_100: bool = True    # Start equity curve at 100
    log_scale: bool = False       # Use log scale for equity
    
    # Trade visualization settings
    show_trades: bool = True
    trade_annotation_type: TradeAnnotationType = TradeAnnotationType.MARKERS
    max_trade_annotations: int = 100  # Limit number of trade annotations
    
    # Benchmark settings
    show_benchmark: bool = True
    benchmark_as_relative: bool = False  # Display as relative performance
    
    # Color settings
    equity_colors: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"
    ])
    benchmark_color: str = "#7f7f7f"
    drawdown_color: str = "rgba(255, 0, 0, 0.7)"
    underwater_color: str = "rgba(255, 165, 0, 0.7)"
    returns_positive_color: str = "rgba(0, 128, 0, 0.7)"
    returns_negative_color: str = "rgba(255, 0, 0, 0.7)"
    trade_colors: Dict[str, str] = field(default_factory=lambda: {
        "win": "rgba(0, 128, 0, 0.7)",
        "loss": "rgba(255, 0, 0, 0.7)",
        "entry": "blue",
        "exit": "purple"
    })
    
    # Layout settings
    show_grid: bool = True
    show_legend: bool = True
    
    # Metrics settings
    show_metrics_table: bool = False  # Whether to include metrics table on chart
    metrics_to_show: List[str] = field(default_factory=lambda: [
        "total_return", "annualized_return", "sharpe_ratio", 
        "sortino_ratio", "max_drawdown", "win_rate"
    ])
    
    # Advanced settings
    rolling_window: int = 20  # For rolling metrics like rolling Sharpe
    show_rolling_metrics: bool = False
    custom_annotations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Container for calculated performance metrics."""
    
    # Return metrics
    total_return: float = None
    annualized_return: float = None
    cumulative_returns: pd.Series = None
    
    # Risk metrics
    volatility: float = None
    downside_deviation: float = None
    max_drawdown: float = None
    max_drawdown_duration: int = None
    
    # Risk-adjusted metrics
    sharpe_ratio: float = None
    sortino_ratio: float = None
    calmar_ratio: float = None
    
    # Trade metrics
    win_rate: float = None
    profit_factor: float = None
    avg_win: float = None
    avg_loss: float = None
    avg_trade: float = None
    
    # Advanced metrics
    value_at_risk: float = None
    expected_shortfall: float = None
    skewness: float = None
    kurtosis: float = None
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary for display."""
        result = {}
        
        # Format and filter the metrics
        if self.total_return is not None:
            result['total_return'] = f"{self.total_return:.2%}"
            
        if self.annualized_return is not None:
            result['annualized_return'] = f"{self.annualized_return:.2%}"
            
        if self.volatility is not None:
            result['volatility'] = f"{self.volatility:.2%}"
            
        if self.max_drawdown is not None:
            result['max_drawdown'] = f"{self.max_drawdown:.2%}"
            
        if self.max_drawdown_duration is not None:
            result['max_drawdown_duration'] = f"{self.max_drawdown_duration} days"
            
        if self.sharpe_ratio is not None:
            result['sharpe_ratio'] = f"{self.sharpe_ratio:.2f}"
            
        if self.sortino_ratio is not None:
            result['sortino_ratio'] = f"{self.sortino_ratio:.2f}"
            
        if self.calmar_ratio is not None:
            result['calmar_ratio'] = f"{self.calmar_ratio:.2f}"
            
        if self.win_rate is not None:
            result['win_rate'] = f"{self.win_rate:.2%}"
            
        if self.profit_factor is not None:
            result['profit_factor'] = f"{self.profit_factor:.2f}"
            
        if self.avg_win is not None:
            result['avg_win'] = f"{self.avg_win:.2f}"
            
        if self.avg_loss is not None:
            result['avg_loss'] = f"{self.avg_loss:.2f}"
            
        if self.avg_trade is not None:
            result['avg_trade'] = f"{self.avg_trade:.2f}"
            
        if self.value_at_risk is not None:
            result['value_at_risk'] = f"{self.value_at_risk:.2%}"
            
        if self.expected_shortfall is not None:
            result['expected_shortfall'] = f"{self.expected_shortfall:.2%}"
            
        if self.skewness is not None:
            result['skewness'] = f"{self.skewness:.2f}"
            
        if self.kurtosis is not None:
            result['kurtosis'] = f"{self.kurtosis:.2f}"
            
        return result


class PerformanceView:
    """
    Class for creating and managing performance visualizations.
    
    This class handles data validation, figure creation, and adding
    different visualizations for performance analysis.
    """
    
    def __init__(
        self,
        performance_data: pd.DataFrame,
        trades_data: Optional[pd.DataFrame] = None,
        benchmark_data: Optional[pd.DataFrame] = None,
        selected_symbols: Optional[List[str]] = None,
        config: Optional[PerformanceViewConfig] = None
    ):
        """
        Initialize the performance view with data and configuration.
        
        Args:
            performance_data: DataFrame containing equity/returns with columns:
                [timestamp, equity, returns(optional), symbol(optional)]
            trades_data: Optional DataFrame with trade data:
                [timestamp, symbol(optional), type, price, size, pnl(optional)]
            benchmark_data: Optional DataFrame with benchmark performance
            selected_symbols: List of symbols to display
            config: Optional configuration object
        
        Raises:
            ValueError: If required data columns are missing
        """
        self.performance_data = performance_data.copy()
        self.trades_data = trades_data.copy() if trades_data is not None else None
        self.benchmark_data = benchmark_data.copy() if benchmark_data is not None else None
        self.selected_symbols = selected_symbols
        self.config = config or PerformanceViewConfig()
        
        # Initialize chart state
        self.fig = None
        self.date_range = None
        self.current_row = 0
        self.subplots = {}  # Track which row contains which subplot
        
        # Initialize metrics container
        self.metrics = {}  # Dict of symbol -> PerformanceMetrics
        
        # Validate input data
        self._validate_data()
        
    def _validate_data(self) -> None:
        """
        Validate the input data has required columns.
        
        Raises:
            ValueError: If required columns are missing
        """
        if self.performance_data.empty:
            return
            
        required_columns = ['timestamp', 'equity']
        
        missing_columns = [col for col in required_columns if col not in self.performance_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in performance data: {missing_columns}")
        
        # Filter for selected symbols if provided
        if self.selected_symbols and 'symbol' in self.performance_data.columns:
            self.performance_data = self.performance_data[
                self.performance_data['symbol'].isin(self.selected_symbols)
            ]
            
            # Ensure we have data after filtering
            if self.performance_data.empty:
                raise ValueError(f"No performance data found for selected symbols: {self.selected_symbols}")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.performance_data['timestamp']):
            self.performance_data['timestamp'] = pd.to_datetime(self.performance_data['timestamp'])
        
        # Filter data by period if specified
        if self.config.period != PerformancePeriod.ALL:
            self._filter_by_period()
            
        # Calculate returns if not provided
        if 'returns' not in self.performance_data.columns:
            self._calculate_returns()
            
        # Rebase equity to 100 if requested
        if self.config.rebase_to_100:
            self._rebase_equity()
            
        # Handle trades data if provided
        if self.trades_data is not None and not self.trades_data.empty:
            trades_required = ['timestamp', 'type']
            trades_missing = [col for col in trades_required if col not in self.trades_data.columns]
            if trades_missing:
                self.trades_data = None  # Don't use invalid trades data
            else:
                # Filter trades for selected symbols if needed
                if self.selected_symbols and 'symbol' in self.trades_data.columns:
                    self.trades_data = self.trades_data[
                        self.trades_data['symbol'].isin(self.selected_symbols)
                    ]
                
                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(self.trades_data['timestamp']):
                    self.trades_data['timestamp'] = pd.to_datetime(self.trades_data['timestamp'])
                    
                # Filter trades by the same period as performance data
                if self.date_range:
                    self.trades_data = self.trades_data[
                        (self.trades_data['timestamp'] >= self.date_range[0]) &
                        (self.trades_data['timestamp'] <= self.date_range[1])
                    ]
        
        # Handle benchmark data if provided
        if self.benchmark_data is not None and not self.benchmark_data.empty:
            benchmark_required = ['timestamp', 'equity']
            benchmark_missing = [col for col in benchmark_required if col not in self.benchmark_data.columns]
            if benchmark_missing:
                self.benchmark_data = None  # Don't use invalid benchmark data
            else:
                # Filter benchmark for selected symbols if needed
                if self.selected_symbols and 'symbol' in self.benchmark_data.columns:
                    self.benchmark_data = self.benchmark_data[
                        self.benchmark_data['symbol'].isin(self.selected_symbols)
                    ]
                
                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(self.benchmark_data['timestamp']):
                    self.benchmark_data['timestamp'] = pd.to_datetime(self.benchmark_data['timestamp'])
                    
                # Filter benchmark by the same period as performance data
                if self.date_range:
                    self.benchmark_data = self.benchmark_data[
                        (self.benchmark_data['timestamp'] >= self.date_range[0]) &
                        (self.benchmark_data['timestamp'] <= self.date_range[1])
                    ]
                    
                # Rebase benchmark if rebasing equity
                if self.config.rebase_to_100:
                    self._rebase_benchmark()
    
    def _filter_by_period(self) -> None:
        """Filter performance data by the selected period."""
        if self.config.period == PerformancePeriod.ALL:
            return
            
        # Get the latest date in the data
        latest_date = self.performance_data['timestamp'].max()
        
        # Calculate the start date based on the period
        if self.config.period == PerformancePeriod.YTD:
            # Year to date - start from January 1st of current year
            start_date = pd.Timestamp(latest_date.year, 1, 1)
        else:
            # Use the period offset
            offset = PerformancePeriod.get_offset(self.config.period)
            start_date = latest_date - offset
        
        # Filter the data
        self.performance_data = self.performance_data[
            self.performance_data['timestamp'] >= start_date
        ]
        
        # Update date range
        self.date_range = [start_date, latest_date]
    
    def _calculate_returns(self) -> None:
        """Calculate returns from equity data if not provided."""
        # Group by symbol if multiple symbols
        if 'symbol' in self.performance_data.columns:
            grouped = self.performance_data.groupby('symbol')
            returns_dfs = []
            
            for symbol, group in grouped:
                # Sort by timestamp
                group = group.sort_values('timestamp')
                
                # Calculate returns
                group['returns'] = group['equity'].pct_change()
                
                returns_dfs.append(group)
            
            self.performance_data = pd.concat(returns_dfs)
        else:
            # Sort by timestamp
            self.performance_data = self.performance_data.sort_values('timestamp')
            
            # Calculate returns
            self.performance_data['returns'] = self.performance_data['equity'].pct_change()
    
    def _rebase_equity(self) -> None:
        """Rebase equity values to start at 100."""
        # Group by symbol if multiple symbols
        if 'symbol' in self.performance_data.columns:
            grouped = self.performance_data.groupby('symbol')
            rebased_dfs = []
            
            for symbol, group in grouped:
                # Sort by timestamp
                group = group.sort_values('timestamp')
                
                # Get initial equity
                initial_equity = group['equity'].iloc[0]
                
                # Rebase to 100
                group['equity'] = group['equity'] / initial_equity * 100
                
                rebased_dfs.append(group)
            
            self.performance_data = pd.concat(rebased_dfs)
        else:
            # Sort by timestamp
            self.performance_data = self.performance_data.sort_values('timestamp')
            
            # Get initial equity
            initial_equity = self.performance_data['equity'].iloc[0]
            
            # Rebase to 100
            self.performance_data['equity'] = self.performance_data['equity'] / initial_equity * 100
    
    def _rebase_benchmark(self) -> None:
        """Rebase benchmark equity values to match strategy rebasing."""
        # Group by symbol if multiple symbols
        if 'symbol' in self.benchmark_data.columns:
            grouped = self.benchmark_data.groupby('symbol')
            rebased_dfs = []
            
            for symbol, group in grouped:
                # Sort by timestamp
                group = group.sort_values('timestamp')
                
                # Get initial equity
                initial_equity = group['equity'].iloc[0]
                
                # Rebase to 100
                group['equity'] = group['equity'] / initial_equity * 100
                
                rebased_dfs.append(group)
            
            self.benchmark_data = pd.concat(rebased_dfs)
        else:
            # Sort by timestamp
            self.benchmark_data = self.benchmark_data.sort_values('timestamp')
            
            # Get initial equity
            initial_equity = self.benchmark_data['equity'].iloc[0]
            
            # Rebase to 100
            self.benchmark_data['equity'] = self.benchmark_data['equity'] / initial_equity * 100
    
    def create_figure(self) -> go.Figure:
        """
        Create the complete performance visualization with all components.
        
        Returns:
            Plotly Figure object with all requested visualization elements
        
        Raises:
            RuntimeError: If figure creation fails
        """
        if self.performance_data.empty:
            # Return empty figure with message
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No performance data available for the selected parameters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return empty_fig
        
        try:
            # Determine number of subplots and setup layout
            self._setup_subplots()
            
            # Calculate performance metrics
            self._calculate_metrics()
            
            # Add equity curve
            self._add_equity_curve()
            
            # Add benchmark if available and requested
            if self.config.show_benchmark and self.benchmark_data is not None and not self.benchmark_data.empty:
                self._add_benchmark()
            
            # Add trades if available and requested
            if self.config.show_trades and self.trades_data is not None and not self.trades_data.empty:
                self._add_trade_annotations()
            
            # Add returns subplot if requested
            if self.config.show_returns and 'returns' in self.performance_data.columns:
                self._add_returns_subplot()
            
            # Add drawdown subplot if requested
            if self.config.show_drawdown:
                self._add_drawdown_subplot()
            
            # Add underwater equity subplot if requested
            if self.config.show_underwater:
                self._add_underwater_subplot()
            
            # Add return histogram if requested
            if self.config.show_histogram and 'returns' in self.performance_data.columns:
                self._add_return_histogram()
            
            # Add rolling metrics if requested
            if self.config.show_rolling_metrics:
                self._add_rolling_metrics()
            
            # Add metrics table if requested
            if self.config.show_metrics_table:
                self._add_metrics_table()
            
            # Add custom annotations
            self._add_custom_annotations()
            
            # Update layout
            self._update_layout()
            
            return self.fig
            
        except Exception as e:
            # Create error figure
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Error creating performance visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return error_fig
    
    def _setup_subplots(self) -> None:
        """Set up the subplot structure based on configuration."""
        # Count how many subplot rows we need
        num_rows = 1  # Main equity curve
        
        # Add rows for additional plots
        if self.config.show_returns:
            num_rows += 1
            
        if self.config.show_drawdown:
            num_rows += 1
            
        if self.config.show_underwater:
            num_rows += 1
            
        if self.config.show_histogram:
            num_rows += 1
        
        # Track which subplot is in which row
        self.subplots = {"equity": 1}
        current_row = 2
        
        if self.config.show_returns:
            self.subplots["returns"] = current_row
            current_row += 1
            
        if self.config.show_drawdown:
            self.subplots["drawdown"] = current_row
            current_row += 1
            
        if self.config.show_underwater:
            self.subplots["underwater"] = current_row
            current_row += 1
            
        if self.config.show_histogram:
            self.subplots["histogram"] = current_row
        
        # Calculate row heights - main chart gets more space
        if num_rows == 1:
            row_heights = [1]
        elif num_rows == 2:
            row_heights = [0.7, 0.3]
        elif num_rows == 3:
            row_heights = [0.6, 0.2, 0.2]
        elif num_rows == 4:
            row_heights = [0.5, 0.17, 0.17, 0.16]
        else:
            # Distribute remaining space evenly
            main_height = 0.5
            remaining = (1 - main_height) / (num_rows - 1)
            row_heights = [main_height] + [remaining] * (num_rows - 1)
        
        # Create subplot titles
        subplot_titles = ["Equity Curve"]
        
        if self.config.show_returns:
            subplot_titles.append("Returns")
            
        if self.config.show_drawdown:
            subplot_titles.append("Drawdown")
            
        if self.config.show_underwater:
            subplot_titles.append("Underwater Equity")
            
        if self.config.show_histogram:
            subplot_titles.append("Return Distribution")
        
        # Create figure with subplots
        self.fig = make_subplots(
            rows=num_rows,
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )
    
    def _calculate_metrics(self) -> None:
        """Calculate performance metrics for all symbols."""
        # If we have multiple symbols, calculate metrics for each
        if 'symbol' in self.performance_data.columns:
            for symbol in self.performance_data['symbol'].unique():
                symbol_data = self.performance_data[self.performance_data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                
                # Get trades for this symbol if available
                symbol_trades = None
                if self.trades_data is not None and not self.trades_data.empty:
                    if 'symbol' in self.trades_data.columns:
                        symbol_trades = self.trades_data[self.trades_data['symbol'] == symbol]
                
                # Calculate metrics
                self.metrics[symbol] = self._calculate_symbol_metrics(symbol_data, symbol_trades)
        else:
            # Single symbol or no symbol column
            self.metrics['overall'] = self._calculate_symbol_metrics(self.performance_data, self.trades_data)
    
    def _calculate_symbol_metrics(self, data: pd.DataFrame, trades: Optional[pd.DataFrame] = None) -> PerformanceMetrics:
        """
        Calculate performance metrics for a single symbol.
        
        Args:
            data: DataFrame containing performance data for one symbol
            trades: Optional DataFrame containing trades for this symbol
            
        Returns:
            PerformanceMetrics object with calculated metrics
        """
        metrics = PerformanceMetrics()
        
        if data.empty:
            return metrics
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Get equity series
        equity = data['equity']
        
        # Get returns series if available
        if 'returns' in data.columns:
            returns = data['returns'].dropna()
        else:
            returns = equity.pct_change().dropna()
        
        # Calculate total return
        first_equity = equity.iloc[0]
        last_equity = equity.iloc[-1]
        total_return = (last_equity / first_equity) - 1
        metrics.total_return = total_return
        
        # Calculate annualized return
        if len(data) > 1:
            start_date = data['timestamp'].iloc[0]
            end_date = data['timestamp'].iloc[-1]
            years = (end_date - start_date).days / 365.25
            
            if years > 0:
                annualized_return = (1 + total_return) ** (1 / years) - 1
                metrics.annualized_return = annualized_return
        
        # Calculate risk metrics if we have returns
        if len(returns) > 1:
            # Volatility (annualized)
            vol = returns.std() * np.sqrt(252)  # Assuming daily returns
            metrics.volatility = vol
            
            # Downside deviation (annualized)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(252)
                metrics.downside_deviation = downside_deviation
            
            # Risk-adjusted metrics
            if vol > 0:
                # Sharpe ratio
                excess_returns = returns - (self.config.risk_free_rate / 252)  # Daily risk-free rate
                sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
                metrics.sharpe_ratio = sharpe
            
            if metrics.downside_deviation and metrics.downside_deviation > 0:
                # Sortino ratio
                sortino = (returns.mean() - (self.config.risk_free_rate / 252)) / downside_returns.std() * np.sqrt(252)
                metrics.sortino_ratio = sortino
            
            # Return distribution metrics
            if len(returns) > 2:  # Need at least 3 points for skewness/kurtosis
                metrics.skewness = returns.skew()
                metrics.kurtosis = returns.kurtosis()
            
            # Value at Risk (95% confidence)
            metrics.value_at_risk = np.percentile(returns, 5)
            
            # Expected Shortfall (CVaR)
            if len(returns[returns <= metrics.value_at_risk]) > 0:
                metrics.expected_shortfall = returns[returns <= metrics.value_at_risk].mean()
        
        # Calculate drawdown metrics
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        metrics.max_drawdown = max_drawdown
        
        # Find max drawdown duration
        if max_drawdown < 0:
            # Find the peak before the max drawdown
            max_dd_idx = drawdown.idxmin()
            # Find the last peak before the max drawdown
            peak_idx = rolling_max.loc[:max_dd_idx].idxmax()
            # Find the recovery (if any)
            recovery_idx = None
            after_dd = drawdown.loc[max_dd_idx:]
            recovery = after_dd[after_dd >= 0]
            if not recovery.empty:
                recovery_idx = recovery.index[0]
            
            # Calculate duration
            if recovery_idx:
                # From peak to recovery
                duration = (data.loc[recovery_idx, 'timestamp'] - data.loc[peak_idx, 'timestamp']).days
            else:
                # From peak to end, if no recovery
                duration = (data.loc[max_dd_idx, 'timestamp'] - data.loc[peak_idx, 'timestamp']).days
            
            metrics.max_drawdown_duration = duration
        
        # Calculate Calmar ratio if we have max drawdown and annualized return
        if metrics.annualized_return is not None and metrics.max_drawdown is not None and metrics.max_drawdown != 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)
        
        # Calculate trade metrics if trades are available
        if trades is not None and not trades.empty and 'pnl' in trades.columns:
            # Win rate
            winning_trades = trades[trades['pnl'] > 0]
            losing_trades = trades[trades['pnl'] < 0]
            
            win_count = len(winning_trades)
            total_trades = len(trades)
            
            if total_trades > 0:
                metrics.win_rate = win_count / total_trades
            
            # Profit factor
            if len(winning_trades) > 0 and len(losing_trades) > 0:
                total_profits = winning_trades['pnl'].sum()
                total_losses = abs(losing_trades['pnl'].sum())
                
                if total_losses > 0:
                    metrics.profit_factor = total_profits / total_losses
            
            # Average trade metrics
            if len(winning_trades) > 0:
                metrics.avg_win = winning_trades['pnl'].mean()
            
            if len(losing_trades) > 0:
                metrics.avg_loss = abs(losing_trades['pnl'].mean())
            
            if total_trades > 0:
                metrics.avg_trade = trades['pnl'].mean()
        
        return metrics
    
    def _add_equity_curve(self) -> None:
        """Add equity curve traces to the figure."""
        # If we have multiple symbols, create separate traces for each
        if 'symbol' in self.performance_data.columns and len(self.performance_data['symbol'].unique()) > 1:
            for i, symbol in enumerate(sorted(self.performance_data['symbol'].unique())):
                symbol_data = self.performance_data[self.performance_data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                    
                color = self.config.equity_colors[i % len(self.config.equity_colors)]
                self._add_equity_trace(symbol_data, symbol, color)
        else:
            # Single symbol or no symbol column
            self._add_equity_trace(
                self.performance_data, 
                None, 
                self.config.equity_colors[0]
            )

    def _add_equity_trace(self, data: pd.DataFrame, symbol: Optional[str] = None, 
                         color: str = "blue") -> None:
        """
        Add an equity curve trace to the figure.
        
        Args:
            data: DataFrame containing equity data
            symbol: Optional symbol name for the trace
            color: Color for the trace
        """
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        trace_name = f"{symbol + ' ' if symbol else ''}Equity"
        
        # Determine y-axis type (log or linear)
        yaxis_type = "log" if self.config.log_scale else "linear"
        
        self.fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['equity'],
                name=trace_name,
                line=dict(width=2, color=color),
                showlegend=True
            ),
            row=self.subplots["equity"], col=1
        )
        
        # Update y-axis type
        self.fig.update_yaxes(
            type=yaxis_type,
            row=self.subplots["equity"], col=1
        )
        
        # Update date range
        if self.date_range is None:
            self.date_range = [data['timestamp'].min(), data['timestamp'].max()]
        else:
            self.date_range[0] = min(self.date_range[0], data['timestamp'].min())
            self.date_range[1] = max(self.date_range[1], data['timestamp'].max())

    def _add_benchmark(self) -> None:
        """Add benchmark comparison to the equity curve."""
        if self.benchmark_data is None or self.benchmark_data.empty:
            return
            
        # If we have multiple symbols in benchmark, create separate traces for each
        if ('symbol' in self.benchmark_data.columns and 
            len(self.benchmark_data['symbol'].unique()) > 1):
            
            for symbol in sorted(self.benchmark_data['symbol'].unique()):
                symbol_data = self.benchmark_data[self.benchmark_data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                    
                if self.config.benchmark_as_relative:
                    self._add_relative_benchmark_trace(symbol_data, symbol)
                else:
                    self._add_benchmark_trace(symbol_data, symbol)
        else:
            # Single symbol or no symbol column
            if self.config.benchmark_as_relative:
                self._add_relative_benchmark_trace(self.benchmark_data)
            else:
                self._add_benchmark_trace(self.benchmark_data)

    def _add_benchmark_trace(self, data: pd.DataFrame, symbol: Optional[str] = None) -> None:
        """
        Add a benchmark equity trace to the figure.
        
        Args:
            data: DataFrame containing benchmark equity data
            symbol: Optional symbol name for the trace
        """
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        trace_name = f"{symbol + ' ' if symbol else ''}Benchmark"
        
        self.fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['equity'],
                name=trace_name,
                line=dict(width=1.5, color=self.config.benchmark_color, dash='dash'),
                showlegend=True
            ),
            row=self.subplots["equity"], col=1
        )

    def _add_relative_benchmark_trace(self, benchmark_data: pd.DataFrame, 
                                     symbol: Optional[str] = None) -> None:
        """
        Add a relative performance trace (strategy vs benchmark).
        
        Args:
            benchmark_data: DataFrame containing benchmark equity data
            symbol: Optional symbol name for the trace
        """
        # Find corresponding strategy data
        if 'symbol' in self.performance_data.columns and symbol:
            strategy_data = self.performance_data[self.performance_data['symbol'] == symbol]
        else:
            strategy_data = self.performance_data
        
        if strategy_data.empty or benchmark_data.empty:
            return
        
        # Align timestamps
        merged = pd.merge_asof(
            strategy_data.sort_values('timestamp')[['timestamp', 'equity']],
            benchmark_data.sort_values('timestamp')[['timestamp', 'equity']],
            on='timestamp',
            direction='nearest',
            suffixes=('_strategy', '_benchmark')
        )
        
        if merged.empty:
            return
        
        # Calculate relative performance
        merged['relative'] = merged['equity_strategy'] / merged['equity_benchmark'] * 100
        
        trace_name = f"{symbol + ' ' if symbol else ''}Relative Performance"
        
        # Add relative performance trace
        self.fig.add_trace(
            go.Scatter(
                x=merged['timestamp'],
                y=merged['relative'],
                name=trace_name,
                line=dict(width=1.5, color=self.config.benchmark_color),
                showlegend=True
            ),
            row=self.subplots["equity"], col=1
        )
        
        # Add a reference line at 100 (equal performance)
        self.fig.add_hline(
            y=100, 
            line_dash="dash", 
            line_color="gray",
            row=self.subplots["equity"], col=1
        )

    def _add_trade_annotations(self) -> None:
        """Add trade markers to the equity curve."""
        if self.trades_data is None or self.trades_data.empty:
            return
            
        # Get row for equity curve
        equity_row = self.subplots["equity"]
        
        # Limit the number of annotations if needed
        trades = self.trades_data
        if len(trades) > self.config.max_trade_annotations:
            # Sample trades to limit annotations
            trades = trades.sample(self.config.max_trade_annotations, random_state=42)
        
        # Add annotations based on selected type
        if self.config.trade_annotation_type == TradeAnnotationType.NONE:
            return
        elif self.config.trade_annotation_type == TradeAnnotationType.DETAILED:
            self._add_detailed_trade_markers(trades, equity_row)
        elif self.config.trade_annotation_type == TradeAnnotationType.TRANSFERS:
            self._add_trade_transfer_lines(trades, equity_row)
        else:  # MARKERS (default)
            self._add_simple_trade_markers(trades, equity_row)

    def _add_simple_trade_markers(self, trades: pd.DataFrame, row: int) -> None:
        """
        Add simple trade markers to the equity curve.
        
        Args:
            trades: DataFrame containing trade data
            row: Row index for the equity subplot
        """
        # Group trades by type
        if 'type' in trades.columns:
            for trade_type in trades['type'].unique():
                type_trades = trades[trades['type'] == trade_type]
                
                if type_trades.empty:
                    continue
                
                # Select marker properties based on type
                if trade_type.upper() in ['BUY', 'ENTRY', 'LONG']:
                    marker_symbol = 'triangle-up'
                    marker_color = self.config.trade_colors.get('entry', 'blue')
                    name = 'Entry'
                elif trade_type.upper() in ['SELL', 'EXIT', 'SHORT']:
                    marker_symbol = 'triangle-down'
                    marker_color = self.config.trade_colors.get('exit', 'purple')
                    name = 'Exit'
                else:
                    marker_symbol = 'circle'
                    marker_color = 'gray'
                    name = trade_type.capitalize()
                
                # Get y-values at each trade timestamp
                y_values = []
                for idx, trade in type_trades.iterrows():
                    # Find equity value at or nearest to trade timestamp
                    nearest_equity = self._get_equity_at_timestamp(trade['timestamp'])
                    y_values.append(nearest_equity)
                
                # Add marker trace
                self.fig.add_trace(
                    go.Scatter(
                        x=type_trades['timestamp'],
                        y=y_values,
                        mode='markers',
                        name=name,
                        marker=dict(
                            symbol=marker_symbol,
                            size=8,
                            color=marker_color,
                            line=dict(width=1, color='darkgray')
                        ),
                        showlegend=True
                    ),
                    row=row, col=1
                )
        else:
            # If no type column, just use generic markers
            y_values = []
            for idx, trade in trades.iterrows():
                nearest_equity = self._get_equity_at_timestamp(trade['timestamp'])
                y_values.append(nearest_equity)
            
            self.fig.add_trace(
                go.Scatter(
                    x=trades['timestamp'],
                    y=y_values,
                    mode='markers',
                    name='Trades',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color='gray',
                        line=dict(width=1, color='darkgray')
                    ),
                    showlegend=True
                ),
                row=row, col=1
            )

    def _add_detailed_trade_markers(self, trades: pd.DataFrame, row: int) -> None:
        """
        Add detailed trade markers with PnL information.
        
        Args:
            trades: DataFrame containing trade data
            row: Row index for the equity subplot
        """
        # We need PnL information for detailed markers
        if 'pnl' not in trades.columns:
            # Fall back to simple markers
            self._add_simple_trade_markers(trades, row)
            return
        
        # Separate winning and losing trades
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        # Get y-values for winning trades
        win_y_values = []
        for idx, trade in winning_trades.iterrows():
            nearest_equity = self._get_equity_at_timestamp(trade['timestamp'])
            win_y_values.append(nearest_equity)
        
        # Get y-values for losing trades
        lose_y_values = []
        for idx, trade in losing_trades.iterrows():
            nearest_equity = self._get_equity_at_timestamp(trade['timestamp'])
            lose_y_values.append(nearest_equity)
        
        # Create hover text with trade details
        win_text = [
            f"Symbol: {trade.get('symbol', 'N/A')}<br>"
            f"Type: {trade.get('type', 'Trade')}<br>"
            f"Time: {pd.to_datetime(trade['timestamp']).strftime('%Y-%m-%d %H:%M')}<br>"
            f"PnL: +{trade['pnl']:.2f}"
            for _, trade in winning_trades.iterrows()
        ]
        
        lose_text = [
            f"Symbol: {trade.get('symbol', 'N/A')}<br>"
            f"Type: {trade.get('type', 'Trade')}<br>"
            f"Time: {pd.to_datetime(trade['timestamp']).strftime('%Y-%m-%d %H:%M')}<br>"
            f"PnL: {trade['pnl']:.2f}"
            for _, trade in losing_trades.iterrows()
        ]
        
        # Add winning trades
        if not winning_trades.empty:
            self.fig.add_trace(
                go.Scatter(
                    x=winning_trades['timestamp'],
                    y=win_y_values,
                    mode='markers',
                    name='Winning Trades',
                    text=win_text,
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color=self.config.trade_colors.get('win', 'green'),
                        line=dict(width=1, color='darkgreen')
                    ),
                    hoverinfo='text',
                    showlegend=True
                ),
                row=row, col=1
            )
        
        # Add losing trades
        if not losing_trades.empty:
            self.fig.add_trace(
                go.Scatter(
                    x=losing_trades['timestamp'],
                    y=lose_y_values,
                    mode='markers',
                    name='Losing Trades',
                    text=lose_text,
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color=self.config.trade_colors.get('loss', 'red'),
                        line=dict(width=1, color='darkred')
                    ),
                    hoverinfo='text',
                    showlegend=True
                ),
                row=row, col=1
            )

    def _add_trade_transfer_lines(self, trades: pd.DataFrame, row: int) -> None:
        """
        Add lines connecting entry and exit trades.
        
        Args:
            trades: DataFrame containing trade data
            row: Row index for the equity subplot
        """
        # We need trade type information to identify entries and exits
        if 'type' not in trades.columns:
            # Fall back to simple markers
            self._add_simple_trade_markers(trades, row)
            return
        
        # Add the basic markers first
        self._add_simple_trade_markers(trades, row)
        
        # Now try to pair entries with exits
        # This is a simplified approach assuming sequential entries and exits
        entry_types = ['BUY', 'ENTRY', 'LONG']
        exit_types = ['SELL', 'EXIT', 'SHORT']
        
        # Filter for just entries and exits
        entries = trades[trades['type'].str.upper().isin(entry_types)].sort_values('timestamp')
        exits = trades[trades['type'].str.upper().isin(exit_types)].sort_values('timestamp')
        
        # Only continue if we have both entries and exits
        if entries.empty or exits.empty:
            return
        
        # Try to pair entries with the next exit
        # This is a simplified approach and won't work correctly for complex trading
        pairs = []
        current_entries = entries.copy()
        
        for _, exit_trade in exits.iterrows():
            if current_entries.empty:
                break
                
            # Find the closest entry before this exit
            valid_entries = current_entries[current_entries['timestamp'] < exit_trade['timestamp']]
            
            if valid_entries.empty:
                continue
                
            # Get the most recent entry
            entry_trade = valid_entries.iloc[-1]
            
            # Add the pair
            pairs.append((entry_trade, exit_trade))
            
            # Remove this entry from consideration
            current_entries = current_entries[current_entries.index != entry_trade.name]
        
        # Create connection lines for each pair
        for entry_trade, exit_trade in pairs:
            entry_equity = self._get_equity_at_timestamp(entry_trade['timestamp'])
            exit_equity = self._get_equity_at_timestamp(exit_trade['timestamp'])
            
            # Determine color based on PnL if available
            if 'pnl' in exit_trade:
                color = (self.config.trade_colors.get('win', 'green') 
                         if exit_trade['pnl'] > 0 
                         else self.config.trade_colors.get('loss', 'red'))
            else:
                color = 'gray'
            
            # Add connection line
            self.fig.add_trace(
                go.Scatter(
                    x=[entry_trade['timestamp'], exit_trade['timestamp']],
                    y=[entry_equity, exit_equity],
                    mode='lines',
                    line=dict(color=color, width=1, dash='dot'),
                    showlegend=False
                ),
                row=row, col=1
            )

    def _get_equity_at_timestamp(self, timestamp: pd.Timestamp) -> float:
        """
        Find the equity value at or nearest to a given timestamp.
        
        Args:
            timestamp: Timestamp to find equity value for
            
        Returns:
            Equity value at or near the timestamp
        """
        # Filter by symbol first if we have multiple symbols
        if 'symbol' in self.performance_data.columns:
            # Try to get the symbol from trades_data if timestamp matches exactly
            if self.trades_data is not None and 'symbol' in self.trades_data.columns:
                exact_match = self.trades_data[self.trades_data['timestamp'] == timestamp]
                if not exact_match.empty:
                    trade_symbol = exact_match.iloc[0]['symbol']
                    symbol_data = self.performance_data[self.performance_data['symbol'] == trade_symbol]
                    
                    if not symbol_data.empty:
                        # Find nearest timestamp in performance data
                        idx = (symbol_data['timestamp'] - timestamp).abs().idxmin()
                        return symbol_data.loc[idx, 'equity']
            
            # If we couldn't find the symbol, use the first symbol's data
            symbol = self.performance_data['symbol'].unique()[0]
            data = self.performance_data[self.performance_data['symbol'] == symbol]
        else:
            data = self.performance_data
        
        # Find nearest timestamp in performance data
        idx = (data['timestamp'] - timestamp).abs().idxmin()
        return data.loc[idx, 'equity']

    def _add_returns_subplot(self) -> None:
        """Add returns subplot to the figure."""
        if 'returns' not in self.performance_data.columns:
            return
            
        returns_row = self.subplots.get("returns")
        if not returns_row:
            return
            
        # If we have multiple symbols, create separate traces for each
        if 'symbol' in self.performance_data.columns and len(self.performance_data['symbol'].unique()) > 1:
            for i, symbol in enumerate(sorted(self.performance_data['symbol'].unique())):
                symbol_data = self.performance_data[self.performance_data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                    
                color = self.config.equity_colors[i % len(self.config.equity_colors)]
                self._add_returns_trace(symbol_data, returns_row, symbol, color)
        else:
            # Single symbol or no symbol column
            self._add_returns_trace(
                self.performance_data, 
                returns_row, 
                None, 
                self.config.equity_colors[0]
            )

    def _add_returns_trace(self, data: pd.DataFrame, row: int, 
                          symbol: Optional[str] = None, color: str = "blue") -> None:
        """
        Add a returns trace to the figure.
        
        Args:
            data: DataFrame containing returns data
            row: Row index for the subplot
            symbol: Optional symbol name for the trace
            color: Color for the trace
        """
        if 'returns' not in data.columns:
            return
            
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        trace_name = f"{symbol + ' ' if symbol else ''}Returns"
        
        # Convert returns to percentage for display
        returns_pct = data['returns'] * 100
        
        # Create colors based on positive/negative returns
        colors = np.where(
            returns_pct >= 0,
            self.config.returns_positive_color,
            self.config.returns_negative_color
        )
        
        self.fig.add_trace(
            go.Bar(
                x=data['timestamp'],
                y=returns_pct,
                name=trace_name,
                marker_color=colors,
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Add a reference line at zero
        self.fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="gray",
            row=row, col=1
        )

    def _add_drawdown_subplot(self) -> None:
        """Add drawdown subplot to the figure."""
        drawdown_row = self.subplots.get("drawdown")
        if not drawdown_row:
            return
            
        # If we have multiple symbols, create separate traces for each
        if 'symbol' in self.performance_data.columns and len(self.performance_data['symbol'].unique()) > 1:
            for i, symbol in enumerate(sorted(self.performance_data['symbol'].unique())):
                symbol_data = self.performance_data[self.performance_data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                    
                color = self.config.equity_colors[i % len(self.config.equity_colors)]
                self._add_drawdown_trace(symbol_data, drawdown_row, symbol, color)
        else:
            # Single symbol or no symbol column
            self._add_drawdown_trace(
                self.performance_data, 
                drawdown_row, 
                None, 
                self.config.drawdown_color
            )

    def _add_drawdown_trace(self, data: pd.DataFrame, row: int, 
                           symbol: Optional[str] = None, color: str = "red") -> None:
        """
        Add a drawdown trace to the figure.
        
        Args:
            data: DataFrame containing equity data
            row: Row index for the subplot
            symbol: Optional symbol name for the trace
            color: Color for the trace
        """
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Calculate drawdown series
        equity = data['equity']
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100  # Convert to percentage
        
        trace_name = f"{symbol + ' ' if symbol else ''}Drawdown"
        
        self.fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=drawdown,
                name=trace_name,
                fill='tozeroy',
                line=dict(color=color),
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Add a reference line at zero
        self.fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="gray",
            row=row, col=1
        )

    def _add_underwater_subplot(self) -> None:
        """Add underwater equity subplot to the figure."""
        underwater_row = self.subplots.get("underwater")
        if not underwater_row:
            return
            
        # If we have multiple symbols, create separate traces for each
        if 'symbol' in self.performance_data.columns and len(self.performance_data['symbol'].unique()) > 1:
            for i, symbol in enumerate(sorted(self.performance_data['symbol'].unique())):
                symbol_data = self.performance_data[self.performance_data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                    
                color = self.config.equity_colors[i % len(self.config.equity_colors)]
                self._add_underwater_trace(symbol_data, underwater_row, symbol, color)
        else:
            # Single symbol or no symbol column
            self._add_underwater_trace(
                self.performance_data, 
                underwater_row, 
                None, 
                self.config.underwater_color
            )

    def _add_underwater_trace(self, data: pd.DataFrame, row: int, 
                             symbol: Optional[str] = None, color: str = "orange") -> None:
        """
        Add an underwater equity trace to the figure.
        
        Args:
            data: DataFrame containing equity data
            row: Row index for the subplot
            symbol: Optional symbol name for the trace
            color: Color for the trace
        """
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Calculate underwater equity
        equity = data['equity']
        rolling_max = equity.cummax()
        underwater = equity / rolling_max  # Ratio (1.0 = at high watermark)
        
        trace_name = f"{symbol + ' ' if symbol else ''}Underwater Equity"
        
        self.fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=underwater,
                name=trace_name,
                fill='tozeroy',
                line=dict(color=color),
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Add a reference line at 1.0 (high watermark)
        self.fig.add_hline(
            y=1.0, 
            line_dash="dash", 
            line_color="gray",
            row=row, col=1
        )

    def _add_return_histogram(self) -> None:
        """Add return distribution histogram to the figure."""
        if 'returns' not in self.performance_data.columns:
            return
            
        histogram_row = self.subplots.get("histogram")
        if not histogram_row:
            return
            
        # If we have multiple symbols, create separate traces for each
        if 'symbol' in self.performance_data.columns and len(self.performance_data['symbol'].unique()) > 1:
            for i, symbol in enumerate(sorted(self.performance_data['symbol'].unique())):
                symbol_data = self.performance_data[self.performance_data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                    
                color = self.config.equity_colors[i % len(self.config.equity_colors)]
                self._add_histogram_trace(symbol_data, histogram_row, symbol, color)
        else:
            # Single symbol or no symbol column
            self._add_histogram_trace(
                self.performance_data, 
                histogram_row, 
                None, 
                self.config.equity_colors[0]
            )

    def _add_histogram_trace(self, data: pd.DataFrame, row: int, 
                            symbol: Optional[str] = None, color: str = "blue") -> None:
        """
        Add a return distribution histogram to the figure.
        
        Args:
            data: DataFrame containing returns data
            row: Row index for the subplot
            symbol: Optional symbol name for the trace
            color: Color for the trace
        """
        if 'returns' not in data.columns:
            return
            
        # Convert returns to percentage for display
        returns_pct = data['returns'] * 100
        
        trace_name = f"{symbol + ' ' if symbol else ''}Return Distribution"
        
        self.fig.add_trace(
            go.Histogram(
                x=returns_pct,
                name=trace_name,
                marker_color=color,
                opacity=0.7,
                nbinsx=30,
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Add a reference line at zero
        self.fig.add_vline(
            x=0, 
            line_dash="dash", 
            line_color="gray",
            row=row, col=1
        )
        
        # Add normal distribution overlay if enough data points
        if len(returns_pct) > 30:
            # Calculate mean and standard deviation
            mean = returns_pct.mean()
            std = returns_pct.std()
            
            # Generate x values for normal curve
            x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
            
            # Calculate PDF values
            from scipy.stats import norm
            y = norm.pdf(x, mean, std)
            
            # Scale to match histogram
            hist_values, _ = np.histogram(returns_pct, bins=30)
            max_hist = hist_values.max()
            y_scaled = y * max_hist / y.max()
            
            self.fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_scaled,
                    mode='lines',
                    name='Normal Dist.',
                    line=dict(color='red', width=1),
                    showlegend=True
                ),
                row=row, col=1
            )

    def _add_rolling_metrics(self) -> None:
        """Add rolling performance metrics to the equity subplot."""
        if 'returns' not in self.performance_data.columns:
            return
            
        # We'll add this to the equity subplot
        equity_row = self.subplots["equity"]
        
        # If we have multiple symbols, create separate traces for each
        if 'symbol' in self.performance_data.columns and len(self.performance_data['symbol'].unique()) > 1:
            for i, symbol in enumerate(sorted(self.performance_data['symbol'].unique())):
                symbol_data = self.performance_data[self.performance_data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                    
                color = self.config.equity_colors[i % len(self.config.equity_colors)]
                self._add_rolling_sharpe_trace(symbol_data, equity_row, symbol, color)
        else:
            # Single symbol or no symbol column
            self._add_rolling_sharpe_trace(
                self.performance_data, 
                equity_row, 
                None, 
                self.config.equity_colors[0]
            )

    def _add_rolling_sharpe_trace(self, data: pd.DataFrame, row: int, 
                                 symbol: Optional[str] = None, color: str = "blue") -> None:
        """
        Add a rolling Sharpe ratio trace to the figure.
        
        Args:
            data: DataFrame containing returns data
            row: Row index for the subplot
            symbol: Optional symbol name for the trace
            color: Color for the trace
        """
        if 'returns' not in data.columns:
            return
            
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Calculate rolling Sharpe ratio
        window = self.config.rolling_window
        if len(data) <= window:
            return
            
        # Calculate rolling mean and std of returns
        rolling_mean = data['returns'].rolling(window=window).mean()
        rolling_std = data['returns'].rolling(window=window).std()
        
        # Daily risk-free rate
        daily_rf = self.config.risk_free_rate / 252
        
        # Calculate rolling Sharpe ratio (annualized)
        rolling_sharpe = (rolling_mean - daily_rf) / rolling_std * np.sqrt(252)
        
        # Create a secondary y-axis for the Sharpe ratio
        secondary_y = f"y{len(self.fig.layout.annotations) + 1}"
        
        trace_name = f"{symbol + ' ' if symbol else ''}Rolling Sharpe ({window}d)"
        
        self.fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=rolling_sharpe,
                name=trace_name,
                line=dict(width=1, color=color, dash='dot'),
                opacity=0.7,
                yaxis=secondary_y,
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Add the secondary y-axis
        self.fig.update_layout(**{
            secondary_y: dict(
                title="Rolling Sharpe Ratio",
                overlaying="y",
                side="right",
                showgrid=False
            )
        })

    def _add_metrics_table(self) -> None:
        """Add a performance metrics table to the figure."""
        if not self.metrics:
            return
            
        # Create a metrics table for each symbol or one overall table
        if len(self.metrics) > 1:
            # Multiple symbols - create a table for the first one only
            # (too cluttered to show all)
            symbol = sorted(self.metrics.keys())[0]
            metrics = self.metrics[symbol]
            title = f"{symbol} Performance Metrics"
        else:
            # Single set of metrics
            metrics = next(iter(self.metrics.values()))
            title = "Performance Metrics"
        
        # Get metrics as a dictionary
        metrics_dict = metrics.as_dict()
        
        # Filter metrics based on config
        if self.config.metrics_to_show:
            metrics_dict = {k: v for k, v in metrics_dict.items() if k in self.config.metrics_to_show}
        
        # Create table data
        table_data = [
            ["Metric", "Value"],
            *[[k.replace('_', ' ').title(), v] for k, v in metrics_dict.items()]
        ]
        
        # Add table to the top right of the figure
        self.fig.add_table(
            header=dict(
                values=table_data[0],
                fill_color='rgba(0, 0, 0, 0.1)',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[[row[0] for row in table_data[1:]], 
                        [row[1] for row in table_data[1:]]],
                fill_color='rgba(255, 255, 255, 0.7)',
                align='left',
                font=dict(size=11)
            ),
            domain=dict(x=[0.7, 0.99], y=[0.75, 0.99])
        )
        
        # Add title for the table
        self.fig.add_annotation(
            xref="paper", yref="paper",
            x=0.85, y=1.0,
            text=title,
            showarrow=False,
            font=dict(size=12, color="black"),
            align="center"
        )

    def _add_custom_annotations(self) -> None:
        """Add custom annotations to the figure."""
        for annotation in self.config.custom_annotations:
            row = annotation.get('row', 1)
            text = annotation.get('text', '')
            x = annotation.get('x', 0.5)
            y = annotation.get('y', 0.5)
            
            self.fig.add_annotation(
                xref="paper" if annotation.get('xref') == 'paper' else 'x',
                yref="paper" if annotation.get('yref') == 'paper' else 'y',
                x=x,
                y=y,
                text=text,
                showarrow=annotation.get('showarrow', False),
                arrowhead=annotation.get('arrowhead', 1),
                font=dict(
                    size=annotation.get('font_size', 12),
                    color=annotation.get('font_color', 'black')
                ),
                bgcolor=annotation.get('bgcolor', 'rgba(255, 255, 255, 0.7)'),
                bordercolor=annotation.get('bordercolor', 'gray'),
                borderwidth=annotation.get('borderwidth', 1),
                borderpad=annotation.get('borderpad', 4),
                align=annotation.get('align', 'center'),
                row=row,
                col=1
            )

    def _update_layout(self) -> None:
        """Update the figure layout with titles and formatting."""
        # Set chart title
        symbols_text = f" ({', '.join(self.selected_symbols)})" if self.selected_symbols else ""
        title = f"Performance Analysis{symbols_text}"
        
        # Base layout updates
        layout_updates = dict(
            title=title,
            height=self.config.chart_height,
            hovermode='x unified',
            margin=dict(l=50, r=50, t=70, b=50)
        )
        
        # Add width if specified
        if self.config.chart_width:
            layout_updates["width"] = self.config.chart_width
        
        # Configure theme
        if self.config.theme == "dark":
            layout_updates.update(
                template="plotly_dark",
                paper_bgcolor="rgba(0, 0, 0, 0.8)",
                plot_bgcolor="rgba(0, 0, 0, 0.8)"
            )
        
        # Configure legend
        if self.config.show_legend:
            layout_updates["legend"] = dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        else:
            layout_updates["showlegend"] = False
        
        # Apply layout updates
        self.fig.update_layout(**layout_updates)
        
        # Update grid settings for all subplots
        self.fig.update_xaxes(
            title_text="Date", 
            showgrid=self.config.show_grid
        )
        
        # Update y-axis labels for each subplot
        self.fig.update_yaxes(
            title_text="Equity" + (" (log scale)" if self.config.log_scale else ""),
            showgrid=self.config.show_grid,
            row=self.subplots["equity"], col=1
        )
        
        if "returns" in self.subplots:
            self.fig.update_yaxes(
                title_text="Returns (%)",
                showgrid=self.config.show_grid,
                row=self.subplots["returns"], col=1
            )
        
        if "drawdown" in self.subplots:
            self.fig.update_yaxes(
                title_text="Drawdown (%)",
                showgrid=self.config.show_grid,
                row=self.subplots["drawdown"], col=1
            )
        
        if "underwater" in self.subplots:
            self.fig.update_yaxes(
                title_text="Underwater Ratio",
                showgrid=self.config.show_grid,
                row=self.subplots["underwater"], col=1
            )
        
        if "histogram" in self.subplots:
            self.fig.update_xaxes(
                title_text="Return (%)",
                showgrid=self.config.show_grid,
                row=self.subplots["histogram"], col=1
            )
            self.fig.update_yaxes(
                title_text="Frequency",
                showgrid=self.config.show_grid,
                row=self.subplots["histogram"], col=1
            )
        
        # Set consistent date range for time-series subplots
        if self.date_range:
            for row in range(1, len(self.subplots)):
                # Skip histogram subplot which has returns, not dates, on x-axis
                if self.subplots.get("histogram") != row + 1:
                    self.fig.update_xaxes(range=self.date_range, row=row, col=1)


def create_performance_view(
    performance_data: pd.DataFrame,
    trades_data: Optional[pd.DataFrame] = None,
    benchmark_data: Optional[pd.DataFrame] = None,
    selected_symbols: Optional[List[str]] = None,
    show_drawdown: bool = True,
    show_underwater: bool = True,
    period: str = "all",
    show_trades: bool = True,
    chart_height: int = 700,
    rebase_to_100: bool = True,
    log_scale: bool = False
) -> go.Figure:
    """
    Create an interactive performance visualization with equity curve and performance metrics.
    
    This function provides a simplified interface to the PerformanceView class.
    
    Args:
        performance_data: DataFrame containing equity/returns data
        trades_data: Optional DataFrame with trade data
        benchmark_data: Optional DataFrame with benchmark performance
        selected_symbols: List of symbols to display
        show_drawdown: Whether to show drawdown subplot
        show_underwater: Whether to show underwater equity subplot
        period: Time period to display ("all", "1m", "3m", "6m", "1y", "ytd")
        show_trades: Whether to show trade markers
        chart_height: Height of the chart in pixels
        rebase_to_100: Whether to rebase equity to start at 100
        log_scale: Whether to use logarithmic scale for equity
        
    Returns:
        Plotly Figure object
    """
    # Map period string to enum
    try:
        period_enum = PerformancePeriod(period)
    except ValueError:
        period_enum = PerformancePeriod.ALL
    
    # Create configuration
    config = PerformanceViewConfig(
        chart_height=chart_height,
        show_drawdown=show_drawdown,
        show_underwater=show_underwater,
        period=period_enum,
        show_trades=show_trades,
        rebase_to_100=rebase_to_100,
        log_scale=log_scale
    )
    
    # Create view instance
    view = PerformanceView(
        performance_data=performance_data,
        trades_data=trades_data,
        benchmark_data=benchmark_data,
        selected_symbols=selected_symbols,
        config=config
    )
    
    # Create and return figure
    return view.create_figure()


def get_performance_metrics(
    performance_data: pd.DataFrame,
    trades_data: Optional[pd.DataFrame] = None,
    selected_symbols: Optional[List[str]] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate performance metrics for the given data.
    
    Args:
        performance_data: DataFrame containing equity/returns data
        trades_data: Optional DataFrame with trade data
        selected_symbols: List of symbols to calculate metrics for
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino
        
    Returns:
        Dictionary mapping symbols to dictionaries of metrics
    """
    # Create a temporary view to calculate metrics
    config = PerformanceViewConfig(risk_free_rate=risk_free_rate)
    
    view = PerformanceView(
        performance_data=performance_data,
        trades_data=trades_data,
        selected_symbols=selected_symbols,
        config=config
    )
    
    # Calculate metrics without creating a figure
    view._validate_data()
    view._calculate_metrics()
    
    # Convert PerformanceMetrics objects to dictionaries
    return {k: v.as_dict() for k, v in view.metrics.items()}