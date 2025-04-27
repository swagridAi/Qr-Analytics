"""
Portfolio view component for the Quant Research dashboard.

This module provides a comprehensive portfolio visualization component
with position analysis, allocation breakdown, and risk metrics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Union, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class RiskMetricType(str, Enum):
    """Types of risk metrics for portfolio analysis."""
    VALUE_AT_RISK = "var"
    EXPECTED_SHORTFALL = "es"
    VOLATILITY = "vol"
    BETA = "beta"
    TRACKING_ERROR = "te"


class AllocationBreakdownType(str, Enum):
    """Types of portfolio allocation breakdowns."""
    ASSET = "asset"           # By individual asset
    SECTOR = "sector"         # By sector
    MARKET_CAP = "market_cap" # By market cap (large, mid, small)
    REGION = "region"         # By geographic region
    STRATEGY = "strategy"     # By trading strategy
    CUSTOM = "custom"         # Custom grouping


@dataclass
class PortfolioViewConfig:
    """Configuration settings for portfolio visualization."""
    
    # Display settings
    chart_height: int = 700
    chart_width: Optional[int] = None
    theme: str = "white"  # "white" or "dark"
    
    # Subplot settings
    show_allocation: bool = True
    show_risk_metrics: bool = True
    show_correlation: bool = False
    show_positions_table: bool = True
    show_historical_allocation: bool = False
    
    # Allocation settings
    allocation_type: AllocationBreakdownType = AllocationBreakdownType.ASSET
    allocation_chart_type: str = "pie"  # "pie", "bar", "treemap"
    custom_grouping_field: Optional[str] = None  # For custom grouping
    
    # Risk settings
    risk_metrics: List[RiskMetricType] = field(default_factory=lambda: [
        RiskMetricType.VALUE_AT_RISK,
        RiskMetricType.VOLATILITY
    ])
    risk_confidence_level: float = 0.95  # For VaR and ES
    risk_horizon_days: int = 1           # Risk horizon
    
    # Position settings
    show_long_short_split: bool = True
    min_position_display: float = 0.01  # Min position size to display (as fraction)
    
    # Correlation settings
    correlation_lookback: int = 60  # Days for correlation calculation
    correlation_method: str = "pearson"  # "pearson", "spearman", "kendall"
    
    # Color settings
    position_colors: Dict[str, str] = field(default_factory=lambda: {
        "long": "rgba(0, 128, 0, 0.7)",
        "short": "rgba(255, 0, 0, 0.7)",
        "cash": "rgba(192, 192, 192, 0.7)"
    })
    sector_colors: Dict[str, str] = field(default_factory=lambda: {
        "Technology": "#1f77b4",
        "Financial": "#ff7f0e",
        "Healthcare": "#2ca02c",
        "Consumer": "#d62728",
        "Industrial": "#9467bd",
        "Energy": "#8c564b",
        "Materials": "#e377c2",
        "Utilities": "#7f7f7f",
        "Real Estate": "#bcbd22",
        "Telecom": "#17becf",
        "Other": "#7f7f7f"
    })
    allocation_colors: List[str] = field(default_factory=lambda: px.colors.qualitative.Plotly)
    risk_bar_colors: Dict[str, str] = field(default_factory=lambda: {
        "var": "rgba(255, 0, 0, 0.7)",
        "es": "rgba(255, 165, 0, 0.7)",
        "vol": "rgba(65, 105, 225, 0.7)",
        "beta": "rgba(128, 0, 128, 0.7)",
        "te": "rgba(0, 128, 128, 0.7)"
    })
    
    # Layout settings
    show_grid: bool = True
    show_legend: bool = True
    
    # Advanced settings
    exclude_cash_from_allocation: bool = False
    normalize_weights: bool = True  # Ensure weights sum to 1
    show_tickers: bool = True      # Show ticker symbols in labels
    max_position_table_rows: int = 20  # Max rows in position table
    currency_symbol: str = "$"     # Currency symbol for displaying values


@dataclass
class PortfolioRiskMetrics:
    """Container for calculated portfolio risk metrics."""
    
    # Value at Risk metrics
    value_at_risk: float = None
    expected_shortfall: float = None
    conditional_var: float = None
    
    # Volatility metrics
    volatility: float = None
    downside_volatility: float = None
    
    # Exposure metrics
    gross_exposure: float = None
    net_exposure: float = None
    leverage: float = None
    
    # Factor metrics
    beta: float = None
    tracking_error: float = None
    
    # Concentration metrics
    herfindahl_index: float = None
    top_position_weight: float = None
    top_positions_concentration: float = None  # Top 5 positions
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary for display."""
        result = {}
        
        # Format and filter the metrics
        if self.value_at_risk is not None:
            result['value_at_risk'] = f"{self.value_at_risk:.2%}"
            
        if self.expected_shortfall is not None:
            result['expected_shortfall'] = f"{self.expected_shortfall:.2%}"
            
        if self.volatility is not None:
            result['volatility'] = f"{self.volatility:.2%}"
            
        if self.downside_volatility is not None:
            result['downside_volatility'] = f"{self.downside_volatility:.2%}"
            
        if self.gross_exposure is not None:
            result['gross_exposure'] = f"{self.gross_exposure:.2f}x"
            
        if self.net_exposure is not None:
            result['net_exposure'] = f"{self.net_exposure:.2f}x"
            
        if self.leverage is not None:
            result['leverage'] = f"{self.leverage:.2f}x"
            
        if self.beta is not None:
            result['beta'] = f"{self.beta:.2f}"
            
        if self.tracking_error is not None:
            result['tracking_error'] = f"{self.tracking_error:.2%}"
            
        if self.herfindahl_index is not None:
            result['herfindahl_index'] = f"{self.herfindahl_index:.4f}"
            
        if self.top_position_weight is not None:
            result['top_position_weight'] = f"{self.top_position_weight:.2%}"
            
        if self.top_positions_concentration is not None:
            result['top_positions_concentration'] = f"{self.top_positions_concentration:.2%}"
            
        return result


class PortfolioView:
    """
    Class for creating and managing portfolio visualizations.
    
    This class handles data validation, figure creation, and adding
    different visualizations for portfolio analysis.
    """
    
    def __init__(
        self,
        positions_data: pd.DataFrame,
        returns_data: Optional[pd.DataFrame] = None,
        historical_positions: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        config: Optional[PortfolioViewConfig] = None
    ):
        """
        Initialize the portfolio view with data and configuration.
        
        Args:
            positions_data: DataFrame containing current positions with columns:
                [symbol, position_size, price, value, side, sector(optional)]
            returns_data: Optional DataFrame with historical returns:
                [timestamp, symbol, returns]
            historical_positions: Optional DataFrame with historical positions:
                [timestamp, symbol, position_size, value]
            market_data: Optional DataFrame with market/benchmark data:
                [timestamp, symbol, price, returns]
            config: Optional configuration object
        
        Raises:
            ValueError: If required data columns are missing
        """
        self.positions_data = positions_data.copy()
        self.returns_data = returns_data.copy() if returns_data is not None else None
        self.historical_positions = historical_positions.copy() if historical_positions is not None else None
        self.market_data = market_data.copy() if market_data is not None else None
        self.config = config or PortfolioViewConfig()
        
        # Initialize chart state
        self.fig = None
        self.current_row = 0
        self.subplots = {}  # Track which row contains which subplot
        
        # Initialize metrics container
        self.risk_metrics = PortfolioRiskMetrics()
        
        # Validate input data
        self._validate_data()
        
    def _validate_data(self) -> None:
        """
        Validate the input data has required columns.
        
        Raises:
            ValueError: If required data columns are missing
        """
        if self.positions_data.empty:
            return
            
        # Required columns for positions data
        required_columns = ['symbol', 'position_size']
        
        missing_columns = [col for col in required_columns if col not in self.positions_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in positions data: {missing_columns}")
        
        # Ensure we have a value column - calculate if missing
        if 'value' not in self.positions_data.columns:
            if 'price' in self.positions_data.columns:
                self.positions_data['value'] = self.positions_data['position_size'] * self.positions_data['price']
            else:
                raise ValueError("Either 'value' or 'price' column is required in positions data")
                
        # Ensure we have a side column - infer from position_size if missing
        if 'side' not in self.positions_data.columns:
            self.positions_data['side'] = np.where(
                self.positions_data['position_size'] >= 0, 
                'LONG', 
                'SHORT'
            )
            
        # If allocation type is by sector, ensure we have a sector column
        if (self.config.allocation_type == AllocationBreakdownType.SECTOR and 
            'sector' not in self.positions_data.columns):
            self.positions_data['sector'] = 'Unknown'
            
        # If custom grouping is selected, ensure the field exists
        if (self.config.allocation_type == AllocationBreakdownType.CUSTOM and 
            self.config.custom_grouping_field and
            self.config.custom_grouping_field not in self.positions_data.columns):
            raise ValueError(f"Custom grouping field '{self.config.custom_grouping_field}' not found in positions data")
        
        # Validate returns data if provided
        if self.returns_data is not None and not self.returns_data.empty:
            returns_required = ['timestamp', 'symbol', 'returns']
            returns_missing = [col for col in returns_required if col not in self.returns_data.columns]
            if returns_missing:
                self.returns_data = None  # Don't use invalid returns data
            else:
                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(self.returns_data['timestamp']):
                    self.returns_data['timestamp'] = pd.to_datetime(self.returns_data['timestamp'])
        
        # Validate historical positions if provided
        if self.historical_positions is not None and not self.historical_positions.empty:
            hist_required = ['timestamp', 'symbol', 'value']
            hist_missing = [col for col in hist_required if col not in self.historical_positions.columns]
            if hist_missing:
                self.historical_positions = None  # Don't use invalid historical data
            else:
                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(self.historical_positions['timestamp']):
                    self.historical_positions['timestamp'] = pd.to_datetime(self.historical_positions['timestamp'])
                    
        # Validate market data if provided
        if self.market_data is not None and not self.market_data.empty:
            market_required = ['timestamp', 'returns']
            market_missing = [col for col in market_required if col not in self.market_data.columns]
            if market_missing:
                self.market_data = None  # Don't use invalid market data
            else:
                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(self.market_data['timestamp']):
                    self.market_data['timestamp'] = pd.to_datetime(self.market_data['timestamp'])
    
    def create_figure(self) -> go.Figure:
        """
        Create the complete portfolio visualization with all components.
        
        Returns:
            Plotly Figure object with all requested visualization elements
        
        Raises:
            RuntimeError: If figure creation fails
        """
        if self.positions_data.empty:
            # Return empty figure with message
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No position data available for the selected parameters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return empty_fig
        
        try:
            # Calculate risk metrics
            self._calculate_risk_metrics()
            
            # Determine number of subplots and setup layout
            self._setup_subplots()
            
            # Add allocation breakdown
            if self.config.show_allocation:
                self._add_allocation_breakdown()
            
            # Add historical allocation if requested
            if self.config.show_historical_allocation and self.historical_positions is not None:
                self._add_historical_allocation()
            
            # Add risk metrics if requested
            if self.config.show_risk_metrics:
                self._add_risk_metrics()
            
            # Add correlation matrix if requested
            if self.config.show_correlation and self.returns_data is not None:
                self._add_correlation_matrix()
            
            # Add positions table if requested
            if self.config.show_positions_table:
                self._add_positions_table()
            
            # Update layout
            self._update_layout()
            
            return self.fig
            
        except Exception as e:
            # Create error figure
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Error creating portfolio visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return error_fig
    
    def _calculate_risk_metrics(self) -> None:
        """Calculate portfolio risk metrics."""
        metrics = PortfolioRiskMetrics()
        
        # Calculate exposure metrics
        total_long = self.positions_data[self.positions_data['side'] == 'LONG']['value'].sum()
        total_short = abs(self.positions_data[self.positions_data['side'] == 'SHORT']['value'].sum())
        total_value = total_long + total_short
        net_value = total_long - total_short
        
        # Get portfolio value - assumption: sum of absolute values for now
        portfolio_value = abs(self.positions_data['value']).sum()
        
        if portfolio_value > 0:
            metrics.gross_exposure = total_value / portfolio_value
            metrics.net_exposure = net_value / portfolio_value
            metrics.leverage = metrics.gross_exposure
        
        # Calculate concentration metrics
        position_weights = abs(self.positions_data['value']) / portfolio_value
        metrics.herfindahl_index = (position_weights ** 2).sum()
        
        if not position_weights.empty:
            metrics.top_position_weight = position_weights.max()
            top_positions = position_weights.nlargest(5)
            metrics.top_positions_concentration = top_positions.sum()
        
        # Calculate VaR and volatility if we have returns data
        if self.returns_data is not None and not self.returns_data.empty:
            self._calculate_var_metrics(metrics)
            self._calculate_volatility_metrics(metrics)
        
        # Calculate beta and tracking error if we have market data
        if self.market_data is not None and not self.market_data.empty:
            self._calculate_beta_metrics(metrics)
        
        self.risk_metrics = metrics
    
    def _calculate_var_metrics(self, metrics: PortfolioRiskMetrics) -> None:
        """
        Calculate Value at Risk related metrics.
        
        Args:
            metrics: PortfolioRiskMetrics object to update
        """
        # We need returns data for portfolio assets
        if self.returns_data is None or self.returns_data.empty:
            return
            
        # Get the current portfolio positions
        current_positions = self.positions_data.copy()
        current_positions_dict = dict(zip(current_positions['symbol'], current_positions['value']))
        
        # Get latest timestamp
        latest_timestamp = self.returns_data['timestamp'].max()
        
        # Filter returns to lookback period
        lookback_days = 60  # Default lookback for VaR calculation
        lookback_date = latest_timestamp - pd.Timedelta(days=lookback_days)
        recent_returns = self.returns_data[self.returns_data['timestamp'] >= lookback_date]
        
        if recent_returns.empty:
            return
            
        # Calculate portfolio returns based on current weights
        # First pivot returns to wide format
        returns_wide = recent_returns.pivot(
            index='timestamp', 
            columns='symbol', 
            values='returns'
        ).fillna(0)
        
        # Only include symbols in our current portfolio
        portfolio_symbols = [s for s in returns_wide.columns if s in current_positions_dict]
        
        if not portfolio_symbols:
            return
            
        returns_wide = returns_wide[portfolio_symbols]
        
        # Calculate portfolio weights
        total_portfolio_value = sum(abs(current_positions_dict[s]) for s in portfolio_symbols)
        
        if total_portfolio_value == 0:
            return
            
        weights = {s: current_positions_dict[s] / total_portfolio_value for s in portfolio_symbols}
        
        # Calculate portfolio returns (vectorized)
        weights_array = np.array([weights.get(s, 0) for s in returns_wide.columns])
        portfolio_returns = returns_wide.dot(weights_array)
        
        # Calculate VaR
        confidence_level = self.config.risk_confidence_level
        var_percentile = 100 * (1 - confidence_level)
        var = np.percentile(portfolio_returns, var_percentile)
        metrics.value_at_risk = abs(var)  # VaR is typically reported as a positive number
        
        # Calculate Expected Shortfall (Conditional VaR)
        if len(portfolio_returns[portfolio_returns <= var]) > 0:
            es = portfolio_returns[portfolio_returns <= var].mean()
            metrics.expected_shortfall = abs(es)
    
    def _calculate_volatility_metrics(self, metrics: PortfolioRiskMetrics) -> None:
        """
        Calculate volatility metrics.
        
        Args:
            metrics: PortfolioRiskMetrics object to update
        """
        # We need returns data for portfolio assets
        if self.returns_data is None or self.returns_data.empty:
            return
            
        # Get the current portfolio positions
        current_positions = self.positions_data.copy()
        current_positions_dict = dict(zip(current_positions['symbol'], current_positions['value']))
        
        # Get latest timestamp
        latest_timestamp = self.returns_data['timestamp'].max()
        
        # Filter returns to lookback period
        lookback_days = 60  # Default lookback for volatility calculation
        lookback_date = latest_timestamp - pd.Timedelta(days=lookback_days)
        recent_returns = self.returns_data[self.returns_data['timestamp'] >= lookback_date]
        
        if recent_returns.empty:
            return
            
        # Calculate portfolio returns based on current weights
        # First pivot returns to wide format
        returns_wide = recent_returns.pivot(
            index='timestamp', 
            columns='symbol', 
            values='returns'
        ).fillna(0)
        
        # Only include symbols in our current portfolio
        portfolio_symbols = [s for s in returns_wide.columns if s in current_positions_dict]
        
        if not portfolio_symbols:
            return
            
        returns_wide = returns_wide[portfolio_symbols]
        
        # Calculate portfolio weights
        total_portfolio_value = sum(abs(current_positions_dict[s]) for s in portfolio_symbols)
        
        if total_portfolio_value == 0:
            return
            
        weights = {s: current_positions_dict[s] / total_portfolio_value for s in portfolio_symbols}
        
        # Calculate portfolio returns (vectorized)
        weights_array = np.array([weights.get(s, 0) for s in returns_wide.columns])
        portfolio_returns = returns_wide.dot(weights_array)
        
        # Calculate volatility
        vol = portfolio_returns.std()
        metrics.volatility = vol
        
        # Calculate downside volatility
        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std()
            metrics.downside_volatility = downside_vol
    
    def _calculate_beta_metrics(self, metrics: PortfolioRiskMetrics) -> None:
        """
        Calculate beta and tracking error metrics.
        
        Args:
            metrics: PortfolioRiskMetrics object to update
        """
        # We need both portfolio returns and market data
        if (self.returns_data is None or self.returns_data.empty or 
            self.market_data is None or self.market_data.empty):
            return
            
        # Get the current portfolio positions
        current_positions = self.positions_data.copy()
        current_positions_dict = dict(zip(current_positions['symbol'], current_positions['value']))
        
        # Get latest timestamp
        latest_timestamp = min(
            self.returns_data['timestamp'].max(),
            self.market_data['timestamp'].max()
        )
        
        # Filter returns to lookback period
        lookback_days = 60  # Default lookback for beta calculation
        lookback_date = latest_timestamp - pd.Timedelta(days=lookback_days)
        
        recent_returns = self.returns_data[self.returns_data['timestamp'] >= lookback_date]
        recent_market = self.market_data[self.market_data['timestamp'] >= lookback_date]
        
        if recent_returns.empty or recent_market.empty:
            return
            
        # Calculate portfolio returns based on current weights
        # First pivot returns to wide format
        returns_wide = recent_returns.pivot(
            index='timestamp', 
            columns='symbol', 
            values='returns'
        ).fillna(0)
        
        # Only include symbols in our current portfolio
        portfolio_symbols = [s for s in returns_wide.columns if s in current_positions_dict]
        
        if not portfolio_symbols:
            return
            
        returns_wide = returns_wide[portfolio_symbols]
        
        # Calculate portfolio weights
        total_portfolio_value = sum(abs(current_positions_dict[s]) for s in portfolio_symbols)
        
        if total_portfolio_value == 0:
            return
            
        weights = {s: current_positions_dict[s] / total_portfolio_value for s in portfolio_symbols}
        
        # Calculate portfolio returns (vectorized)
        weights_array = np.array([weights.get(s, 0) for s in returns_wide.columns])
        portfolio_returns = returns_wide.dot(weights_array)
        
        # Prepare market returns (assumes single market index)
        market_returns = recent_market.set_index('timestamp')['returns']
        
        # Align dates
        aligned_data = pd.concat(
            [portfolio_returns, market_returns], 
            axis=1, 
            join='inner'
        )
        
        if aligned_data.empty or aligned_data.shape[0] < 2:
            return
            
        aligned_data.columns = ['portfolio', 'market']
        
        # Calculate beta using covariance
        cov = aligned_data['portfolio'].cov(aligned_data['market'])
        market_var = aligned_data['market'].var()
        
        if market_var > 0:
            beta = cov / market_var
            metrics.beta = beta
        
        # Calculate tracking error
        active_returns = aligned_data['portfolio'] - aligned_data['market']
        tracking_error = active_returns.std()
        metrics.tracking_error = tracking_error
    
    def _setup_subplots(self) -> None:
        """Set up the subplot structure based on configuration."""
        # Count how many subplot rows we need
        num_rows = 0
        row_heights = []
        subplot_titles = []
        
        # Track which subplot is in which row
        self.subplots = {}
        current_row = 1
        
        if self.config.show_allocation:
            num_rows += 1
            row_heights.append(0.4)
            subplot_titles.append("Portfolio Allocation")
            self.subplots["allocation"] = current_row
            current_row += 1
            
        if self.config.show_historical_allocation and self.historical_positions is not None:
            num_rows += 1
            row_heights.append(0.3)
            subplot_titles.append("Historical Allocation")
            self.subplots["historical"] = current_row
            current_row += 1
            
        if self.config.show_risk_metrics:
            num_rows += 1
            row_heights.append(0.3)
            subplot_titles.append("Risk Metrics")
            self.subplots["risk"] = current_row
            current_row += 1
            
        if self.config.show_correlation and self.returns_data is not None:
            num_rows += 1
            row_heights.append(0.4)
            subplot_titles.append("Correlation Matrix")
            self.subplots["correlation"] = current_row
            current_row += 1
            
        if self.config.show_positions_table:
            num_rows += 1
            row_heights.append(0.3)
            subplot_titles.append("Positions")
            self.subplots["positions"] = current_row
            
        # If no visualizations are enabled, show at least allocation
        if num_rows == 0:
            num_rows = 1
            row_heights = [1.0]
            subplot_titles = ["Portfolio Allocation"]
            self.subplots["allocation"] = 1
            self.config.show_allocation = True
        
        # Normalize heights to sum to 1
        if row_heights:
            total_height = sum(row_heights)
            row_heights = [h / total_height for h in row_heights]
        
        # Create figure with subplots
        self.fig = make_subplots(
            rows=num_rows,
            cols=1, 
            shared_xaxes=False,
            vertical_spacing=0.1,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )
    
    def _add_allocation_breakdown(self) -> None:
        """Add portfolio allocation breakdown visualization."""
        allocation_row = self.subplots.get("allocation")
        if not allocation_row:
            return
            
        # Get allocation data based on the selected type
        allocation_data = self._get_allocation_data()
        
        if allocation_data.empty:
            self._add_empty_message(allocation_row, "No allocation data available")
            return
            
        # Create visualization based on chart type
        if self.config.allocation_chart_type == "pie":
            self._add_allocation_pie(allocation_data, allocation_row)
        elif self.config.allocation_chart_type == "bar":
            self._add_allocation_bar(allocation_data, allocation_row)
        elif self.config.allocation_chart_type == "treemap":
            self._add_allocation_treemap(allocation_data, allocation_row)
        else:
            # Default to pie chart
            self._add_allocation_pie(allocation_data, allocation_row)
    
    def _get_allocation_data(self) -> pd.DataFrame:
        """
        Get allocation data based on the selected breakdown type.
        
        Returns:
            DataFrame with allocation breakdown and values
        """
        # Start with position data
        data = self.positions_data.copy()
        
        # Exclude cash if requested
        if self.config.exclude_cash_from_allocation:
            data = data[data['symbol'] != 'CASH']
            
        if data.empty:
            return pd.DataFrame()
            
        # Get absolute values for allocation
        data['abs_value'] = data['value'].abs()
        
        # Create grouping column based on allocation type
        if self.config.allocation_type == AllocationBreakdownType.ASSET:
            # Group by individual asset
            groupby_col = 'symbol'
        elif self.config.allocation_type == AllocationBreakdownType.SECTOR:
            # Group by sector
            if 'sector' not in data.columns:
                data['sector'] = 'Unknown'
            groupby_col = 'sector'
        elif self.config.allocation_type == AllocationBreakdownType.MARKET_CAP:
            # Group by market cap
            if 'market_cap' not in data.columns:
                data['market_cap'] = 'Unknown'
            groupby_col = 'market_cap'
        elif self.config.allocation_type == AllocationBreakdownType.REGION:
            # Group by region
            if 'region' not in data.columns:
                data['region'] = 'Unknown'
            groupby_col = 'region'
        elif self.config.allocation_type == AllocationBreakdownType.STRATEGY:
            # Group by strategy
            if 'strategy' not in data.columns:
                data['strategy'] = 'Unknown'
            groupby_col = 'strategy'
        elif self.config.allocation_type == AllocationBreakdownType.CUSTOM:
            # Group by custom field
            custom_field = self.config.custom_grouping_field
            if custom_field and custom_field in data.columns:
                groupby_col = custom_field
            else:
                groupby_col = 'symbol'
        else:
            # Default to symbol
            groupby_col = 'symbol'
        
        # Group and calculate allocation
        grouped = data.groupby(groupby_col)['abs_value'].sum().reset_index()
        
        # Calculate percentage
        total_value = grouped['abs_value'].sum()
        if total_value > 0:
            grouped['percentage'] = grouped['abs_value'] / total_value * 100
        else:
            grouped['percentage'] = 0
            
        # Sort by value descending
        grouped = grouped.sort_values('abs_value', ascending=False)
        
        # Format values with currency symbol
        grouped['value_str'] = grouped['abs_value'].apply(
            lambda x: f"{self.config.currency_symbol}{x:,.2f}"
        )
        
        return grouped
    
    def _add_allocation_pie(self, allocation_data: pd.DataFrame, row: int) -> None:
        """
        Add a pie chart for portfolio allocation.
        
        Args:
            allocation_data: DataFrame with allocation breakdown
            row: Row index for the subplot
        """
        # Filter to minimum display size if requested
        min_pct = self.config.min_position_display * 100
        display_data = allocation_data[allocation_data['percentage'] >= min_pct]
        
        # If any items were filtered out, add an "Other" category
        if len(display_data) < len(allocation_data):
            other_value = allocation_data[allocation_data['percentage'] < min_pct]['abs_value'].sum()
            other_pct = allocation_data[allocation_data['percentage'] < min_pct]['percentage'].sum()
            other_row = pd.DataFrame({
                allocation_data.columns[0]: ['Other'],
                'abs_value': [other_value],
                'percentage': [other_pct],
                'value_str': [f"{self.config.currency_symbol}{other_value:,.2f}"]
            })
            display_data = pd.concat([display_data, other_row], ignore_index=True)
        
        # Create hover text
        hover_text = [
            f"{row[0]}<br>Value: {row['value_str']}<br>Weight: {row['percentage']:.2f}%"
            for _, row in display_data.iterrows()
        ]
        
        # Create pie chart
        self.fig.add_trace(
            go.Pie(
                labels=display_data[display_data.columns[0]],
                values=display_data['abs_value'],
                text=display_data['percentage'].apply(lambda x: f"{x:.1f}%"),
                hovertext=hover_text,
                hoverinfo="text",
                textinfo="label+percent",
                marker=dict(colors=self.config.allocation_colors),
                hole=0.3
            ),
            row=row, col=1
        )
    
    def _add_allocation_bar(self, allocation_data: pd.DataFrame, row: int) -> None:
        """
        Add a bar chart for portfolio allocation.
        
        Args:
            allocation_data: DataFrame with allocation breakdown
            row: Row index for the subplot
        """
        # Filter to minimum display size if requested
        min_pct = self.config.min_position_display * 100
        display_data = allocation_data[allocation_data['percentage'] >= min_pct]
        
        # Create bar colors
        colors = self.config.allocation_colors[:len(display_data)]
        
        # Create bar chart
        self.fig.add_trace(
            go.Bar(
                x=display_data[display_data.columns[0]],
                y=display_data['percentage'],
                text=display_data['value_str'],
                hovertext=[
                    f"{row[0]}<br>Value: {row['value_str']}<br>Weight: {row['percentage']:.2f}%"
                    for _, row in display_data.iterrows()
                ],
                hoverinfo="text",
                marker_color=colors,
                textposition='auto'
            ),
            row=row, col=1
        )
        
        # Update y-axis label
        self.fig.update_yaxes(
            title_text="Allocation (%)",
            row=row, col=1
        )
    
    def _add_allocation_treemap(self, allocation_data: pd.DataFrame, row: int) -> None:
        """
        Add a treemap for portfolio allocation.
        
        Args:
            allocation_data: DataFrame with allocation breakdown
            row: Row index for the subplot
        """
        # For treemap, we need to create a separate figure and then add it as an image
        # because Plotly's make_subplots doesn't support treemap directly
        
        # Create treemap figure
        treemap_fig = px.treemap(
            allocation_data,
            path=[allocation_data.columns[0]],
            values='abs_value',
            color='percentage',
            color_continuous_scale=px.colors.sequential.Blues,
            hover_data=['value_str', 'percentage'],
            custom_data=['value_str', 'percentage']
        )
        
        # Update hover template
        treemap_fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Value: %{customdata[0]}<br>Weight: %{customdata[1]:.2f}%',
            texttemplate='<b>%{label}</b><br>%{value:.1f}%'
        )
        
        # Convert treemap to image
        treemap_img = treemap_fig.to_image(format="png")
        
        # Add image to the subplot
        self.fig.add_layout_image(
            dict(
                source=treemap_img,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                sizex=1,
                sizey=1,
                xanchor="center",
                yanchor="middle",
                layer="above"
            ),
            row=row, col=1
        )
    
    def _add_historical_allocation(self) -> None:
        """Add historical allocation visualization."""
        if self.historical_positions is None or self.historical_positions.empty:
            return
            
        historical_row = self.subplots.get("historical")
        if not historical_row:
            return
            
        # Get allocation type for grouping
        if self.config.allocation_type == AllocationBreakdownType.ASSET:
            groupby_col = 'symbol'
        elif self.config.allocation_type == AllocationBreakdownType.SECTOR:
            groupby_col = 'sector'
        elif self.config.allocation_type == AllocationBreakdownType.CUSTOM:
            groupby_col = self.config.custom_grouping_field
        else:
            # Default to symbol for other types
            groupby_col = 'symbol'
            
        # Check if grouping column exists in historical data
        if groupby_col not in self.historical_positions.columns and groupby_col != 'symbol':
            # Try to map from current positions
            mapping = {}
            if groupby_col in self.positions_data.columns:
                for _, row in self.positions_data.iterrows():
                    mapping[row['symbol']] = row[groupby_col]
                    
                # Apply mapping
                self.historical_positions[groupby_col] = self.historical_positions['symbol'].map(
                    mapping
                ).fillna('Unknown')
            else:
                # Fall back to symbol
                groupby_col = 'symbol'
        
        # Process historical data
        hist_data = self.historical_positions.copy()
        
        # Convert value to absolute value for allocation
        hist_data['abs_value'] = hist_data['value'].abs()
        
        # Group by date and allocation type
        grouped = hist_data.groupby(['timestamp', groupby_col])['abs_value'].sum().reset_index()
        
        # Calculate percentage within each date
        date_totals = grouped.groupby('timestamp')['abs_value'].sum().reset_index()
        date_totals.columns = ['timestamp', 'total_value']
        
        grouped = pd.merge(grouped, date_totals, on='timestamp')
        grouped['percentage'] = grouped['abs_value'] / grouped['total_value'] * 100
        
        # Get top N categories to avoid overcrowding
        top_categories = (
            grouped.groupby(groupby_col)['abs_value'].sum()
            .sort_values(ascending=False)
            .head(10)
            .index.tolist()
        )
        
        # Filter to top categories and add "Other"
        top_grouped = grouped[grouped[groupby_col].isin(top_categories)]
        
        # Calculate "Other" for each date
        other_grouped = (
            grouped[~grouped[groupby_col].isin(top_categories)]
            .groupby('timestamp')
            .agg({'abs_value': 'sum', 'total_value': 'first'})
            .reset_index()
        )
        other_grouped['percentage'] = (
            other_grouped['abs_value'] / other_grouped['total_value'] * 100
        )
        other_grouped[groupby_col] = 'Other'
        
        # Combine top categories with "Other"
        final_grouped = pd.concat([top_grouped, other_grouped], ignore_index=True)
        
        # Create a pivot for area chart
        pivot_data = final_grouped.pivot(
            index='timestamp', 
            columns=groupby_col, 
            values='percentage'
        ).fillna(0).reset_index()
        
        # Create area chart
        categories = [col for col in pivot_data.columns if col != 'timestamp']
        
        for i, category in enumerate(categories):
            color = self.config.allocation_colors[i % len(self.config.allocation_colors)]
            
            self.fig.add_trace(
                go.Scatter(
                    x=pivot_data['timestamp'],
                    y=pivot_data[category],
                    name=category,
                    mode='lines',
                    line=dict(width=0.5, color=color),
                    stackgroup='one',  # Stack the areas
                    hovertemplate='%{x}<br>%{y:.1f}%<extra>' + category + '</extra>'
                ),
                row=historical_row, col=1
            )
        
        # Update axes
        self.fig.update_xaxes(
            title_text="Date",
            row=historical_row, col=1
        )
        self.fig.update_yaxes(
            title_text="Allocation (%)",
            range=[0, 100],
            row=historical_row, col=1
        )
    
    def _add_risk_metrics(self) -> None:
        """Add risk metrics visualization."""
        risk_row = self.subplots.get("risk")
        if not risk_row:
            return
            
        # Get risk metrics to show
        risk_metrics = self.config.risk_metrics
        
        if not risk_metrics:
            self._add_empty_message(risk_row, "No risk metrics selected")
            return
            
        # Create a bar chart for risk metrics
        metric_names = []
        metric_values = []
        metric_colors = []
        hover_texts = []
        
        for metric in risk_metrics:
            if metric == RiskMetricType.VALUE_AT_RISK:
                if self.risk_metrics.value_at_risk is not None:
                    metric_names.append("Value at Risk (VaR)")
                    metric_values.append(self.risk_metrics.value_at_risk * 100)  # Convert to percentage
                    metric_colors.append(self.config.risk_bar_colors.get('var', 'red'))
                    hover_texts.append(
                        f"Value at Risk ({self.config.risk_confidence_level*100:.0f}% confidence):<br>"
                        f"{self.risk_metrics.value_at_risk:.2%}<br>"
                        f"Expected maximum loss over {self.config.risk_horizon_days} day(s)"
                    )
            elif metric == RiskMetricType.EXPECTED_SHORTFALL:
                if self.risk_metrics.expected_shortfall is not None:
                    metric_names.append("Expected Shortfall")
                    metric_values.append(self.risk_metrics.expected_shortfall * 100)  # Convert to percentage
                    metric_colors.append(self.config.risk_bar_colors.get('es', 'orange'))
                    hover_texts.append(
                        f"Expected Shortfall (CVaR):<br>"
                        f"{self.risk_metrics.expected_shortfall:.2%}<br>"
                        f"Average loss when exceeding VaR"
                    )
            elif metric == RiskMetricType.VOLATILITY:
                if self.risk_metrics.volatility is not None:
                    metric_names.append("Volatility")
                    metric_values.append(self.risk_metrics.volatility * 100)  # Convert to percentage
                    metric_colors.append(self.config.risk_bar_colors.get('vol', 'blue'))
                    hover_texts.append(
                        f"Portfolio Volatility:<br>"
                        f"{self.risk_metrics.volatility:.2%}<br>"
                        f"Standard deviation of returns"
                    )
            elif metric == RiskMetricType.BETA:
                if self.risk_metrics.beta is not None:
                    metric_names.append("Beta")
                    metric_values.append(self.risk_metrics.beta)  # Beta is already a ratio
                    metric_colors.append(self.config.risk_bar_colors.get('beta', 'purple'))
                    hover_texts.append(
                        f"Portfolio Beta:<br>"
                        f"{self.risk_metrics.beta:.2f}<br>"
                        f"Sensitivity to market movements"
                    )
            elif metric == RiskMetricType.TRACKING_ERROR:
                if self.risk_metrics.tracking_error is not None:
                    metric_names.append("Tracking Error")
                    metric_values.append(self.risk_metrics.tracking_error * 100)  # Convert to percentage
                    metric_colors.append(self.config.risk_bar_colors.get('te', 'teal'))
                    hover_texts.append(
                        f"Tracking Error:<br>"
                        f"{self.risk_metrics.tracking_error:.2%}<br>"
                        f"Volatility of active returns"
                    )
        
        if not metric_names:
            self._add_empty_message(risk_row, "No risk metrics available")
            return
            
        # Create bar chart
        self.fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=metric_colors,
                text=[f"{v:.2f}" for v in metric_values],
                textposition='auto',
                hovertext=hover_texts,
                hoverinfo="text"
            ),
            row=risk_row, col=1
        )
        
        # Add exposure metrics as a separate trace or annotation
        exposure_text = (
            f"Gross Exposure: {self.risk_metrics.gross_exposure:.2f}x<br>"
            f"Net Exposure: {self.risk_metrics.net_exposure:.2f}x<br>"
            f"Top Position: {self.risk_metrics.top_position_weight:.1%}"
        )
        
        # Add exposure text as annotation
        self.fig.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.95,
            text=exposure_text,
            showarrow=False,
            font=dict(size=10),
            align="right",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            row=risk_row, col=1
        )
    
    def _add_correlation_matrix(self) -> None:
        """Add correlation matrix visualization."""
        if self.returns_data is None or self.returns_data.empty:
            return
            
        correlation_row = self.subplots.get("correlation")
        if not correlation_row:
            return
            
        # Get current positions
        current_symbols = self.positions_data['symbol'].unique().tolist()
        
        # Filter returns data to current positions
        position_returns = self.returns_data[self.returns_data['symbol'].isin(current_symbols)]
        
        if position_returns.empty:
            self._add_empty_message(correlation_row, "No return data available for current positions")
            return
            
        # Get latest timestamp
        latest_timestamp = position_returns['timestamp'].max()
        
        # Filter to lookback period
        lookback_days = self.config.correlation_lookback
        lookback_date = latest_timestamp - pd.Timedelta(days=lookback_days)
        recent_returns = position_returns[position_returns['timestamp'] >= lookback_date]
        
        if recent_returns.empty:
            self._add_empty_message(correlation_row, "Insufficient return data for correlation analysis")
            return
            
        # Pivot returns to wide format
        returns_wide = recent_returns.pivot(
            index='timestamp', 
            columns='symbol', 
            values='returns'
        ).fillna(0)
        
        # Calculate correlation matrix
        correlation_matrix = returns_wide.corr(method=self.config.correlation_method)
        
        # Limit to top assets by value if matrix is too large
        if len(correlation_matrix) > 15:
            top_positions = (
                self.positions_data
                .sort_values('value', ascending=False)
                ['symbol']
                .head(15)
                .tolist()
            )
            correlation_matrix = correlation_matrix.loc[top_positions, top_positions]
        
        # Create heatmap
        heatmap = go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu_r',  # Red-Blue scale, reversed so blue=positive
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="Correlation",
                titleside="right"
            ),
            hovertemplate='%{y} & %{x}<br>Correlation: %{z:.2f}<extra></extra>'
        )
        
        self.fig.add_trace(
            heatmap,
            row=correlation_row, col=1
        )
        
        # Add text annotations (optional)
        for i, row in enumerate(correlation_matrix.index):
            for j, col in enumerate(correlation_matrix.columns):
                value = correlation_matrix.loc[row, col]
                text_color = "white" if abs(value) > 0.7 else "black"
                
                self.fig.add_annotation(
                    x=col,
                    y=row,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(color=text_color, size=9),
                    row=correlation_row, col=1
                )
    
    def _add_positions_table(self) -> None:
        """Add positions table visualization."""
        positions_row = self.subplots.get("positions")
        if not positions_row:
            return
            
        # Get positions data
        positions = self.positions_data.copy()
        
        if positions.empty:
            self._add_empty_message(positions_row, "No position data available")
            return
            
        # Sort by absolute value
        positions['abs_value'] = positions['value'].abs()
        positions = positions.sort_values('abs_value', ascending=False)
        
        # Limit number of rows
        if len(positions) > self.config.max_position_table_rows:
            positions = positions.head(self.config.max_position_table_rows)
        
        # Select and format columns
        display_columns = ['symbol', 'side', 'position_size', 'price', 'value']
        
        # Add sector if available and used
        if 'sector' in positions.columns and self.config.allocation_type == AllocationBreakdownType.SECTOR:
            display_columns.insert(2, 'sector')
            
        # Add custom grouping field if used
        if (self.config.allocation_type == AllocationBreakdownType.CUSTOM and 
            self.config.custom_grouping_field and 
            self.config.custom_grouping_field in positions.columns):
            display_columns.insert(2, self.config.custom_grouping_field)
        
        # Ensure all columns are in the dataframe
        display_columns = [col for col in display_columns if col in positions.columns]
        
        table_data = positions[display_columns].copy()
        
        # Format numeric columns
        if 'price' in table_data.columns:
            table_data['price'] = table_data['price'].apply(
                lambda x: f"{self.config.currency_symbol}{abs(x):,.2f}" if pd.notnull(x) else ""
            )
            
        if 'value' in table_data.columns:
            table_data['value'] = table_data['value'].apply(
                lambda x: f"{self.config.currency_symbol}{abs(x):,.2f}" if pd.notnull(x) else ""
            )
            
        if 'position_size' in table_data.columns:
            table_data['position_size'] = table_data['position_size'].apply(
                lambda x: f"{abs(x):,.2f}" if pd.notnull(x) else ""
            )
        
        # Create table
        table = go.Table(
            header=dict(
                values=[col.replace('_', ' ').title() for col in table_data.columns],
                align='left',
                fill_color='rgba(0, 0, 0, 0.1)',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[table_data[col] for col in table_data.columns],
                align='left',
                fill_color=[
                    np.where(
                        positions['side'] == 'LONG',
                        'rgba(200, 255, 200, 0.7)',  # Light green for long
                        'rgba(255, 200, 200, 0.7)'   # Light red for short
                    )
                ],
                font=dict(size=11, color='black'),
                height=25
            )
        )
        
        self.fig.add_trace(
            table,
            row=positions_row, col=1
        )
    
    def _add_empty_message(self, row: int, message: str) -> None:
        """
        Add an empty state message to a subplot.
        
        Args:
            row: Row index for the subplot
            message: Message to display
        """
        self.fig.add_annotation(
            xref="x domain", yref="y domain",
            x=0.5, y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=12, color="gray"),
            row=row, col=1
        )
    
    def _update_layout(self) -> None:
        """Update the figure layout with titles and formatting."""
        # Set chart title
        title = "Portfolio Analysis"
        
        # Base layout updates
        layout_updates = dict(
            title=title,
            height=self.config.chart_height,
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
        self.fig.update_xaxes(showgrid=self.config.show_grid)
        self.fig.update_yaxes(showgrid=self.config.show_grid)
        
        # Add a summary annotation in the top right
        total_value = self.positions_data['value'].abs().sum()
        total_long = self.positions_data[self.positions_data['side'] == 'LONG']['value'].sum()
        total_short = abs(self.positions_data[self.positions_data['side'] == 'SHORT']['value'].sum())
        
        summary_text = (
            f"Total Value: {self.config.currency_symbol}{total_value:,.2f}<br>"
            f"Long: {self.config.currency_symbol}{total_long:,.2f}<br>"
            f"Short: {self.config.currency_symbol}{total_short:,.2f}"
        )
        
        self.fig.add_annotation(
            xref="paper", yref="paper",
            x=0.99, y=0.99,
            text=summary_text,
            showarrow=False,
            font=dict(size=10),
            align="right",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )


def create_portfolio_view(
    positions_data: pd.DataFrame,
    returns_data: Optional[pd.DataFrame] = None,
    historical_positions: Optional[pd.DataFrame] = None,
    market_data: Optional[pd.DataFrame] = None,
    allocation_type: str = "asset",
    allocation_chart_type: str = "pie",
    show_risk_metrics: bool = True,
    show_correlation: bool = False,
    chart_height: int = 700,
    currency_symbol: str = "$"
) -> go.Figure:
    """
    Create an interactive portfolio visualization with allocation and risk metrics.
    
    This function provides a simplified interface to the PortfolioView class.
    
    Args:
        positions_data: DataFrame containing current positions
        returns_data: Optional DataFrame with historical returns
        historical_positions: Optional DataFrame with historical positions
        market_data: Optional DataFrame with market/benchmark data
        allocation_type: Type of allocation breakdown ("asset", "sector", etc.)
        allocation_chart_type: Chart type for allocation ("pie", "bar", "treemap")
        show_risk_metrics: Whether to show risk metrics visualization
        show_correlation: Whether to show correlation matrix
        chart_height: Height of the chart in pixels
        currency_symbol: Symbol to use for currency values
        
    Returns:
        Plotly Figure object
    """
    # Map allocation_type string to enum
    try:
        allocation_type_enum = AllocationBreakdownType(allocation_type)
    except ValueError:
        allocation_type_enum = AllocationBreakdownType.ASSET
    
    # Create configuration
    config = PortfolioViewConfig(
        chart_height=chart_height,
        allocation_type=allocation_type_enum,
        allocation_chart_type=allocation_chart_type,
        show_risk_metrics=show_risk_metrics,
        show_correlation=show_correlation,
        currency_symbol=currency_symbol
    )
    
    # Create view instance
    view = PortfolioView(
        positions_data=positions_data,
        returns_data=returns_data,
        historical_positions=historical_positions,
        market_data=market_data,
        config=config
    )
    
    # Create and return figure
    return view.create_figure()


def get_portfolio_risk_metrics(
    positions_data: pd.DataFrame,
    returns_data: Optional[pd.DataFrame] = None,
    market_data: Optional[pd.DataFrame] = None,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate portfolio risk metrics for the given positions.
    
    Args:
        positions_data: DataFrame containing current positions
        returns_data: Optional DataFrame with historical returns
        market_data: Optional DataFrame with market/benchmark data
        confidence_level: Confidence level for VaR calculation
        
    Returns:
        Dictionary of risk metrics
    """
    # Create a temporary view to calculate metrics
    config = PortfolioViewConfig(risk_confidence_level=confidence_level)
    
    view = PortfolioView(
        positions_data=positions_data,
        returns_data=returns_data,
        market_data=market_data,
        config=config
    )
    
    # Calculate metrics without creating a figure
    view._validate_data()
    view._calculate_risk_metrics()
    
    # Return metrics as a dictionary
    return view.risk_metrics.as_dict()