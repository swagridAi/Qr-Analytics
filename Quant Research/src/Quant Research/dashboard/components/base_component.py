"""
Base visualization component for the Quant Research dashboard.

This module provides a common base class for all visualization components
with shared utilities and standardized interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class BaseConfig:
    """Base configuration class for visualization components."""
    
    # Display settings
    chart_height: int = 600
    chart_width: Optional[int] = None
    theme: str = "white"  # "white" or "dark"
    
    # Layout settings
    show_grid: bool = True
    show_legend: bool = True
    margin: Dict[str, int] = field(default_factory=lambda: dict(l=50, r=50, t=70, b=50))
    
    # Color settings
    color_palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    
    # Text settings
    title_font_size: int = 16
    axis_font_size: int = 12
    legend_font_size: int = 10
    
    # Override in subclasses
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        pass


class BaseVisualization(ABC):
    """
    Abstract base class for creating visualization components.
    
    This class provides common functionality for all visualization
    components including figure creation, layout management, data 
    validation, and utility methods.
    """
    
    def __init__(self, config: Optional[BaseConfig] = None):
        """
        Initialize the visualization component.
        
        Args:
            config: Optional configuration object
        """
        self.fig = None
        self.subplots = {}  # Track subplot locations
        self.current_row = 0
        self.config = config or BaseConfig()
        self.error = None  # Track any errors during processing
        self.date_range = None  # Track overall date range for visualization
    
    @abstractmethod
    def _validate_data(self) -> None:
        """
        Validate input data has required columns and correct format.
        
        This method should be implemented by subclasses to validate
        their specific data requirements.
        
        Raises:
            ValueError: If required data is missing or invalid
        """
        pass
    
    @abstractmethod
    def _setup_subplots(self) -> None:
        """
        Set up the subplot structure for the visualization.
        
        This method should be implemented by subclasses to create
        the appropriate subplot layout.
        """
        pass
    
    @abstractmethod
    def create_figure(self) -> go.Figure:
        """
        Create the complete visualization with all components.
        
        Returns:
            Plotly Figure object with all visualization elements
        """
        pass
    
    def _update_layout(self, title: str) -> None:
        """
        Update the figure layout with common settings.
        
        Args:
            title: Title for the figure
        """
        if self.fig is None:
            return
        
        # Base layout updates
        layout_updates = dict(
            title=dict(
                text=title,
                font=dict(size=self.config.title_font_size)
            ),
            height=self.config.chart_height,
            margin=self.config.margin,
            hovermode='x unified'
        )
        
        # Add width if specified
        if self.config.chart_width:
            layout_updates["width"] = self.config.chart_width
        
        # Configure theme
        if self.config.theme.lower() == "dark":
            layout_updates.update(
                template="plotly_dark",
                paper_bgcolor="rgba(0, 0, 0, 0.8)",
                plot_bgcolor="rgba(0, 0, 0, 0.8)",
                font=dict(color="white")
            )
        else:
            layout_updates.update(
                template="plotly_white",
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="black")
            )
        
        # Configure legend
        if self.config.show_legend:
            layout_updates["legend"] = dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=self.config.legend_font_size)
            )
        else:
            layout_updates["showlegend"] = False
        
        # Apply layout updates
        self.fig.update_layout(**layout_updates)
        
        # Update grid settings for all subplots
        self.fig.update_xaxes(
            showgrid=self.config.show_grid,
            gridcolor="lightgray",
            tickfont=dict(size=self.config.axis_font_size)
        )
        self.fig.update_yaxes(
            showgrid=self.config.show_grid,
            gridcolor="lightgray",
            tickfont=dict(size=self.config.axis_font_size)
        )
        
        # Set consistent date range if we have one
        if self.date_range:
            for row in range(1, len(self.subplots) + 1):
                # Skip any non-time series subplots
                subplot_type = next((k for k, v in self.subplots.items() if v == row), None)
                if subplot_type not in ["histogram", "correlation", "heatmap"]:
                    self.fig.update_xaxes(range=self.date_range, row=row, col=1)
    
    def _create_empty_figure(self, message: str = "No data available") -> go.Figure:
        """
        Create an empty figure with a message.
        
        Args:
            message: Message to display in the figure
            
        Returns:
            Empty Plotly Figure with message
        """
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        
        # Apply basic styling
        empty_fig.update_layout(
            height=self.config.chart_height,
            width=self.config.chart_width,
            template="plotly_white" if self.config.theme.lower() != "dark" else "plotly_dark"
        )
        
        return empty_fig
    
    def _add_empty_message(self, row: int, message: str) -> None:
        """
        Add an empty state message to a subplot.
        
        Args:
            row: Row index for the subplot
            message: Message to display
        """
        if self.fig is None:
            return
            
        self.fig.add_annotation(
            xref="x domain", yref="y domain",
            x=0.5, y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=12, color="gray"),
            align="center",
            bgcolor="rgba(255, 255, 255, 0.7)" if self.config.theme.lower() != "dark" else "rgba(0, 0, 0, 0.7)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            row=row, col=1
        )
    
    def validate_required_columns(
        self, 
        df: pd.DataFrame, 
        required_columns: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Check if DataFrame contains all required columns.
        
        Args:
            df: DataFrame to check
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        if df is None or df.empty:
            return False, required_columns
            
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        return len(missing_columns) == 0, missing_columns
    
    def ensure_datetime_column(self, df: pd.DataFrame, column: str = 'timestamp') -> pd.DataFrame:
        """
        Ensure a column is in datetime format.
        
        Args:
            df: DataFrame to process
            column: Column name to convert
            
        Returns:
            DataFrame with column converted to datetime
        """
        if df is None or df.empty or column not in df.columns:
            return df
            
        result = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = pd.to_datetime(result[column], errors='coerce')
            
        return result
    
    def filter_by_symbols(
        self, 
        df: pd.DataFrame, 
        symbols: List[str], 
        symbol_column: str = 'symbol'
    ) -> pd.DataFrame:
        """
        Filter a DataFrame by list of symbols.
        
        Args:
            df: DataFrame to filter
            symbols: List of symbols to include
            symbol_column: Name of symbol column
            
        Returns:
            Filtered DataFrame
        """
        if df is None or df.empty or symbol_column not in df.columns or not symbols:
            return df
            
        return df[df[symbol_column].isin(symbols)]
    
    def filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        date_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Filter a DataFrame by date range.
        
        Args:
            df: DataFrame to filter
            start_date: Start date for filtering
            end_date: End date for filtering
            date_column: Name of date column
            
        Returns:
            Filtered DataFrame
        """
        if df is None or df.empty or date_column not in df.columns:
            return df
            
        result = df.copy()
        
        # Ensure date column is datetime
        result = self.ensure_datetime_column(result, date_column)
        
        # Apply filters
        if start_date is not None:
            result = result[result[date_column] >= start_date]
            
        if end_date is not None:
            result = result[result[date_column] <= end_date]
            
        return result
    
    def calculate_moving_average(
        self,
        series: pd.Series,
        window: int,
        method: str = 'sma'
    ) -> pd.Series:
        """
        Calculate moving average of a series.
        
        Args:
            series: Data series
            window: Window size for moving average
            method: Method ('sma', 'ema', 'wma')
            
        Returns:
            Series with moving average values
        """
        if series is None or series.empty or window <= 0:
            return pd.Series(index=series.index)
            
        if method.lower() == 'ema':
            return series.ewm(span=window, adjust=False).mean()
        elif method.lower() == 'wma':
            # Weighted moving average with linearly decreasing weights
            weights = np.arange(1, window + 1)
            return series.rolling(window).apply(
                lambda x: np.sum(weights * x) / np.sum(weights),
                raw=True
            )
        else:  # Default to SMA
            return series.rolling(window=window).mean()
    
    def calculate_bollinger_bands(
        self,
        series: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands for a series.
        
        Args:
            series: Data series
            window: Window size for moving average
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (middle_band, upper_band, lower_band)
        """
        if series is None or series.empty or window <= 0:
            return (
                pd.Series(index=series.index),
                pd.Series(index=series.index),
                pd.Series(index=series.index)
            )
            
        middle_band = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return middle_band, upper_band, lower_band
    
    def calculate_rsi(
        self,
        series: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            series: Data series
            window: RSI period
            
        Returns:
            Series with RSI values
        """
        if series is None or series.empty or window <= 0:
            return pd.Series(index=series.index)
            
        # Calculate price changes
        delta = series.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=window).mean()
        avg_loss = losses.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_drawdowns(
        self,
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
            return pd.Series(), 0.0, 0
            
        # Calculate rolling maximum
        rolling_max = equity_series.cummax()
        
        # Calculate drawdown series (as percentage)
        drawdown_series = (equity_series - rolling_max) / rolling_max
        
        # Find maximum drawdown
        max_drawdown = drawdown_series.min()
        
        # Calculate maximum drawdown duration
        max_drawdown_duration = 0
        
        if max_drawdown < 0:
            # Find the index of max drawdown
            max_dd_idx = drawdown_series.idxmin()
            
            # Find the last peak before the max drawdown
            prev_peak_idx = rolling_max.loc[:max_dd_idx].idxmax()
            
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
    
    def add_reference_line(
        self,
        value: float,
        axis: str = 'y',
        line_dash: str = 'dash',
        line_color: str = 'gray',
        line_width: float = 1.0,
        row: int = 1,
        col: int = 1
    ) -> None:
        """
        Add a reference line to a subplot.
        
        Args:
            value: Line value
            axis: Axis for line ('x' or 'y')
            line_dash: Line dash style
            line_color: Line color
            line_width: Line width
            row: Row of subplot
            col: Column of subplot
        """
        if self.fig is None:
            return
            
        line_props = dict(
            line_dash=line_dash,
            line_color=line_color,
            line_width=line_width,
            row=row,
            col=col
        )
        
        if axis.lower() == 'x':
            self.fig.add_vline(x=value, **line_props)
        else:
            self.fig.add_hline(y=value, **line_props)
    
    def add_axis_title(
        self,
        title: str,
        axis: str = 'x',
        row: int = 1,
        col: int = 1
    ) -> None:
        """
        Add title to an axis in a subplot.
        
        Args:
            title: Title text
            axis: Axis to update ('x' or 'y')
            row: Row of subplot
            col: Column of subplot
        """
        if self.fig is None:
            return
            
        axis_props = dict(
            title_text=title,
            title_font=dict(size=self.config.axis_font_size),
            row=row,
            col=col
        )
        
        if axis.lower() == 'x':
            self.fig.update_xaxes(**axis_props)
        else:
            self.fig.update_yaxes(**axis_props)
    
    def add_annotation(
        self,
        text: str,
        x: Union[float, str] = 0.5,
        y: Union[float, str] = 0.5,
        xref: str = "paper",
        yref: str = "paper",
        showarrow: bool = False,
        bgcolor: Optional[str] = "rgba(255, 255, 255, 0.7)",
        row: Optional[int] = None,
        col: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Add an annotation to the figure.
        
        Args:
            text: Annotation text
            x: X position
            y: Y position
            xref: X reference ('paper' or 'x')
            yref: Y reference ('paper' or 'y')
            showarrow: Whether to show arrow
            bgcolor: Background color (None for transparent)
            row: Row of subplot (None for global annotation)
            col: Column of subplot (None for global annotation)
            **kwargs: Additional annotation properties
        """
        if self.fig is None:
            return
            
        # Adjust background color based on theme
        if bgcolor == "rgba(255, 255, 255, 0.7)" and self.config.theme.lower() == "dark":
            bgcolor = "rgba(0, 0, 0, 0.7)"
            
        annotation = dict(
            text=text,
            x=x,
            y=y,
            xref=xref,
            yref=yref,
            showarrow=showarrow,
            font=dict(size=10),
            align="center",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            **kwargs
        )
        
        if bgcolor:
            annotation["bgcolor"] = bgcolor
        
        if row is not None and col is not None:
            annotation["row"] = row
            annotation["col"] = col
        
        self.fig.add_annotation(**annotation)
    
    def get_date_range(self, df: pd.DataFrame, date_col: str = 'timestamp') -> Optional[List[pd.Timestamp]]:
        """
        Get the date range from a DataFrame.
        
        Args:
            df: DataFrame containing date column
            date_col: Date column name
            
        Returns:
            List of [min_date, max_date] or None if no dates
        """
        if df is None or df.empty or date_col not in df.columns:
            return None
            
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = self.ensure_datetime_column(df, date_col)
            
        return [df[date_col].min(), df[date_col].max()]
    
    def update_date_range(self, new_range: List[pd.Timestamp]) -> None:
        """
        Update the global date range, expanding if necessary.
        
        Args:
            new_range: New date range [min_date, max_date]
        """
        if not new_range or len(new_range) != 2:
            return
            
        if self.date_range is None:
            self.date_range = new_range
        else:
            self.date_range[0] = min(self.date_range[0], new_range[0])
            self.date_range[1] = max(self.date_range[1], new_range[1])
    
    def get_color_for_index(self, index: int) -> str:
        """
        Get a color from the palette based on index.
        
        Args:
            index: Index in the color palette
            
        Returns:
            Color string
        """
        palette = self.config.color_palette
        return palette[index % len(palette)]
    
    def format_currency(self, value: float, symbol: str = "$", decimals: int = 2) -> str:
        """
        Format a value as currency.
        
        Args:
            value: Value to format
            symbol: Currency symbol
            decimals: Number of decimal places
            
        Returns:
            Formatted currency string
        """
        if value is None or np.isnan(value):
            return "N/A"
            
        return f"{symbol}{abs(value):,.{decimals}f}"
    
    def format_percentage(self, value: float, decimals: int = 2) -> str:
        """
        Format a value as percentage.
        
        Args:
            value: Value to format (0.01 = 1%)
            decimals: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        if value is None or np.isnan(value):
            return "N/A"
            
        return f"{value * 100:.{decimals}f}%"
    
    def create_subplots(
        self,
        subplot_specs: Dict[str, int],
        row_heights: Optional[List[float]] = None,
        subplot_titles: Optional[List[str]] = None
    ) -> None:
        """
        Create subplot structure based on specifications.
        
        Args:
            subplot_specs: Dictionary mapping subplot names to row indices
            row_heights: Optional list of row heights (normalized)
            subplot_titles: Optional list of subplot titles
        """
        num_rows = max(subplot_specs.values())
        self.subplots = subplot_specs
        
        # Create default row heights if not provided
        if not row_heights:
            if num_rows == 1:
                row_heights = [1]
            elif num_rows == 2:
                row_heights = [0.7, 0.3]
            elif num_rows == 3:
                row_heights = [0.6, 0.2, 0.2]
            else:
                main_height = 0.5
                remaining = (1 - main_height) / (num_rows - 1)
                row_heights = [main_height] + [remaining] * (num_rows - 1)
        
        # Create subplot titles if not provided
        if not subplot_titles:
            subplot_titles = [""] * num_rows
        
        # Create figure with subplots
        self.fig = make_subplots(
            rows=num_rows,
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )
    
    def handle_error(self, error: Exception, message: str = "Error creating visualization") -> go.Figure:
        """
        Handle an error by creating an error figure.
        
        Args:
            error: Exception that occurred
            message: Base error message
            
        Returns:
            Error figure with message
        """
        self.error = error
        error_message = f"{message}: {str(error)}"
        return self._create_empty_figure(error_message)
    
    def create_color_scale(
        self,
        min_value: float,
        max_value: float,
        color_scale: Optional[List[Tuple[float, str]]] = None
    ) -> Callable[[float], str]:
        """
        Create a function that maps values to colors in a scale.
        
        Args:
            min_value: Minimum value in range
            max_value: Maximum value in range
            color_scale: Optional custom color scale as [(pos, color), ...]
            
        Returns:
            Function that takes a value and returns a color
        """
        if color_scale is None:
            # Default to red-white-green scale
            color_scale = [
                (0.0, "rgb(255, 0, 0)"),      # Red
                (0.5, "rgb(255, 255, 255)"),  # White
                (1.0, "rgb(0, 255, 0)")       # Green
            ]
        
        def get_color(value: float) -> str:
            """Map a value to a color based on the color scale."""
            if value is None or np.isnan(value):
                return "rgb(200, 200, 200)"  # Gray for missing values
                
            # Normalize value to [0, 1]
            if min_value == max_value:
                norm_value = 0.5
            else:
                norm_value = (value - min_value) / (max_value - min_value)
                norm_value = max(0, min(1, norm_value))  # Clamp to [0, 1]
            
            # Find the right color segment
            for i in range(len(color_scale) - 1):
                pos1, color1 = color_scale[i]
                pos2, color2 = color_scale[i + 1]
                
                if pos1 <= norm_value <= pos2:
                    # Interpolate between colors
                    ratio = (norm_value - pos1) / (pos2 - pos1)
                    
                    # Parse rgb colors
                    r1, g1, b1 = parse_rgb(color1)
                    r2, g2, b2 = parse_rgb(color2)
                    
                    # Interpolate RGB components
                    r = int(r1 + ratio * (r2 - r1))
                    g = int(g1 + ratio * (g2 - g1))
                    b = int(b1 + ratio * (b2 - b1))
                    
                    return f"rgb({r}, {g}, {b})"
            
            # Fallback - should not reach here
            return color_scale[-1][1]
        
        return get_color

def parse_rgb(color_str: str) -> Tuple[int, int, int]:
    """
    Parse an RGB color string into components.
    
    Args:
        color_str: Color string in format "rgb(r, g, b)" or hex "#rrggbb"
        
    Returns:
        Tuple of (r, g, b) values
    """
    if color_str.startswith("rgb"):
        # Parse "rgb(r, g, b)"
        rgb_match = color_str.replace(" ", "").strip("rgb()").split(",")
        return tuple(int(x) for x in rgb_match)
    elif color_str.startswith("#"):
        # Parse hex "#rrggbb"
        hex_str = color_str.lstrip("#")
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    else:
        # Default fallback
        return (128, 128, 128)