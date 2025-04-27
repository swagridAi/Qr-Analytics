"""
Price chart component for the Quant Research dashboard.

This module provides a comprehensive price chart visualization component
with customizable indicators, overlays, and multi-asset support.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quant_research.dashboard.base_component import BaseVisualization, BaseConfig
from quant_research.dashboard.chart_config import ChartTheme, ColorPalettes
from quant_research.dashboard.chart_utils import (
    validate_dataframe,
    filter_by_date_range,
    filter_by_symbols,
    calculate_moving_average,
    calculate_bollinger_bands,
    calculate_rsi,
    add_axis_title,
    add_reference_line,
    create_empty_figure
)


class ChartType(str, Enum):
    """Chart type options."""
    CANDLESTICK = "candlestick"
    OHLC = "ohlc"
    LINE = "line"


@dataclass
class PriceChartConfig(BaseConfig):
    """Configuration settings for price chart display."""
    
    # Chart type
    chart_type: ChartType = ChartType.CANDLESTICK
    
    # Display settings
    show_volume: bool = True
    show_signals: bool = True
    show_regimes: bool = True
    
    # Indicator settings
    show_ma: bool = False
    ma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    ma_types: List[str] = field(default_factory=lambda: ["sma", "sma", "sma"])
    show_bollinger: bool = False
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    show_rsi: bool = False
    rsi_period: int = 14
    
    # Color settings
    candlestick_colors: Dict[str, str] = field(default_factory=lambda: {
        "increasing": ColorPalettes.PRICE_COLORS["increasing"],
        "decreasing": ColorPalettes.PRICE_COLORS["decreasing"]
    })
    volume_colors: Dict[str, str] = field(default_factory=lambda: {
        "increasing": ColorPalettes.PRICE_COLORS["volume_up"],
        "decreasing": ColorPalettes.PRICE_COLORS["volume_down"]
    })
    ma_colors: List[str] = field(default_factory=lambda: ColorPalettes.ASSET_COLORS[:3])
    signal_colors: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "BUY": ColorPalettes.SIGNAL_MARKERS["buy"],
        "SELL": ColorPalettes.SIGNAL_MARKERS["sell"],
        "ENTRY": ColorPalettes.SIGNAL_MARKERS["entry"],
        "EXIT": ColorPalettes.SIGNAL_MARKERS["exit"]
    })
    
    # Layout settings
    show_grid: bool = True
    show_legend: bool = True
    show_rangeslider: bool = False
    show_crosshair: bool = True
    
    # Advanced settings
    display_timezone: Optional[str] = None
    custom_indicators: List[Dict[str, Any]] = field(default_factory=list)
    hover_info: str = "all"  # "all", "ohlcv", "minimal"
    log_scale: bool = False  # Use log scale for price axis
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Ensure ma_periods and ma_types have matching lengths
        if len(self.ma_periods) > len(self.ma_types):
            # Extend ma_types with defaults
            self.ma_types.extend(["sma"] * (len(self.ma_periods) - len(self.ma_types)))
        elif len(self.ma_types) > len(self.ma_periods):
            # Truncate ma_types to match ma_periods
            self.ma_types = self.ma_types[:len(self.ma_periods)]


class PriceChart(BaseVisualization):
    """
    Class for creating and managing price charts with various overlays.
    
    This class handles data validation, figure creation, and adding
    different types of indicators and overlays to the chart.
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        signals_data: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.DataFrame] = None,
        selected_symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        config: Optional[PriceChartConfig] = None
    ):
        """
        Initialize the price chart with data and configuration.
        
        Args:
            price_data: DataFrame containing OHLCV data with columns:
                [timestamp, open, high, low, close, volume(optional), symbol(optional)]
            signals_data: Optional DataFrame with columns:
                [timestamp, signal_type, price(optional), symbol(optional)]
            regime_data: Optional DataFrame with columns:
                [timestamp, regime_*, symbol(optional)]
            selected_symbols: List of symbols to display
            timeframe: Timeframe of the chart (e.g., '1h', '1d')
            config: Optional configuration object
        
        Raises:
            ValueError: If required data columns are missing
        """
        super().__init__()
        
        self.price_data = price_data.copy() if price_data is not None else pd.DataFrame()
        self.signals_data = signals_data.copy() if signals_data is not None else None
        self.regime_data = regime_data.copy() if regime_data is not None else None
        self.selected_symbols = selected_symbols
        self.timeframe = timeframe
        self.config = config or PriceChartConfig()
        
        # Initialize chart state
        self.date_range = None
        
        # Map custom indicators by name
        self.custom_indicator_funcs = {
            "bollinger": self._add_bollinger_bands,
            "rsi": self._add_rsi,
            "volume_profile": self._add_volume_profile
        }
        
        # Validate input data
        self._validate_data()
        
    def _validate_data(self) -> None:
        """
        Validate the input data has required columns.
        
        Raises:
            ValueError: If required data columns are missing
        """
        if self.price_data.empty:
            return
            
        # Define required columns based on configuration
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        if self.config.show_volume:
            required_columns.append('volume')
        
        # Validate price data
        is_valid, missing_columns = validate_dataframe(self.price_data, required_columns)
        if not is_valid:
            raise ValueError(f"Missing required columns in price data: {missing_columns}")
        
        # Filter for selected symbols if provided
        if self.selected_symbols and 'symbol' in self.price_data.columns:
            self.price_data = filter_by_symbols(self.price_data, self.selected_symbols)
            
            # Ensure we have data after filtering
            if self.price_data.empty:
                raise ValueError(f"No data found for selected symbols: {self.selected_symbols}")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.price_data['timestamp']):
            self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
        
        # Validate signals data if provided
        if self.signals_data is not None:
            signals_required = ['timestamp', 'signal_type']
            is_valid, missing_columns = validate_dataframe(self.signals_data, signals_required)
            if not is_valid:
                self.signals_data = None
            elif self.selected_symbols and 'symbol' in self.signals_data.columns:
                self.signals_data = filter_by_symbols(self.signals_data, self.selected_symbols)
        
        # Validate regime data if provided
        if self.regime_data is not None:
            # Check if we have regime columns
            regime_cols = [col for col in self.regime_data.columns if col.startswith('regime_')]
            if not regime_cols or 'timestamp' not in self.regime_data.columns:
                self.regime_data = None
            elif self.selected_symbols and 'symbol' in self.regime_data.columns:
                self.regime_data = filter_by_symbols(self.regime_data, self.selected_symbols)
    
    def _setup_subplots(self) -> None:
        """Set up the subplot structure based on configuration."""
        # Count how many subplot rows we need
        rows = 1  # Main price chart
        
        # Add additional rows based on config
        if self.config.show_volume:
            rows += 1
            self.subplots["volume"] = rows
        
        if self.config.show_rsi:
            rows += 1
            self.subplots["rsi"] = rows
        
        # Calculate row heights - main chart gets more space
        if rows == 1:
            row_heights = [1]
        elif rows == 2:
            row_heights = [0.8, 0.2]
        elif rows == 3:
            row_heights = [0.6, 0.2, 0.2]
        else:
            row_heights = [0.5] + [0.5 / (rows - 1)] * (rows - 1)
        
        # Create subplot figure
        self.fig = make_subplots(
            rows=rows, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights
        )
    
    def create_figure(self) -> go.Figure:
        """
        Create the complete price chart figure with all components.
        
        Returns:
            Plotly Figure object with all requested chart elements
        
        Raises:
            RuntimeError: If figure creation fails
        """
        if self.price_data.empty:
            # Return empty figure with message
            return create_empty_figure("No price data available for the selected parameters")
        
        try:
            # Determine subplot structure
            self.subplots = {"price": 1}  # Price is always in row 1
            self._setup_subplots()
            
            # Add price data
            self._add_price_data()
            
            # Add volume if requested
            if self.config.show_volume:
                self._add_volume_data()
            
            # Add moving averages if requested
            if self.config.show_ma:
                self._add_moving_averages()
            
            # Add Bollinger Bands if requested
            if self.config.show_bollinger:
                self._add_bollinger_bands()
            
            # Add RSI if requested
            if self.config.show_rsi:
                self._add_rsi()
            
            # Add signals if available and requested
            if self.signals_data is not None and not self.signals_data.empty and self.config.show_signals:
                self._add_signals()
            
            # Add regime overlays if available and requested
            if self.regime_data is not None and not self.regime_data.empty and self.config.show_regimes:
                self._add_regimes()
            
            # Add any custom indicators
            for indicator in self.config.custom_indicators:
                indicator_type = indicator.get('type')
                if indicator_type and indicator_type in self.custom_indicator_funcs:
                    # Call the indicator function with params
                    params = indicator.get('params', {})
                    self.custom_indicator_funcs[indicator_type](**params)
            
            # Update layout
            title = f"Price Chart ({', '.join(self.selected_symbols) if self.selected_symbols else 'All Symbols'}) - {self.timeframe}"
            self._update_layout(title)
            
            # Apply specific layout settings for price chart
            self._apply_price_chart_layout()
            
            return self.fig
            
        except Exception as e:
            # Return error figure
            error_fig = create_empty_figure(f"Error creating chart: {str(e)}")
            return error_fig
    
    def _apply_price_chart_layout(self) -> None:
        """Apply price chart specific layout settings."""
        if self.fig is None:
            return
            
        # Set rangeslider visibility
        self.fig.update_layout(
            xaxis_rangeslider_visible=self.config.show_rangeslider,
            hovermode='x unified' if self.config.show_crosshair else 'closest'
        )
        
        # Set y-axis type for price
        yaxis_type = "log" if self.config.log_scale else "linear"
        self.fig.update_yaxes(
            type=yaxis_type,
            row=1, col=1
        )
        
        # Update axis titles
        add_axis_title(self.fig, "Date", "x", row=1, col=1)
        add_axis_title(self.fig, "Price", "y", row=1, col=1)
        
        if self.config.show_volume and "volume" in self.subplots:
            volume_row = self.subplots["volume"]
            add_axis_title(self.fig, "Volume", "y", row=volume_row, col=1)
        
        if self.config.show_rsi and "rsi" in self.subplots:
            rsi_row = self.subplots["rsi"]
            add_axis_title(self.fig, "RSI", "y", row=rsi_row, col=1)
            self.fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)
        
        # Set consistent date range if we have one
        if self.date_range:
            self.fig.update_xaxes(range=self.date_range)
    
    def _add_price_data(self) -> None:
        """Add price data traces to the figure."""
        # If we have multiple symbols, create separate traces for each
        if 'symbol' in self.price_data.columns and len(self.price_data['symbol'].unique()) > 1:
            for symbol in self.selected_symbols or self.price_data['symbol'].unique():
                symbol_data = self.price_data[self.price_data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                    
                self._add_price_trace(symbol_data, symbol)
                
                # Update date range
                symbol_range = self.format_date_range(symbol_data)
                if symbol_range:
                    if self.date_range is None:
                        self.date_range = symbol_range
                    else:
                        self.date_range[0] = min(self.date_range[0], symbol_range[0])
                        self.date_range[1] = max(self.date_range[1], symbol_range[1])
        else:
            # Single symbol or no symbol column
            self._add_price_trace(self.price_data)
            
            # Set date range
            self.date_range = self.format_date_range(self.price_data)

    def _add_price_trace(self, data: pd.DataFrame, symbol: Optional[str] = None) -> None:
        """
        Add a price data trace to the figure based on chart type.
        
        Args:
            data: DataFrame containing OHLCV data
            symbol: Optional symbol name for the trace
        """
        if self.fig is None:
            return
            
        trace_name = symbol or "Price"
        
        if self.config.chart_type == ChartType.CANDLESTICK:
            self.fig.add_trace(
                go.Candlestick(
                    x=data['timestamp'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name=trace_name,
                    increasing_line_color=self.config.candlestick_colors["increasing"],
                    decreasing_line_color=self.config.candlestick_colors["decreasing"],
                    showlegend=True
                ),
                row=1, col=1
            )
        elif self.config.chart_type == ChartType.OHLC:
            self.fig.add_trace(
                go.Ohlc(
                    x=data['timestamp'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name=trace_name,
                    increasing_line_color=self.config.candlestick_colors["increasing"],
                    decreasing_line_color=self.config.candlestick_colors["decreasing"],
                    showlegend=True
                ),
                row=1, col=1
            )
        elif self.config.chart_type == ChartType.LINE:
            self.fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['close'],
                    name=trace_name,
                    mode='lines',
                    line=dict(width=2),
                    showlegend=True
                ),
                row=1, col=1
            )

    def _add_volume_data(self) -> None:
        """Add volume data to the figure."""
        if self.fig is None or 'volume' not in self.price_data.columns:
            return
            
        volume_row = self.subplots.get("volume", 2)
        
        # If we have multiple symbols, create separate volume traces for each
        if 'symbol' in self.price_data.columns and len(self.price_data['symbol'].unique()) > 1:
            for symbol in self.selected_symbols or self.price_data['symbol'].unique():
                symbol_data = self.price_data[self.price_data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                
                self._add_volume_trace(symbol_data, volume_row, symbol)
        else:
            # Single symbol or no symbol column
            self._add_volume_trace(self.price_data, volume_row)

    def _add_volume_trace(self, data: pd.DataFrame, row: int, symbol: Optional[str] = None) -> None:
        """
        Add a volume trace to the figure.
        
        Args:
            data: DataFrame containing volume data
            row: Row index for the subplot
            symbol: Optional symbol name for the trace
        """
        if self.fig is None or 'volume' not in data.columns:
            return
            
        trace_name = f"{symbol + ' ' if symbol else ''}Volume"
        
        # Calculate colors based on price movement if available
        if all(col in data.columns for col in ['open', 'close']):
            colors = np.where(
                data['close'] >= data['open'],
                self.config.volume_colors["increasing"],
                self.config.volume_colors["decreasing"]
            )
        else:
            # Use a single color if we can't determine direction
            colors = self.config.volume_colors["increasing"]
        
        self.fig.add_trace(
            go.Bar(
                x=data['timestamp'],
                y=data['volume'],
                name=trace_name,
                marker_color=colors,
                showlegend=True
            ),
            row=row, col=1
        )

    def _add_moving_averages(self) -> None:
        """Add moving average traces to the figure."""
        if self.fig is None or not self.config.ma_periods:
            return
            
        # If we have multiple symbols, create MA traces for each
        if 'symbol' in self.price_data.columns and len(self.price_data['symbol'].unique()) > 1:
            for symbol in self.selected_symbols or self.price_data['symbol'].unique():
                symbol_data = self.price_data[self.price_data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                
                for i, period in enumerate(self.config.ma_periods):
                    ma_type = self.config.ma_types[i] if i < len(self.config.ma_types) else "sma"
                    color = self.config.ma_colors[i % len(self.config.ma_colors)]
                    self._add_ma_trace(symbol_data, period, ma_type, color, symbol)
        else:
            # Single symbol or no symbol column
            for i, period in enumerate(self.config.ma_periods):
                ma_type = self.config.ma_types[i] if i < len(self.config.ma_types) else "sma"
                color = self.config.ma_colors[i % len(self.config.ma_colors)]
                self._add_ma_trace(self.price_data, period, ma_type, color)

    def _add_ma_trace(self, data: pd.DataFrame, period: int, ma_type: str, 
                     color: str, symbol: Optional[str] = None) -> None:
        """
        Add a moving average trace to the figure.
        
        Args:
            data: DataFrame containing price data
            period: MA period length
            ma_type: Type of moving average ('sma', 'ema', 'wma')
            color: Line color
            symbol: Optional symbol name for the trace
        """
        if self.fig is None or 'close' not in data.columns:
            return
            
        # Calculate the moving average
        ma_values = calculate_moving_average(data['close'], period, ma_type)
        
        # Format the name based on MA type
        if ma_type.lower() == 'ema':
            ma_name = f"EMA({period})"
        elif ma_type.lower() == 'wma':
            ma_name = f"WMA({period})"
        else:
            ma_name = f"SMA({period})"
        
        trace_name = f"{symbol + ' ' if symbol else ''}{ma_name}"
        
        self.fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=ma_values,
                name=trace_name,
                line=dict(width=1, color=color),
                showlegend=True
            ),
            row=1, col=1
        )

    def _add_bollinger_bands(self, period: Optional[int] = None, 
                            std: Optional[float] = None) -> None:
        """
        Add Bollinger Bands to the figure.
        
        Args:
            period: Optional period override
            std: Optional standard deviation override
        """
        if self.fig is None:
            return
            
        period = period or self.config.bollinger_period
        std = std or self.config.bollinger_std
        
        # If we have multiple symbols, create BB traces for each
        if 'symbol' in self.price_data.columns and len(self.price_data['symbol'].unique()) > 1:
            for symbol in self.selected_symbols or self.price_data['symbol'].unique():
                symbol_data = self.price_data[self.price_data['symbol'] == symbol]
                
                if symbol_data.empty or 'close' not in symbol_data.columns:
                    continue
                
                self._add_bollinger_trace(symbol_data, period, std, symbol)
        else:
            # Single symbol or no symbol column
            if 'close' in self.price_data.columns:
                self._add_bollinger_trace(self.price_data, period, std)

    def _add_bollinger_trace(self, data: pd.DataFrame, period: int, 
                            std: float, symbol: Optional[str] = None) -> None:
        """
        Add Bollinger Band traces for a single symbol.
        
        Args:
            data: DataFrame containing price data
            period: Bollinger Band period length
            std: Number of standard deviations
            symbol: Optional symbol name for the trace
        """
        if self.fig is None or 'close' not in data.columns:
            return
            
        prefix = f"{symbol + ' ' if symbol else ''}"
        
        # Calculate Bollinger Bands
        middle_band, upper_band, lower_band = calculate_bollinger_bands(
            data['close'], period, std
        )
        
        # Add the middle band (SMA)
        self.fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=middle_band,
                name=f"{prefix}BB Middle ({period})",
                line=dict(width=1, color='rgba(100, 100, 100, 0.7)'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add the upper band
        self.fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=upper_band,
                name=f"{prefix}BB Upper ({period}, {std}σ)",
                line=dict(width=1, color='rgba(100, 100, 100, 0.5)'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add the lower band
        self.fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=lower_band,
                name=f"{prefix}BB Lower ({period}, {std}σ)",
                line=dict(width=1, color='rgba(100, 100, 100, 0.5)'),
                fill='tonexty',  # Fill between upper and lower bands
                showlegend=True
            ),
            row=1, col=1
        )

    def _add_rsi(self, period: Optional[int] = None) -> None:
        """
        Add Relative Strength Index indicator.
        
        Args:
            period: Optional period override
        """
        if self.fig is None:
            return
            
        period = period or self.config.rsi_period
        rsi_row = self.subplots.get("rsi", 3)
        
        # If we have multiple symbols, create RSI traces for each
        if 'symbol' in self.price_data.columns and len(self.price_data['symbol'].unique()) > 1:
            for symbol in self.selected_symbols or self.price_data['symbol'].unique():
                symbol_data = self.price_data[self.price_data['symbol'] == symbol]
                
                if symbol_data.empty or 'close' not in symbol_data.columns:
                    continue
                
                self._add_rsi_trace(symbol_data, period, rsi_row, symbol)
        else:
            # Single symbol or no symbol column
            if 'close' in self.price_data.columns:
                self._add_rsi_trace(self.price_data, period, rsi_row)

    def _add_rsi_trace(self, data: pd.DataFrame, period: int, 
                      row: int, symbol: Optional[str] = None) -> None:
        """
        Add RSI trace for a single symbol.
        
        Args:
            data: DataFrame containing price data
            period: RSI period length
            row: Row index for the subplot
            symbol: Optional symbol name for the trace
        """
        if self.fig is None or 'close' not in data.columns:
            return
            
        # Calculate RSI
        rsi = calculate_rsi(data['close'], period)
        
        trace_name = f"{symbol + ' ' if symbol else ''}RSI ({period})"
        
        # Add RSI trace
        self.fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=rsi,
                name=trace_name,
                line=dict(width=1, color='purple'),
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Add overbought/oversold lines
        add_reference_line(self.fig, 70, "y", "dash", "red", 1, row, 1)
        add_reference_line(self.fig, 30, "y", "dash", "green", 1, row, 1)
        add_reference_line(self.fig, 50, "y", "dash", "gray", 0.5, row, 1)

    def _add_volume_profile(self, bins: int = 20) -> None:
        """
        Add volume profile to the right side of the chart.
        
        Args:
            bins: Number of price bins
        """
        # Not implementing detailed volume profile in this sample
        pass

    def _add_signals(self) -> None:
        """Add signal markers to the figure."""
        if self.fig is None or self.signals_data is None or self.signals_data.empty:
            return
            
        signal_type_col = next((col for col in ['signal_type', 'type'] 
                              if col in self.signals_data.columns), None)
        
        if signal_type_col is None:
            return
            
        # Filter signals for selected symbols if needed
        signals = self.signals_data
        if self.selected_symbols and 'symbol' in signals.columns:
            signals = filter_by_symbols(signals, self.selected_symbols)
        
        # Group signals by type and create marker traces
        for signal_type in signals[signal_type_col].unique():
            # Standardize signal type to uppercase
            signal_type_upper = signal_type.upper()
            
            # Skip unknown signal types
            if signal_type_upper not in self.config.signal_colors:
                continue
                
            type_signals = signals[signals[signal_type_col] == signal_type]
            if type_signals.empty:
                continue
                
            style = self.config.signal_colors[signal_type_upper]
            self._add_signal_markers(type_signals, signal_type, style)

    def _add_signal_markers(self, signals: pd.DataFrame, 
                           name: str, style: Dict[str, Any]) -> None:
        """
        Add signal markers to the figure.
        
        Args:
            signals: DataFrame containing signal data
            name: Name for the signal trace
            style: Dictionary of style parameters
        """
        if self.fig is None:
            return
            
        # Get the price column to use for marker positions
        price_col = next((col for col in ['price', 'close'] 
                         if col in signals.columns), None)
                         
        if price_col is None and 'symbol' in signals.columns:
            # Try to merge price data to get a price value
            signals = pd.merge_asof(
                signals.sort_values('timestamp'),
                self.price_data[['timestamp', 'close', 'symbol']],
                on='timestamp',
                by='symbol',
                direction='nearest'
            )
            price_col = 'close'
            
        if price_col is None:
            # Can't display signals without price data
            return
        
        # Group by symbol if present
        if 'symbol' in signals.columns:
            for symbol in signals['symbol'].unique():
                symbol_signals = signals[signals['symbol'] == symbol]
                
                trace_name = f"{symbol} {name}"
                self.fig.add_trace(
                    go.Scatter(
                        x=symbol_signals['timestamp'],
                        y=symbol_signals[price_col],
                        mode='markers',
                        name=trace_name,
                        marker=dict(
                            symbol=style.get("symbol", "circle"),
                            size=style.get("size", 10),
                            color=style.get("color", "blue"),
                            line=dict(width=1, color=f"dark{style.get('color', 'blue')}")
                        ),
                        showlegend=True
                    ),
                    row=1, col=1
                )
        else:
            self.fig.add_trace(
                go.Scatter(
                    x=signals['timestamp'],
                    y=signals[price_col],
                    mode='markers',
                    name=name,
                    marker=dict(
                        symbol=style.get("symbol", "circle"),
                        size=style.get("size", 10),
                        color=style.get("color", "blue"),
                        line=dict(width=1, color=f"dark{style.get('color', 'blue')}")
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )

    def _add_regimes(self) -> None:
        """Add regime overlay traces to the figure."""
        if self.fig is None or self.regime_data is None or self.regime_data.empty:
            return
            
        # Check if we have regime probability columns
        regime_cols = [col for col in self.regime_data.columns if col.startswith('regime_')]
        if not regime_cols or 'timestamp' not in self.regime_data.columns:
            return
            
        # Filter regimes for selected symbols if needed
        regime_data = self.regime_data
        if self.selected_symbols and 'symbol' in regime_data.columns:
            regime_data = filter_by_symbols(regime_data, self.selected_symbols)
        
        # Create a secondary y-axis for regime probabilities
        self.fig.update_layout(
            yaxis2=dict(
                title="Regime Probability",
                overlaying="y",
                side="right",
                range=[0, 1],
                showgrid=False
            )
        )
        
        # If we have multiple symbols, create regime traces for each
        if 'symbol' in regime_data.columns and len(regime_data['symbol'].unique()) > 1:
            for symbol in self.selected_symbols or regime_data['symbol'].unique():
                symbol_regimes = regime_data[regime_data['symbol'] == symbol]
                
                if symbol_regimes.empty:
                    continue
                    
                for regime_col in regime_cols:
                    regime_name = regime_col.replace('regime_', '').capitalize()
                    trace_name = f"{symbol} {regime_name} Regime"
                    
                    self.fig.add_trace(
                        go.Scatter(
                            x=symbol_regimes['timestamp'],
                            y=symbol_regimes[regime_col],
                            name=trace_name,
                            line=dict(dash='dot'),
                            opacity=0.7,
                            yaxis="y2",
                            showlegend=True
                        ),
                        row=1, col=1
                    )
        else:
            # Single symbol or no symbol column
            for regime_col in regime_cols:
                regime_name = regime_col.replace('regime_', '').capitalize()
                
                self.fig.add_trace(
                    go.Scatter(
                        x=regime_data['timestamp'],
                        y=regime_data[regime_col],
                        name=f"{regime_name} Regime",
                        line=dict(dash='dot'),
                        opacity=0.7,
                        yaxis="y2",
                        showlegend=True
                    ),
                    row=1, col=1
                )


def create_price_chart(
    price_data: pd.DataFrame,
    signals_data: Optional[pd.DataFrame] = None,
    regime_data: Optional[pd.DataFrame] = None,
    selected_symbols: Optional[List[str]] = None,
    timeframe: str = "1h",
    show_volume: bool = True,
    show_ma: bool = False,
    ma_periods: Optional[List[int]] = None,
    chart_type: str = "candlestick",
    show_bollinger: bool = False,
    show_rsi: bool = False,
    chart_height: int = 600
) -> go.Figure:
    """
    Create an interactive price chart with optional signal and regime overlays.
    
    This function provides a simplified interface to the PriceChart class.
    
    Args:
        price_data: DataFrame containing OHLCV data
        signals_data: Optional DataFrame containing signal data
        regime_data: Optional DataFrame containing regime data
        selected_symbols: List of symbols to display
        timeframe: Timeframe of the chart (e.g., '1h', '1d')
        show_volume: Whether to show volume subplot
        show_ma: Whether to show moving averages
        ma_periods: List of periods for moving averages
        chart_type: Type of chart ('candlestick', 'ohlc', 'line')
        show_bollinger: Whether to show Bollinger Bands
        show_rsi: Whether to show RSI indicator
        chart_height: Height of the chart in pixels
        
    Returns:
        Plotly Figure object
    """
    try:
        chart_type_enum = ChartType(chart_type)
    except ValueError:
        chart_type_enum = ChartType.CANDLESTICK
    
    # Create configuration
    config = PriceChartConfig(
        chart_type=chart_type_enum,
        show_volume=show_volume,
        show_ma=show_ma,
        ma_periods=ma_periods if ma_periods else [20, 50, 200],
        show_bollinger=show_bollinger,
        show_rsi=show_rsi,
        chart_height=chart_height
    )
    
    # Create chart instance
    chart = PriceChart(
        price_data=price_data,
        signals_data=signals_data,
        regime_data=regime_data,
        selected_symbols=selected_symbols,
        timeframe=timeframe,
        config=config
    )
    
    # Create and return figure
    return chart.create_figure()


def get_technical_indicators(
    price_data: pd.DataFrame,
    ma_periods: Optional[List[int]] = None,
    rsi_period: int = 14,
    bollinger_period: int = 20,
    bollinger_std: float = 2.0
) -> Dict[str, pd.DataFrame]:
    """
    Calculate technical indicators for the given price data.
    
    Args:
        price_data: DataFrame containing price data
        ma_periods: List of periods for moving averages
        rsi_period: Period for RSI calculation
        bollinger_period: Period for Bollinger Bands
        bollinger_std: Standard deviations for Bollinger Bands
    
    Returns:
        Dictionary of indicator DataFrames
    """
    if price_data.empty or 'close' not in price_data.columns:
        return {}
    
    result = {}
    
    # Calculate moving averages
    if ma_periods:
        ma_data = pd.DataFrame({'timestamp': price_data['timestamp']})
        for period in ma_periods:
            ma_data[f'ma_{period}'] = calculate_moving_average(
                price_data['close'], period, 'sma'
            )
        result['moving_averages'] = ma_data
    
    # Calculate RSI
    rsi_data = pd.DataFrame({
        'timestamp': price_data['timestamp'],
        'rsi': calculate_rsi(price_data['close'], rsi_period)
    })
    result['rsi'] = rsi_data
    
    # Calculate Bollinger Bands
    middle, upper, lower = calculate_bollinger_bands(
        price_data['close'], bollinger_period, bollinger_std
    )
    bb_data = pd.DataFrame({
        'timestamp': price_data['timestamp'],
        'middle_band': middle,
        'upper_band': upper,
        'lower_band': lower
    })
    result['bollinger_bands'] = bb_data
    
    return result