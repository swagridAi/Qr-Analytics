"""
Enhanced base configuration module for visualization components.

This module provides an extended BaseConfig class with additional shared
functionality for consistent visualization styling and behavior across
all dashboard components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from quant_research.dashboard.chart_config import ChartTheme, ColorPalettes


@dataclass
class EnhancedBaseConfig:
    """Base configuration class for visualization components with enhanced shared functionality."""
    
    # Display settings
    chart_height: int = 600
    chart_width: Optional[int] = None
    theme: ChartTheme = ChartTheme.LIGHT
    
    # Layout settings
    show_grid: bool = True
    show_legend: bool = True
    margin: Dict[str, int] = field(default_factory=lambda: dict(l=50, r=50, t=70, b=50))
    
    # Date range settings
    start_date: Optional[Union[str, datetime, pd.Timestamp]] = None
    end_date: Optional[Union[str, datetime, pd.Timestamp]] = None
    
    # Symbol selection
    selected_symbols: List[str] = field(default_factory=list)
    
    # Responsive settings
    auto_height: bool = False
    min_height: int = 300
    max_height: int = 1200
    height_per_item: int = 50  # Height allocation per data item (for auto-scaling)
    
    # Interaction settings
    hover_mode: str = "x unified"  # "x unified", "closest", "x", "y"
    show_crosshair: bool = True
    tooltip_precision: int = 4    # Decimal precision in tooltips
    
    # Export settings
    enable_export: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["png", "svg", "csv"])
    
    # Color settings
    color_palette: Optional[List[str]] = None
    
    # Formatting settings
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    decimals: int = 2
    percentage_decimals: int = 2
    currency_symbol: str = "$"
    
    # Internationalization
    locale: str = "en-US"
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_defaults()
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate height
        if self.chart_height <= 0:
            raise ValueError("Chart height must be positive")
            
        # Validate width if provided
        if self.chart_width is not None and self.chart_width <= 0:
            raise ValueError("Chart width must be positive")
            
        # Validate dates if provided
        if self.start_date and self.end_date:
            try:
                start = pd.to_datetime(self.start_date)
                end = pd.to_datetime(self.end_date)
                if start > end:
                    raise ValueError("Start date must be before end date")
            except (ValueError, TypeError):
                raise ValueError("Invalid date format for start_date or end_date")
                
        # If auto_height is enabled, make sure min/max are valid
        if self.auto_height and self.min_height > self.max_height:
            raise ValueError("min_height must be less than or equal to max_height")
    
    def _setup_defaults(self) -> None:
        """Set up defaults based on configuration."""
        # Set up default color palette if not provided
        if self.color_palette is None:
            self.color_palette = ColorPalettes.ASSET_COLORS
    
    def get_theme_template(self) -> str:
        """Get the Plotly template name for the selected theme."""
        return "plotly_dark" if self.theme == ChartTheme.DARK else "plotly"
    
    def get_bg_colors(self) -> Dict[str, str]:
        """Get background colors for the selected theme."""
        if self.theme == ChartTheme.DARK:
            return {
                "paper_bgcolor": "rgba(0, 0, 0, 0.8)",
                "plot_bgcolor": "rgba(0, 0, 0, 0.8)"
            }
        return {
            "paper_bgcolor": "white",
            "plot_bgcolor": "white"
        }
    
    def get_text_color(self) -> str:
        """Get text color for the current theme."""
        return "white" if self.theme == ChartTheme.DARK else "black"
    
    def get_grid_color(self) -> str:
        """Get grid color for the current theme."""
        return "rgba(255, 255, 255, 0.1)" if self.theme == ChartTheme.DARK else "rgba(0, 0, 0, 0.1)"
    
    def get_legend_settings(self) -> Dict[str, Any]:
        """Get legend settings for the chart."""
        if not self.show_legend:
            return {"showlegend": False}
            
        return {
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
                "bgcolor": "rgba(255, 255, 255, 0.5)" if self.theme == ChartTheme.LIGHT else "rgba(0, 0, 0, 0.5)"
            }
        }
    
    def get_layout_settings(self, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Get common layout settings for the chart.
        
        Args:
            title: Optional title for the chart
            
        Returns:
            Dictionary of layout settings
        """
        # Calculate effective height
        effective_height = self.chart_height
        if self.auto_height and self.selected_symbols:
            item_count = len(self.selected_symbols)
            calculated_height = max(self.min_height, min(self.max_height, 
                                                       item_count * self.height_per_item))
            effective_height = calculated_height
        
        layout = {
            "height": effective_height,
            "template": self.get_theme_template(),
            "margin": self.margin,
            **self.get_bg_colors()
        }
        
        # Add title if provided
        if title:
            layout["title"] = title
            
        # Add width if specified
        if self.chart_width:
            layout["width"] = self.chart_width
        
        # Set hover mode based on crosshair setting
        layout["hovermode"] = self.hover_mode if self.show_crosshair else "closest"
            
        # Add legend settings
        layout.update(self.get_legend_settings())
            
        return layout
    
    def get_axis_settings(self, is_x_axis: bool = True) -> Dict[str, Any]:
        """
        Get common axis settings.
        
        Args:
            is_x_axis: Whether settings are for x-axis (True) or y-axis (False)
            
        Returns:
            Dictionary of axis settings
        """
        settings = {
            "showgrid": self.show_grid,
            "gridcolor": self.get_grid_color(),
            "gridwidth": 1,
            "linecolor": self.get_grid_color(),
            "title": {"font": {"color": self.get_text_color()}}
        }
        
        # X-axis specific settings
        if is_x_axis:
            settings.update({
                "rangeslider": {"visible": False},
                "showspikes": self.show_crosshair,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikecolor": self.get_grid_color(),
                "spikethickness": 1
            })
        # Y-axis specific settings
        else:
            settings.update({
                "showspikes": self.show_crosshair,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikecolor": self.get_grid_color(),
                "spikethickness": 1,
                "automargin": True
            })
            
        return settings
    
    def format_date(self, date_value: Union[datetime, str, pd.Timestamp]) -> str:
        """Format a date value according to configuration."""
        try:
            if isinstance(date_value, str):
                date_value = pd.to_datetime(date_value)
            return date_value.strftime(self.date_format)
        except (AttributeError, ValueError, TypeError):
            return str(date_value)
    
    def format_time(self, time_value: Union[datetime, str, pd.Timestamp]) -> str:
        """Format a time value according to configuration."""
        try:
            if isinstance(time_value, str):
                time_value = pd.to_datetime(time_value)
            return time_value.strftime(self.time_format)
        except (AttributeError, ValueError, TypeError):
            return str(time_value)
    
    def format_datetime(self, dt_value: Union[datetime, str, pd.Timestamp]) -> str:
        """Format a datetime value according to configuration."""
        try:
            if isinstance(dt_value, str):
                dt_value = pd.to_datetime(dt_value)
            return dt_value.strftime(f"{self.date_format} {self.time_format}")
        except (AttributeError, ValueError, TypeError):
            return str(dt_value)
    
    def format_number(self, value: float) -> str:
        """Format a numeric value according to configuration."""
        try:
            return f"{value:,.{self.decimals}f}"
        except (ValueError, TypeError):
            return str(value)
    
    def format_currency(self, value: float, include_sign: bool = False) -> str:
        """
        Format a currency value according to configuration.
        
        Args:
            value: The value to format
            include_sign: Whether to include sign for positive values
            
        Returns:
            Formatted currency string
        """
        try:
            sign = "+" if include_sign and value > 0 else ""
            return f"{sign}{self.currency_symbol}{abs(value):,.{self.decimals}f}"
        except (ValueError, TypeError):
            return str(value)
    
    def format_percentage(self, value: float, include_sign: bool = False) -> str:
        """
        Format a percentage value according to configuration.
        
        Args:
            value: The value to format (0.01 = 1%)
            include_sign: Whether to include sign for positive values
            
        Returns:
            Formatted percentage string
        """
        try:
            sign = "+" if include_sign and value > 0 else ""
            return f"{sign}{value * 100:.{self.percentage_decimals}f}%"
        except (ValueError, TypeError):
            return str(value)
    
    def get_color_for_value(self, value: float, neutral_threshold: float = 0.0) -> str:
        """
        Get color for a numeric value based on theme and sign.
        
        Args:
            value: The numeric value
            neutral_threshold: Threshold for neutral color
            
        Returns:
            Color string in rgba format
        """
        try:
            if value > neutral_threshold:
                return "rgba(0, 128, 0, 0.7)"  # Green for positive
            elif value < neutral_threshold:
                return "rgba(255, 0, 0, 0.7)"  # Red for negative
            else:
                return "rgba(128, 128, 128, 0.7)"  # Gray for neutral
        except (ValueError, TypeError):
            return "rgba(128, 128, 128, 0.7)"  # Gray for invalid
    
    def get_symbol_color(self, symbol_index: int) -> str:
        """
        Get color for a symbol based on index.
        
        Args:
            symbol_index: Index of the symbol
            
        Returns:
            Color string from palette
        """
        if self.color_palette and len(self.color_palette) > 0:
            return self.color_palette[symbol_index % len(self.color_palette)]
        return ColorPalettes.ASSET_COLORS[symbol_index % len(ColorPalettes.ASSET_COLORS)]
    
    def filter_data_by_date_range(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Filter DataFrame by date range if configured.
        
        Args:
            df: DataFrame to filter
            timestamp_col: Name of timestamp column
            
        Returns:
            Filtered DataFrame
        """
        if df is None or df.empty or timestamp_col not in df.columns:
            return df
        
        filtered_df = df.copy()
        
        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(filtered_df[timestamp_col]):
            filtered_df[timestamp_col] = pd.to_datetime(filtered_df[timestamp_col])
        
        # Apply start date filter
        if self.start_date is not None:
            start_date = pd.to_datetime(self.start_date)
            filtered_df = filtered_df[filtered_df[timestamp_col] >= start_date]
            
        # Apply end date filter
        if self.end_date is not None:
            end_date = pd.to_datetime(self.end_date)
            filtered_df = filtered_df[filtered_df[timestamp_col] <= end_date]
            
        return filtered_df
    
    def filter_data_by_symbols(self, df: pd.DataFrame, symbol_col: str = 'symbol') -> pd.DataFrame:
        """
        Filter DataFrame by selected symbols if configured.
        
        Args:
            df: DataFrame to filter
            symbol_col: Name of symbol column
            
        Returns:
            Filtered DataFrame
        """
        if df is None or df.empty or symbol_col not in df.columns or not self.selected_symbols:
            return df
            
        return df[df[symbol_col].isin(self.selected_symbols)]
    
    def calculate_adaptive_height(self, data_count: int) -> int:
        """
        Calculate adaptive chart height based on data quantity.
        
        Args:
            data_count: Number of data items to display
            
        Returns:
            Calculated chart height
        """
        if not self.auto_height:
            return self.chart_height
            
        base_height = 200  # Minimum base height
        item_height = self.height_per_item
        
        calculated_height = base_height + (data_count * item_height)
        
        # Apply min/max constraints
        return max(self.min_height, min(self.max_height, calculated_height))
    
    def create_empty_message_trace(self, message: str = "No data available") -> Dict[str, Any]:
        """
        Create a trace configuration for an empty state message.
        
        Args:
            message: Message to display
            
        Returns:
            Dictionary with trace configuration
        """
        return {
            "type": "scatter",
            "x": [0.5],
            "y": [0.5],
            "text": [message],
            "mode": "text",
            "textfont": {
                "color": self.get_text_color(),
                "size": 14
            },
            "showlegend": False,
            "xaxis": "x",
            "yaxis": "y"
        }
    
    def generate_hover_template(
        self, 
        mode: str = "price", 
        include_fields: Optional[List[str]] = None
    ) -> str:
        """
        Generate a hover template for consistent tooltips.
        
        Args:
            mode: Template mode ('price', 'signal', 'performance', etc.)
            include_fields: Optional list of fields to include
            
        Returns:
            Hover template string
        """
        if mode == "price":
            return (
                "<b>%{x|" + self.date_format + " " + self.time_format + "}</b><br>" +
                "Open: " + self.currency_symbol + "%{customdata[0]:,." + str(self.decimals) + "f}<br>" +
                "High: " + self.currency_symbol + "%{customdata[1]:,." + str(self.decimals) + "f}<br>" +
                "Low: " + self.currency_symbol + "%{customdata[2]:,." + str(self.decimals) + "f}<br>" +
                "Close: " + self.currency_symbol + "%{customdata[3]:,." + str(self.decimals) + "f}<br>" +
                "Volume: %{customdata[4]:,.0f}<extra></extra>"
            )
        elif mode == "signal":
            return (
                "<b>%{x|" + self.date_format + " " + self.time_format + "}</b><br>" +
                "Signal: %{customdata[0]}<br>" +
                "Price: " + self.currency_symbol + "%{y:,." + str(self.decimals) + "f}<br>" +
                "Strength: %{customdata[1]:." + str(self.decimals) + "f}<extra></extra>"
            )
        elif mode == "performance":
            return (
                "<b>%{x|" + self.date_format + "}</b><br>" +
                "Value: " + self.currency_symbol + "%{y:,." + str(self.decimals) + "f}<br>" +
                "Change: %{customdata[0]:+." + str(self.percentage_decimals) + "%}<extra></extra>"
            )
        else:
            # Generic template
            return "<b>%{x}</b><br>%{y}<extra></extra>"


# Create a singleton instance for global access
default_config = EnhancedBaseConfig()