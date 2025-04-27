"""
Visualization configuration for the Quant Research dashboard.

This module provides standardized configuration classes and constants
for all dashboard visualization components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union

import plotly.express as px


class ChartTheme(str, Enum):
    """Chart theme options."""
    LIGHT = "light"
    DARK = "dark"


class ColorPalettes:
    """Color palettes for different chart elements."""
    
    # Main asset colors (consistent across views)
    ASSET_COLORS = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf"   # Cyan
    ]
    
    # Price chart colors
    PRICE_COLORS = {
        "increasing": "#26a69a",
        "decreasing": "#ef5350",
        "volume_up": "rgba(38, 166, 154, 0.5)",
        "volume_down": "rgba(239, 83, 80, 0.5)"
    }
    
    # Sentiment colors
    SENTIMENT_COLORS = {
        "positive": "#26a69a",
        "negative": "#ef5350",
        "neutral": "#7f7f7f",
        "correlation": "#9467bd"
    }
    
    # Performance colors
    PERFORMANCE_COLORS = {
        "equity": "#1f77b4",
        "benchmark": "#7f7f7f",
        "drawdown": "rgba(255, 0, 0, 0.7)",
        "underwater": "rgba(255, 165, 0, 0.7)",
        "returns_positive": "rgba(0, 128, 0, 0.7)",
        "returns_negative": "rgba(255, 0, 0, 0.7)"
    }
    
    # Portfolio colors
    PORTFOLIO_COLORS = {
        "long": "rgba(0, 128, 0, 0.7)",
        "short": "rgba(255, 0, 0, 0.7)",
        "cash": "rgba(192, 192, 192, 0.7)",
        "other": "rgba(128, 128, 128, 0.7)"
    }
    
    # Risk metrics colors
    RISK_COLORS = {
        "var": "rgba(255, 0, 0, 0.7)",
        "es": "rgba(255, 165, 0, 0.7)",
        "vol": "rgba(65, 105, 225, 0.7)",
        "beta": "rgba(128, 0, 128, 0.7)",
        "te": "rgba(0, 128, 128, 0.7)"
    }
    
    # Signal markers
    SIGNAL_MARKERS = {
        "buy": {"color": "green", "symbol": "triangle-up", "size": 10},
        "sell": {"color": "red", "symbol": "triangle-down", "size": 10},
        "entry": {"color": "blue", "symbol": "circle", "size": 8},
        "exit": {"color": "purple", "symbol": "circle-x", "size": 8}
    }
    
    # Sector colors (for portfolio breakdowns)
    SECTOR_COLORS = {
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
    }
    
    # Qualitative color scales for allocation charts
    QUALITATIVE_COLORS = px.colors.qualitative.Plotly


@dataclass
class BaseChartConfig:
    """Base configuration for all chart components."""
    
    # Display settings
    chart_height: int = 600
    chart_width: Optional[int] = None
    theme: ChartTheme = ChartTheme.LIGHT
    
    # Layout settings
    show_grid: bool = True
    show_legend: bool = True
    margin: Dict[str, int] = field(default_factory=lambda: dict(l=50, r=50, t=70, b=50))
    
    # Time period settings
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
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


@dataclass
class TextFormatting:
    """Text formatting options for visualizations."""
    
    currency_symbol: str = "$"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    decimals: int = 2
    percentage_decimals: int = 2
    
    def format_currency(self, value: float) -> str:
        """Format a value as currency."""
        return f"{self.currency_symbol}{abs(value):,.{self.decimals}f}"
    
    def format_percentage(self, value: float) -> str:
        """Format a value as percentage."""
        return f"{value * 100:.{self.percentage_decimals}f}%"
    
    def format_datetime(self, dt) -> str:
        """Format a datetime value."""
        try:
            return dt.strftime(f"{self.date_format} {self.time_format}")
        except (AttributeError, ValueError):
            return str(dt)
    
    def format_date(self, dt) -> str:
        """Format a date value."""
        try:
            return dt.strftime(self.date_format)
        except (AttributeError, ValueError):
            return str(dt)


# Default configuration instances
DEFAULT_CONFIG = BaseChartConfig()
DEFAULT_FORMATTING = TextFormatting()