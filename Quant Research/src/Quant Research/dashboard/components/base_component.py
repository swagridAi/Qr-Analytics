"""
Base visualization component for the Quant Research dashboard.

This module provides a common base class for all visualization components
with shared utilities and standardized interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go


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
    
    # Override in subclasses
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        pass


class BaseVisualization(ABC):
    """
    Abstract base class for creating visualization components.
    
    This class provides common functionality for all visualization
    components including figure creation, layout management, and 
    utility methods.
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
            
        # Get base config from first level subclass
        config = getattr(self, 'config', None)
        if not isinstance(config, BaseConfig):
            config = BaseConfig()
            
        # Base layout updates
        layout_updates = dict(
            title=title,
            height=config.chart_height,
            margin=dict(l=50, r=50, t=70, b=50)
        )
        
        # Add width if specified
        if config.chart_width:
            layout_updates["width"] = config.chart_width
        
        # Configure theme
        if config.theme == "dark":
            layout_updates.update(
                template="plotly_dark",
                paper_bgcolor="rgba(0, 0, 0, 0.8)",
                plot_bgcolor="rgba(0, 0, 0, 0.8)"
            )
        
        # Configure legend
        if config.show_legend:
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
        self.fig.update_xaxes(showgrid=config.show_grid)
        self.fig.update_yaxes(showgrid=config.show_grid)
    
    def _create_empty_figure(self, message: str) -> go.Figure:
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
            x=0.5, y=0.5, showarrow=False
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
            row=row, col=1
        )
    
    @staticmethod
    def format_date_range(df: pd.DataFrame, date_col: str = 'timestamp') -> Optional[List[pd.Timestamp]]:
        """
        Extract date range from a DataFrame.
        
        Args:
            df: DataFrame containing date column
            date_col: Name of the date column
            
        Returns:
            List of [min_date, max_date] or None if no dates
        """
        if df.empty or date_col not in df.columns:
            return None
            
        return [df[date_col].min(), df[date_col].max()]
    
    @staticmethod
    def format_currency(value: float, symbol: str = "$", decimals: int = 2) -> str:
        """
        Format a value as currency.
        
        Args:
            value: Value to format
            symbol: Currency symbol
            decimals: Number of decimal places
            
        Returns:
            Formatted currency string
        """
        return f"{symbol}{abs(value):,.{decimals}f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """
        Format a value as percentage.
        
        Args:
            value: Value to format (0.01 = 1%)
            decimals: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        return f"{value * 100:.{decimals}f}%"
    
    @staticmethod
    def filter_dataframe(
        df: pd.DataFrame, 
        filter_col: str, 
        filter_values: List[Any]
    ) -> pd.DataFrame:
        """
        Filter a DataFrame to include only specified values.
        
        Args:
            df: DataFrame to filter
            filter_col: Column to filter on
            filter_values: Values to include
            
        Returns:
            Filtered DataFrame
        """
        if df.empty or filter_col not in df.columns:
            return df
            
        return df[df[filter_col].isin(filter_values)]