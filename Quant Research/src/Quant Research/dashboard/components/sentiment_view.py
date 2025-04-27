"""
Sentiment view component for the Quant Research dashboard.

This module provides a comprehensive sentiment visualization component
with correlation analysis, source breakdown, and interactive features.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CorrelationMethod(str, Enum):
    """Methods for calculating correlation between sentiment and prices."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class SentimentDisplayMode(str, Enum):
    """Display modes for sentiment visualization."""
    COMBINED = "combined"  # All sources in one line
    STACKED = "stacked"    # Stacked area chart for sources
    SEPARATE = "separate"  # Separate lines for each source


@dataclass
class SentimentViewConfig:
    """Configuration settings for sentiment visualization."""
    
    # Display settings
    chart_height: int = 600
    chart_width: Optional[int] = None
    theme: str = "white"  # "white" or "dark"
    display_mode: SentimentDisplayMode = SentimentDisplayMode.COMBINED
    
    # Correlation settings
    show_correlation: bool = True
    correlation_window: int = 20
    correlation_method: CorrelationMethod = CorrelationMethod.PEARSON
    sentiment_lag: int = 1  # Lag sentiment by N periods (for predictive analysis)
    
    # Source settings
    show_sources: bool = True
    source_opacity: float = 0.7
    
    # Distribution settings
    show_distribution: bool = False
    distribution_bins: int = 20
    
    # Metric settings
    show_metrics: bool = True
    
    # Color settings
    sentiment_color: str = "blue"
    source_colors: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"
    ])
    correlation_color: str = "purple"
    
    # Layout settings
    show_grid: bool = True
    show_legend: bool = True
    
    # Advanced settings
    hover_info: str = "all"  # "all", "minimal"
    smoothing_factor: Optional[int] = None  # Apply smoothing to sentiment line
    normalize_sentiment: bool = True  # Scale sentiment to [-1, 1]


class SentimentView:
    """
    Class for creating and managing sentiment visualizations.
    
    This class handles data validation, figure creation, and adding
    different visualizations for sentiment analysis.
    """
    
    def __init__(
        self,
        sentiment_data: pd.DataFrame,
        price_data: Optional[pd.DataFrame] = None,
        selected_symbols: Optional[List[str]] = None,
        config: Optional[SentimentViewConfig] = None
    ):
        """
        Initialize the sentiment view with data and configuration.
        
        Args:
            sentiment_data: DataFrame containing sentiment scores with columns:
                [timestamp, sentiment_score, symbol(optional), source_*(optional)]
            price_data: Optional DataFrame with price data for correlation analysis
            selected_symbols: List of symbols to display
            config: Optional configuration object
        
        Raises:
            ValueError: If required data columns are missing
        """
        self.sentiment_data = sentiment_data.copy()
        self.price_data = price_data.copy() if price_data is not None else None
        self.selected_symbols = selected_symbols
        self.config = config or SentimentViewConfig()
        
        # Initialize chart state
        self.fig = None
        self.date_range = None
        self.current_row = 0
        self.metrics = {}
        
        # Validate input data
        self._validate_data()
        
    def _validate_data(self) -> None:
        """
        Validate the input data has required columns.
        
        Raises:
            ValueError: If required columns are missing
        """
        if self.sentiment_data.empty:
            return
            
        required_columns = ['timestamp', 'sentiment_score']
        
        missing_columns = [col for col in required_columns if col not in self.sentiment_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in sentiment data: {missing_columns}")
        
        # Filter for selected symbols if provided
        if self.selected_symbols and 'symbol' in self.sentiment_data.columns:
            self.sentiment_data = self.sentiment_data[self.sentiment_data['symbol'].isin(self.selected_symbols)]
            
            # Ensure we have data after filtering
            if self.sentiment_data.empty:
                raise ValueError(f"No sentiment data found for selected symbols: {self.selected_symbols}")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.sentiment_data['timestamp']):
            self.sentiment_data['timestamp'] = pd.to_datetime(self.sentiment_data['timestamp'])
        
        # Preprocess price data if available
        if self.price_data is not None and not self.price_data.empty:
            if 'timestamp' not in self.price_data.columns or 'close' not in self.price_data.columns:
                raise ValueError("Price data must contain 'timestamp' and 'close' columns for correlation analysis")
                
            # Filter price data for selected symbols if provided
            if self.selected_symbols and 'symbol' in self.price_data.columns:
                self.price_data = self.price_data[self.price_data['symbol'].isin(self.selected_symbols)]
                
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(self.price_data['timestamp']):
                self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
        
        # Normalize sentiment if requested
        if self.config.normalize_sentiment:
            self._normalize_sentiment()
            
        # Identify source columns
        self.source_columns = [col for col in self.sentiment_data.columns 
                              if col.startswith('source_') and col != 'source_count']
        
    def _normalize_sentiment(self) -> None:
        """Normalize sentiment scores to the range [-1, 1]."""
        # Check if already in range
        min_score = self.sentiment_data['sentiment_score'].min()
        max_score = self.sentiment_data['sentiment_score'].max()
        
        if min_score >= -1 and max_score <= 1:
            return
            
        # Find absolute max to preserve sign
        abs_max = max(abs(min_score), abs(max_score))
        if abs_max > 0:
            self.sentiment_data['sentiment_score'] = self.sentiment_data['sentiment_score'] / abs_max
        
    def create_figure(self) -> go.Figure:
        """
        Create the complete sentiment visualization with all components.
        
        Returns:
            Plotly Figure object with all requested visualization elements
        
        Raises:
            RuntimeError: If figure creation fails
        """
        if self.sentiment_data.empty:
            # Return empty figure with message
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No sentiment data available for the selected parameters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return empty_fig
        
        try:
            # Determine number of subplots
            num_rows = 1  # Sentiment plot
            
            if self.config.show_correlation and self.price_data is not None and not self.price_data.empty:
                num_rows += 1  # Correlation plot
                
            if self.config.show_distribution:
                num_rows += 1  # Distribution plot
            
            # Create subplot titles
            subplot_titles = ["Sentiment Analysis"]
            if self.config.show_correlation and self.price_data is not None:
                subplot_titles.append("Price-Sentiment Correlation")
            if self.config.show_distribution:
                subplot_titles.append("Sentiment Distribution")
            
            # Calculate row heights
            if num_rows == 1:
                row_heights = [1]
            elif num_rows == 2:
                row_heights = [0.7, 0.3]
            else:
                row_heights = [0.6, 0.2, 0.2]
            
            # Create figure with subplots
            self.fig = make_subplots(
                rows=num_rows,
                cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=subplot_titles,
                row_heights=row_heights
            )
            
            # Track current row for adding traces
            self.current_row = 1
            
            # Add sentiment score visualization
            self._add_sentiment_traces()
            
            # Add correlation analysis if requested
            if self.config.show_correlation and self.price_data is not None and not self.price_data.empty:
                self.current_row += 1
                self._add_correlation_traces()
            
            # Add sentiment distribution if requested
            if self.config.show_distribution:
                self.current_row += 1
                self._add_distribution_traces()
            
            # Calculate sentiment metrics
            self._calculate_metrics()
            
            # Update layout
            self._update_layout()
            
            return self.fig
            
        except Exception as e:
            # Create error figure
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Error creating sentiment visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return error_fig
    
    def _add_sentiment_traces(self) -> None:
        """Add sentiment score traces to the figure."""
        # If we have multiple symbols, create separate traces for each
        if 'symbol' in self.sentiment_data.columns and len(self.sentiment_data['symbol'].unique()) > 1:
            for i, symbol in enumerate(self.selected_symbols):
                symbol_sentiment = self.sentiment_data[self.sentiment_data['symbol'] == symbol]
                
                if symbol_sentiment.empty:
                    continue
                    
                # Add main sentiment trace
                color = self.config.source_colors[i % len(self.config.source_colors)]
                self._add_sentiment_trace(symbol_sentiment, symbol, color)
                
                # Add source breakdown if available and requested
                if self.config.show_sources and self.source_columns:
                    self._add_source_traces(symbol_sentiment, symbol)
        else:
            # Single symbol or no symbol column
            self._add_sentiment_trace(self.sentiment_data, None, self.config.sentiment_color)
            
            # Add source breakdown if available and requested
            if self.config.show_sources and self.source_columns:
                self._add_source_traces(self.sentiment_data)

    def _add_sentiment_trace(self, data: pd.DataFrame, symbol: Optional[str] = None, 
                           color: str = "blue") -> None:
        """
        Add a sentiment score trace to the figure.
        
        Args:
            data: DataFrame containing sentiment data
            symbol: Optional symbol name for the trace
            color: Color for the trace
        """
        trace_name = f"{symbol + ' ' if symbol else ''}Sentiment"
        
        # Apply smoothing if requested
        if self.config.smoothing_factor:
            y_data = data['sentiment_score'].rolling(
                window=self.config.smoothing_factor, 
                center=True
            ).mean()
        else:
            y_data = data['sentiment_score']
        
        self.fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=y_data,
                name=trace_name,
                line=dict(width=2, color=color),
                showlegend=True
            ),
            row=self.current_row, col=1
        )
        
        # Update date range
        if self.date_range is None:
            self.date_range = [data['timestamp'].min(), data['timestamp'].max()]
        else:
            self.date_range[0] = min(self.date_range[0], data['timestamp'].min())
            self.date_range[1] = max(self.date_range[1], data['timestamp'].max())
        
        # Add zero reference line
        self.fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="gray",
            row=self.current_row, col=1
        )

    def _add_source_traces(self, data: pd.DataFrame, symbol: Optional[str] = None) -> None:
        """
        Add source breakdown traces to the figure.
        
        Args:
            data: DataFrame containing sentiment source data
            symbol: Optional symbol name for the traces
        """
        if not self.source_columns:
            return
            
        # Handle different display modes
        if self.config.display_mode == SentimentDisplayMode.STACKED:
            # Create a stacked area chart for sources
            for i, source_col in enumerate(self.source_columns):
                source_name = source_col.replace('source_', '').capitalize()
                trace_name = f"{symbol + ' ' if symbol else ''}{source_name}"
                color = self.config.source_colors[i % len(self.config.source_colors)]
                
                self.fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data[source_col],
                        name=trace_name,
                        mode='lines',
                        stackgroup='one',  # Stack sources
                        line=dict(width=0.5, color=color),
                        opacity=self.config.source_opacity,
                        showlegend=True
                    ),
                    row=self.current_row, col=1
                )
                
        elif self.config.display_mode == SentimentDisplayMode.SEPARATE:
            # Create separate lines for each source
            for i, source_col in enumerate(self.source_columns):
                source_name = source_col.replace('source_', '').capitalize()
                trace_name = f"{symbol + ' ' if symbol else ''}{source_name}"
                color = self.config.source_colors[i % len(self.config.source_colors)]
                
                self.fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data[source_col],
                        name=trace_name,
                        line=dict(width=1, color=color),
                        opacity=self.config.source_opacity,
                        showlegend=True
                    ),
                    row=self.current_row, col=1
                )
                
        else:  # COMBINED (default)
            # Add each source as a dotted line
            for i, source_col in enumerate(self.source_columns):
                source_name = source_col.replace('source_', '').capitalize()
                trace_name = f"{symbol + ' ' if symbol else ''}{source_name}"
                color = self.config.source_colors[i % len(self.config.source_colors)]
                
                self.fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data[source_col],
                        name=trace_name,
                        line=dict(width=1, dash='dot', color=color),
                        opacity=self.config.source_opacity,
                        showlegend=True
                    ),
                    row=self.current_row, col=1
                )

    def _add_correlation_traces(self) -> None:
        """Add correlation analysis traces to the figure."""
        if self.price_data is None or self.price_data.empty:
            return
            
        # If we have multiple symbols, create correlation traces for each
        if ('symbol' in self.sentiment_data.columns and 
            'symbol' in self.price_data.columns and 
            len(self.sentiment_data['symbol'].unique()) > 1):
            
            for i, symbol in enumerate(self.selected_symbols):
                symbol_sentiment = self.sentiment_data[self.sentiment_data['symbol'] == symbol]
                symbol_price = self.price_data[self.price_data['symbol'] == symbol]
                
                if symbol_sentiment.empty or symbol_price.empty:
                    continue
                    
                color = self.config.source_colors[i % len(self.config.source_colors)]
                self._add_correlation_trace(symbol_sentiment, symbol_price, symbol, color)
        else:
            # Single symbol or no symbol column
            self._add_correlation_trace(
                self.sentiment_data, 
                self.price_data, 
                None, 
                self.config.correlation_color
            )

    def _add_correlation_trace(self, sentiment_data: pd.DataFrame, price_data: pd.DataFrame,
                              symbol: Optional[str] = None, color: str = "purple") -> None:
        """
        Add a correlation trace between sentiment and price data.
        
        Args:
            sentiment_data: DataFrame containing sentiment scores
            price_data: DataFrame containing price data
            symbol: Optional symbol name for the trace
            color: Color for the trace
        """
        # Align timestamps for correlation
        merged_data = pd.merge_asof(
            sentiment_data.sort_values('timestamp'),
            price_data[['timestamp', 'close']].sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        # Need sufficient data for correlation
        window = self.config.correlation_window
        if len(merged_data) <= window:
            return
            
        # Calculate price returns
        merged_data['price_return'] = merged_data['close'].pct_change()
        
        # Calculate lagged sentiment
        lag = self.config.sentiment_lag
        if lag > 0:
            merged_data['sentiment_lagged'] = merged_data['sentiment_score'].shift(lag)
        else:
            merged_data['sentiment_lagged'] = merged_data['sentiment_score']
        
        # Calculate rolling correlation based on method
        if self.config.correlation_method == CorrelationMethod.SPEARMAN:
            # Spearman rank correlation
            merged_data['correlation'] = merged_data['sentiment_lagged'].rolling(
                window=window
            ).corr(merged_data['price_return'], method='spearman')
        elif self.config.correlation_method == CorrelationMethod.KENDALL:
            # Kendall's tau correlation
            merged_data['correlation'] = merged_data['sentiment_lagged'].rolling(
                window=window
            ).corr(merged_data['price_return'], method='kendall')
        else:
            # Default to Pearson correlation
            merged_data['correlation'] = merged_data['sentiment_lagged'].rolling(
                window=window
            ).corr(merged_data['price_return'])
        
        trace_name = f"{symbol + ' ' if symbol else ''}Sentiment-Price Correlation"
        
        self.fig.add_trace(
            go.Scatter(
                x=merged_data['timestamp'],
                y=merged_data['correlation'],
                name=trace_name,
                line=dict(color=color),
                showlegend=True
            ),
            row=self.current_row, col=1
        )
        
        # Add reference lines
        self.fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="gray",
            row=self.current_row, col=1
        )
        
        # Add bands for correlation significance
        # (rough approximation, actual significance depends on sample size)
        self.fig.add_hline(
            y=0.5, 
            line_dash="dot", 
            line_color="rgba(0,255,0,0.3)",
            row=self.current_row, col=1
        )
        self.fig.add_hline(
            y=-0.5, 
            line_dash="dot", 
            line_color="rgba(255,0,0,0.3)",
            row=self.current_row, col=1
        )

    def _add_distribution_traces(self) -> None:
        """Add sentiment distribution visualization to the figure."""
        # If we have multiple symbols, create distribution traces for each
        if 'symbol' in self.sentiment_data.columns and len(self.sentiment_data['symbol'].unique()) > 1:
            for i, symbol in enumerate(self.selected_symbols):
                symbol_sentiment = self.sentiment_data[self.sentiment_data['symbol'] == symbol]
                
                if symbol_sentiment.empty:
                    continue
                    
                color = self.config.source_colors[i % len(self.config.source_colors)]
                self._add_distribution_trace(symbol_sentiment, symbol, color)
        else:
            # Single symbol or no symbol column
            self._add_distribution_trace(
                self.sentiment_data, 
                None, 
                self.config.sentiment_color
            )

    def _add_distribution_trace(self, data: pd.DataFrame, symbol: Optional[str] = None, 
                               color: str = "blue") -> None:
        """
        Add a sentiment distribution histogram.
        
        Args:
            data: DataFrame containing sentiment data
            symbol: Optional symbol name for the trace
            color: Color for the trace
        """
        trace_name = f"{symbol + ' ' if symbol else ''}Sentiment Distribution"
        
        # Create histogram
        self.fig.add_trace(
            go.Histogram(
                x=data['sentiment_score'],
                name=trace_name,
                marker_color=color,
                opacity=0.7,
                nbinsx=self.config.distribution_bins,
                showlegend=True
            ),
            row=self.current_row, col=1
        )
        
        # Add reference line at zero
        self.fig.add_vline(
            x=0, 
            line_dash="dash", 
            line_color="gray",
            row=self.current_row, col=1
        )

    def _calculate_metrics(self) -> None:
        """Calculate summary metrics from sentiment data."""
        # If we have multiple symbols, calculate metrics for each
        if 'symbol' in self.sentiment_data.columns and len(self.sentiment_data['symbol'].unique()) > 1:
            for symbol in self.selected_symbols:
                symbol_sentiment = self.sentiment_data[self.sentiment_data['symbol'] == symbol]
                
                if symbol_sentiment.empty:
                    continue
                    
                self.metrics[symbol] = self._calculate_symbol_metrics(symbol_sentiment)
        else:
            # Single symbol or no symbol column
            self.metrics['overall'] = self._calculate_symbol_metrics(self.sentiment_data)

    def _calculate_symbol_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate sentiment metrics for a single symbol.
        
        Args:
            data: DataFrame containing sentiment data for one symbol
            
        Returns:
            Dictionary of calculated metrics
        """
        if data.empty:
            return {}
            
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Calculate current sentiment (most recent)
        current_sentiment = data['sentiment_score'].iloc[-1]
        
        # Calculate sentiment trend
        if len(data) > 5:
            # Linear regression for trend
            x = np.arange(len(data.iloc[-5:]))
            y = data.iloc[-5:]['sentiment_score'].values
            trend_coef = np.polyfit(x, y, 1)[0]
            
            if trend_coef > 0.01:
                trend = "Improving"
            elif trend_coef < -0.01:
                trend = "Deteriorating"
            else:
                trend = "Stable"
        else:
            trend = "Insufficient data"
        
        # Calculate sentiment volatility
        volatility = data['sentiment_score'].std() if len(data) > 1 else None
        
        # Calculate source distribution if available
        source_distribution = {}
        for source_col in self.source_columns:
            source_name = source_col.replace('source_', '').capitalize()
            if source_col in data.columns:
                source_distribution[source_name] = data[source_col].iloc[-1]
        
        # Calculate extremes
        sentiment_max = data['sentiment_score'].max()
        sentiment_min = data['sentiment_score'].min()
        
        # Calculate correlation with price if available
        price_correlation = None
        if self.price_data is not None and not self.price_data.empty and 'close' in self.price_data.columns:
            price_symbol_data = self.price_data
            if 'symbol' in self.price_data.columns:
                symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else None
                if symbol:
                    price_symbol_data = self.price_data[self.price_data['symbol'] == symbol]
            
            if not price_symbol_data.empty:
                # Merge sentiment and price data
                merged = pd.merge_asof(
                    data.sort_values('timestamp'),
                    price_symbol_data[['timestamp', 'close']].sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest'
                )
                
                # Calculate correlation
                if len(merged) > 5 and 'close' in merged.columns:
                    price_correlation = merged['sentiment_score'].corr(
                        merged['close'],
                        method=self.config.correlation_method.value
                    )
        
        return {
            'current': current_sentiment,
            'trend': trend,
            'volatility': volatility,
            'max': sentiment_max,
            'min': sentiment_min,
            'sources': source_distribution,
            'price_correlation': price_correlation
        }

    def _update_layout(self) -> None:
        """Update the figure layout with titles and formatting."""
        # Set chart title
        symbols_text = f" ({', '.join(self.selected_symbols)})" if self.selected_symbols else ""
        title = f"Sentiment Analysis{symbols_text}"
        
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
        
        # Update grid settings
        self.fig.update_xaxes(
            title_text="Date", 
            showgrid=self.config.show_grid
        )
        
        # Update y-axis labels for sentiment
        y_min = min(0, self.sentiment_data['sentiment_score'].min() * 1.1)
        y_max = max(0, self.sentiment_data['sentiment_score'].max() * 1.1)
        
        self.fig.update_yaxes(
            title_text="Sentiment Score", 
            range=[y_min, y_max],
            showgrid=self.config.show_grid,
            row=1, col=1
        )
        
        # Update y-axis for correlation subplot
        if self.config.show_correlation and self.price_data is not None:
            self.fig.update_yaxes(
                title_text=f"{self.config.correlation_method.value.capitalize()} Correlation", 
                range=[-1, 1],
                showgrid=self.config.show_grid,
                row=2, col=1
            )
        
        # Update axes for distribution subplot
        if self.config.show_distribution:
            distribution_row = 3 if self.config.show_correlation else 2
            self.fig.update_xaxes(
                title_text="Sentiment Score",
                showgrid=self.config.show_grid,
                row=distribution_row, col=1
            )
            self.fig.update_yaxes(
                title_text="Frequency",
                showgrid=self.config.show_grid,
                row=distribution_row, col=1
            )
        
        # Set consistent date range for time-series plots
        if self.date_range:
            # Only apply to time-series subplots, not distribution
            rows_to_update = 1
            if self.config.show_correlation:
                rows_to_update += 1
                
            for row in range(1, rows_to_update + 1):
                self.fig.update_xaxes(range=self.date_range, row=row, col=1)
                
        # Add metrics annotations if requested
        if self.config.show_metrics and self.metrics:
            self._add_metrics_annotations()

    def _add_metrics_annotations(self) -> None:
        """Add sentiment metrics as annotations on the chart."""
        if not self.metrics:
            return
            
        # For multiple symbols, use the first one or 'overall'
        metric_key = 'overall'
        if 'overall' not in self.metrics and self.selected_symbols:
            metric_key = self.selected_symbols[0]
            
        if metric_key not in self.metrics:
            return
            
        metrics = self.metrics[metric_key]
        if not metrics:
            return
            
        # Format metrics for display
        current = f"Current: {metrics.get('current', 'N/A'):.2f}"
        trend = f"Trend: {metrics.get('trend', 'N/A')}"
        
        # Create annotation for sentiment plot
        self.fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            text=f"{current}<br>{trend}",
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.7)" if self.config.theme == "white" else "rgba(0, 0, 0, 0.7)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            row=1, col=1
        )
        
        # Add correlation annotation if available
        if self.config.show_correlation and metrics.get('price_correlation') is not None:
            corr_value = metrics['price_correlation']
            corr_text = f"Overall Correlation: {corr_value:.2f}"
            
            self.fig.add_annotation(
                xref="paper", yref="paper",
                x=0.01, y=0.99,
                text=corr_text,
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.7)" if self.config.theme == "white" else "rgba(0, 0, 0, 0.7)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=4,
                row=2, col=1
            )


def create_sentiment_view(
    sentiment_data: pd.DataFrame,
    price_data: Optional[pd.DataFrame] = None,
    selected_symbols: Optional[List[str]] = None,
    correlation_window: int = 20,
    show_distribution: bool = False,
    chart_height: int = 600,
    sentiment_lag: int = 1,
    display_mode: str = "combined"
) -> go.Figure:
    """
    Create an interactive sentiment visualization with optional price correlation.
    
    This function provides a simplified interface to the SentimentView class.
    
    Args:
        sentiment_data: DataFrame containing sentiment scores
        price_data: Optional DataFrame containing price data for correlation
        selected_symbols: List of symbols to display
        correlation_window: Window size for rolling correlation calculation
        show_distribution: Whether to include sentiment distribution histogram
        chart_height: Height of the chart in pixels
        sentiment_lag: Lag periods for sentiment in correlation analysis
        display_mode: Visualization mode for sources ("combined", "stacked", "separate")
        
    Returns:
        Plotly Figure object
    """
    # Create configuration
    config = SentimentViewConfig(
        chart_height=chart_height,
        correlation_window=correlation_window,
        sentiment_lag=sentiment_lag,
        show_distribution=show_distribution,
        display_mode=SentimentDisplayMode(display_mode)
    )
    
    # Create view instance
    view = SentimentView(
        sentiment_data=sentiment_data,
        price_data=price_data,
        selected_symbols=selected_symbols,
        config=config
    )
    
    # Create and return figure
    return view.create_figure()


def get_sentiment_summary(
    sentiment_data: pd.DataFrame,
    price_data: Optional[pd.DataFrame] = None,
    selected_symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate a summary of sentiment metrics for display.
    
    Args:
        sentiment_data: DataFrame containing sentiment scores
        price_data: Optional DataFrame containing price data
        selected_symbols: List of symbols to display
        
    Returns:
        Dictionary of sentiment metrics
    """
    # Create a temporary view to calculate metrics
    view = SentimentView(
        sentiment_data=sentiment_data,
        price_data=price_data,
        selected_symbols=selected_symbols
    )
    
    # Calculate metrics without creating a figure
    view._validate_data()
    view._calculate_metrics()
    
    return view.metrics