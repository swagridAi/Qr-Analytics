"""
Visualization components for the Quant Research dashboard.

This module provides access to all visualization components and their
simplified function interfaces for easy use in the Streamlit application.
"""

# Import simplified component factory functions
from quant_research.dashboard.components.price_chart import create_price_chart
from quant_research.dashboard.components.sentiment_view import create_sentiment_view
from quant_research.dashboard.components.performance_view import create_performance_view
from quant_research.dashboard.components.portfolio_view import create_portfolio_view

# Import component classes for advanced usage
from quant_research.dashboard.components.price_chart import PriceChart, PriceChartConfig
from quant_research.dashboard.components.sentiment_view import SentimentView, SentimentViewConfig
from quant_research.dashboard.components.performance_view import PerformanceView, PerformanceViewConfig
from quant_research.dashboard.components.portfolio_view import PortfolioView, PortfolioViewConfig

# Import utility functions
from quant_research.dashboard.components.price_chart import get_technical_indicators
from quant_research.dashboard.components.sentiment_view import get_sentiment_summary
from quant_research.dashboard.components.performance_view import get_performance_metrics
from quant_research.dashboard.components.portfolio_view import get_portfolio_risk_metrics

# Version
__version__ = '1.0.0'


# Public API
__all__ = [
    # Factory functions
    'create_price_chart',
    'create_sentiment_view',
    'create_performance_view',
    'create_portfolio_view',
    
    # Component classes
    'PriceChart',
    'PriceChartConfig',
    'SentimentView',
    'SentimentViewConfig',
    'PerformanceView',
    'PerformanceViewConfig',
    'PortfolioView',
    'PortfolioViewConfig',
    
    # Utility functions
    'get_technical_indicators',
    'get_sentiment_summary',
    'get_performance_metrics',
    'get_portfolio_risk_metrics'
]