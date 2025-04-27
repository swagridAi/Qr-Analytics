#!/usr/bin/env python
"""
Main Streamlit application for the Quant Research Lab dashboard.
This serves as the entry point for the dashboard UI.
"""

import os
import sys
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import streamlit as st

# Add the project root to path to enable relative imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import components
from quant_research.dashboard.components import (
    create_price_chart,
    create_sentiment_view,
    create_performance_view
)

# Import utilities
from quant_research.dashboard.utils.data_loader import (
    load_cached_data, 
    load_configuration
)
from quant_research.dashboard.utils.metrics import (
    calculate_performance_metrics,
    calculate_sentiment_metrics
)
from quant_research.dashboard.config.app_config import (
    STRATEGIES,
    TIMEFRAMES,
    DEFAULT_CONFIG_PATH,
    DATA_PATH
)


def setup_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Quant Research Lab",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def setup_sidebar() -> Dict[str, Any]:
    """
    Set up the sidebar with navigation and control elements.
    
    Returns:
        Dictionary of selected parameters
    """
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50.png?text=Quant+Lab", use_column_width=True)
        st.subheader("Strategy Configuration")
        
        # Strategy selector
        selected_strategy = st.selectbox(
            "Select Strategy",
            options=STRATEGIES,
            index=0
        )
        
        # Timeframe selector
        selected_timeframe = st.selectbox(
            "Timeframe",
            options=TIMEFRAMES,
            index=TIMEFRAMES.index("1h")
        )
        
        # Date range selector
        today = datetime.now().date()
        start_date = st.date_input(
            "Start Date",
            value=today - timedelta(days=30)
        )
        end_date = st.date_input(
            "End Date",
            value=today
        )
        
        # Asset selector based on available data
        available_symbols = st.session_state.config.get('symbols', ["BTC/USD", "ETH/USD", "SOL/USD"])
        selected_symbols = st.multiselect(
            "Select Symbols",
            options=available_symbols,
            default=[available_symbols[0]]
        )
        
        # Strategy specific parameters
        strategy_params = get_strategy_parameters(selected_strategy)
        
        # Run backtest button
        if st.button("Run Backtest"):
            st.session_state.run_backtest = True
            st.info("Running backtest... This may take a moment.")
        
        # Data freshness indicator
        st.divider()
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"Last updated: {last_update}")
        
        # Additional utilities
        st.divider()
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()
            
        # Export options
        export_format = st.selectbox(
            "Export Format",
            options=["CSV", "Excel", "PDF Report"],
            index=0
        )
        if st.button("Export"):
            st.info(f"Exporting as {export_format}...")
            # Export logic would be implemented here
    
    return {
        "strategy": selected_strategy,
        "timeframe": selected_timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "symbols": selected_symbols,
        "params": strategy_params,
        "export_format": export_format
    }


def get_strategy_parameters(strategy: str) -> Dict[str, Any]:
    """
    Generate strategy-specific parameter inputs.
    
    Args:
        strategy: Selected strategy name
    
    Returns:
        Dictionary of strategy parameters
    """
    params = {}
    
    st.sidebar.subheader("Parameters")
    
    if strategy == "momentum":
        params["lookback_period"] = st.sidebar.slider("Lookback Period", 5, 200, 20)
        params["threshold"] = st.sidebar.slider("Signal Threshold", 0.0, 2.0, 1.0, 0.1)
    
    elif strategy == "mean_reversion":
        params["z_score_window"] = st.sidebar.slider("Z-Score Window", 5, 100, 20)
        params["entry_threshold"] = st.sidebar.slider("Entry Threshold", 1.0, 3.0, 2.0, 0.1)
        params["exit_threshold"] = st.sidebar.slider("Exit Threshold", 0.0, 1.0, 0.5, 0.1)
    
    elif strategy == "cross_exchange_arbitrage":
        params["min_spread"] = st.sidebar.slider("Min Spread (%)", 0.1, 5.0, 0.5, 0.1)
        params["exchanges"] = st.sidebar.multiselect(
            "Exchanges",
            options=["Binance", "Coinbase", "Kraken", "FTX"],
            default=["Binance", "Coinbase"]
        )
    
    elif strategy == "adaptive_regime":
        params["regime_threshold"] = st.sidebar.slider("Regime Threshold", 0.0, 1.0, 0.7, 0.05)
    
    return params


def load_data(params: Dict[str, Any]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, bool]]:
    """
    Load data based on selected parameters.
    
    Args:
        params: Dictionary of selected parameters
    
    Returns:
        Tuple of (data_dict, availability_flags)
    """
    data_dict = {}
    has_flags = {}
    
    try:
        # Load price data
        data_dict["price_data"] = load_cached_data(
            "price_bars", 
            start_date=params["start_date"],
            end_date=params["end_date"],
            symbols=params["symbols"]
        )
        
        # Load signals
        data_dict["signals_data"] = load_cached_data(
            "signals",
            start_date=params["start_date"],
            end_date=params["end_date"],
            symbols=params["symbols"]
        )
        
        # Load performance data
        data_dict["performance_data"] = load_cached_data(
            "performance",
            start_date=params["start_date"],
            end_date=params["end_date"],
            symbols=params["symbols"]
        )
        
        # Load trades data
        data_dict["trades_data"] = load_cached_data(
            "trades",
            start_date=params["start_date"],
            end_date=params["end_date"],
            symbols=params["symbols"]
        )
        
        # Load sentiment data if available
        try:
            data_dict["sentiment_data"] = load_cached_data(
                "sentiment",
                start_date=params["start_date"],
                end_date=params["end_date"],
                symbols=params["symbols"]
            )
            has_flags["has_sentiment"] = not data_dict["sentiment_data"].empty
        except Exception:
            data_dict["sentiment_data"] = pd.DataFrame()
            has_flags["has_sentiment"] = False
            
        # Load regime data if available
        try:
            data_dict["regime_data"] = load_cached_data(
                "regimes",
                start_date=params["start_date"],
                end_date=params["end_date"],
                symbols=params["symbols"]
            )
            has_flags["has_regimes"] = not data_dict["regime_data"].empty
        except Exception:
            data_dict["regime_data"] = pd.DataFrame()
            has_flags["has_regimes"] = False
        
        # Set main data availability flag
        has_flags["has_data"] = not data_dict["price_data"].empty
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        has_flags["has_data"] = False
        data_dict = {
            "price_data": pd.DataFrame(),
            "signals_data": pd.DataFrame(),
            "performance_data": pd.DataFrame(),
            "trades_data": pd.DataFrame(),
            "sentiment_data": pd.DataFrame(),
            "regime_data": pd.DataFrame()
        }
    
    return data_dict, has_flags


def render_overview_tab(data_dict: Dict[str, pd.DataFrame], 
                        has_flags: Dict[str, bool], 
                        params: Dict[str, Any]) -> None:
    """Render the Overview tab with summary metrics."""
    st.subheader(f"Strategy: {params['strategy'].replace('_', ' ').title()}")
    
    if has_flags["has_data"]:
        # Summary metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        # Get performance data
        perf_data = data_dict["performance_data"]
        trades_data = data_dict["trades_data"]
        
        # Calculate metrics
        if not perf_data.empty:
            metrics = calculate_performance_metrics(perf_data, trades_data)
            
            with col1:
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 'N/A')}")
            
            with col2:
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 'N/A')}%")
            
            with col3:
                st.metric("Win Rate", f"{metrics.get('win_rate', 'N/A')}%")
            
            with col4:
                st.metric("Total Return", f"{metrics.get('total_return', 'N/A')}%")
        
        # Recent signals alert box
        if not data_dict["signals_data"].empty:
            st.subheader("Recent Signals")
            recent_signals = data_dict["signals_data"].sort_values('timestamp', ascending=False).head(5)
            st.dataframe(recent_signals, hide_index=True)
        
        # Market regime indicator
        if has_flags.get("has_regimes", False):
            render_regime_visualization(data_dict["regime_data"])
    else:
        st.warning("No data available for the selected parameters. Please adjust your filters or check data sources.")


def render_regime_visualization(regime_data: pd.DataFrame) -> None:
    """Render the market regime visualization."""
    import plotly.graph_objects as go
    
    st.subheader("Market Regime")
    regime_fig = go.Figure()
    
    # Get regime columns
    regime_cols = [col for col in regime_data.columns if col.startswith('regime_')]
    
    if regime_cols and 'timestamp' in regime_data.columns:
        # Create a stacked area chart for regime probabilities
        for col in regime_cols:
            regime_name = col.replace('regime_', '').capitalize()
            regime_fig.add_trace(
                go.Scatter(
                    x=regime_data['timestamp'],
                    y=regime_data[col],
                    name=regime_name,
                    mode='lines',
                    stackgroup='one',
                    groupnorm='percent'
                )
            )
        
        regime_fig.update_layout(
            title="Regime Probabilities Over Time",
            xaxis_title="Date",
            yaxis_title="Probability",
            hovermode="x unified"
        )
        
        st.plotly_chart(regime_fig, use_container_width=True)


def render_price_tab(data_dict: Dict[str, pd.DataFrame], 
                    has_flags: Dict[str, bool], 
                    params: Dict[str, Any]) -> None:
    """Render the Price Analysis tab."""
    st.subheader("Price Analysis")
    
    if has_flags["has_data"]:
        # Technical indicator options
        with st.expander("Chart Options"):
            indicator_col1, indicator_col2 = st.columns(2)
            with indicator_col1:
                show_ma = st.checkbox("Moving Averages", value=True)
                ma_periods = st.multiselect(
                    "MA Periods",
                    options=[9, 20, 50, 200],
                    default=[20, 50]
                ) if show_ma else None
            
            with indicator_col2:
                show_volume = st.checkbox("Volume", value=True)
                show_signals = st.checkbox("Show Signals", value=True)
        
        # Create price chart
        price_chart = create_price_chart(
            price_data=data_dict["price_data"],
            signals_data=data_dict["signals_data"] if show_signals else None,
            regime_data=data_dict["regime_data"] if has_flags.get("has_regimes", False) else None,
            selected_symbols=params["symbols"],
            timeframe=params["timeframe"],
            show_volume=show_volume,
            show_ma=show_ma,
            ma_periods=ma_periods
        )
        
        st.plotly_chart(price_chart, use_container_width=True)
    else:
        st.warning("No price data available for the selected parameters.")


def render_signals_tab(data_dict: Dict[str, pd.DataFrame], 
                      has_flags: Dict[str, bool], 
                      params: Dict[str, Any]) -> None:
    """Render the Signals & Alpha tab."""
    if has_flags["has_data"] and not data_dict["signals_data"].empty:
        st.subheader("Signal Analysis")
        
        # Signal visualization and analysis
        if 'strength' in data_dict["signals_data"].columns:
            import plotly.graph_objects as go
            
            signal_fig = go.Figure()
            signal_fig.add_trace(
                go.Scatter(
                    x=data_dict["signals_data"]['timestamp'],
                    y=data_dict["signals_data"]['strength'],
                    mode='lines+markers',
                    name='Signal Strength'
                )
            )
            signal_fig.update_layout(
                title="Signal Strength Over Time",
                xaxis_title="Date",
                yaxis_title="Signal Strength"
            )
            st.plotly_chart(signal_fig, use_container_width=True)
        
        # Signal distribution
        if len(data_dict["signals_data"]) > 0:
            st.subheader("Signal Distribution")
            signal_dist_col1, signal_dist_col2 = st.columns(2)
            
            with signal_dist_col1:
                # Signal type distribution
                if 'type' in data_dict["signals_data"].columns:
                    signal_types = data_dict["signals_data"]['type'].value_counts()
                    st.bar_chart(signal_types)
            
            with signal_dist_col2:
                # Signal strength histogram
                if 'strength' in data_dict["signals_data"].columns:
                    hist_data = data_dict["signals_data"]['strength']
                    st.bar_chart(hist_data.value_counts())
        
        # Sentiment analysis results
        if has_flags.get("has_sentiment", False):
            st.subheader("Sentiment Analysis")
            sentiment_view = create_sentiment_view(
                sentiment_data=data_dict["sentiment_data"],
                price_data=data_dict["price_data"],
                selected_symbols=params["symbols"]
            )
            st.plotly_chart(sentiment_view, use_container_width=True)
    else:
        st.warning("No signal data available for the selected parameters.")


def render_performance_tab(data_dict: Dict[str, pd.DataFrame], 
                          has_flags: Dict[str, bool], 
                          params: Dict[str, Any]) -> None:
    """Render the Performance Analysis tab."""
    if has_flags["has_data"] and not data_dict["performance_data"].empty:
        st.subheader("Performance Analysis")
        
        # Performance view options
        with st.expander("Performance Chart Options"):
            show_drawdown = st.checkbox("Show Drawdown", value=True)
            show_underwater = st.checkbox("Show Underwater Equity", value=True)
            
            period_options = {"All Time": "all", "1 Month": "1m", "3 Months": "3m", 
                             "6 Months": "6m", "1 Year": "1y"}
            selected_period = st.selectbox(
                "Time Period",
                options=list(period_options.keys()),
                index=0
            )
            period = period_options[selected_period]
        
        # Create performance view
        performance_view = create_performance_view(
            performance_data=data_dict["performance_data"],
            trades_data=data_dict["trades_data"],
            selected_symbols=params["symbols"],
            benchmark_data=None,  # Could add benchmark data here
            show_drawdown=show_drawdown,
            show_underwater=show_underwater,
            period=period
        )
        
        st.plotly_chart(performance_view, use_container_width=True)
        
        # Performance metrics table
        st.subheader("Performance Metrics")
        metrics = calculate_performance_metrics(
            data_dict["performance_data"], 
            data_dict["trades_data"]
        )
        
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        
        st.table(metrics_df)
        
        # Trade blotter
        if not data_dict["trades_data"].empty:
            st.subheader("Trade Blotter")
            st.dataframe(
                data_dict["trades_data"].sort_values('timestamp', ascending=False), 
                hide_index=True
            )
    else:
        st.warning("No performance data available for the selected parameters.")


def render_portfolio_tab(data_dict: Dict[str, pd.DataFrame], 
                        has_flags: Dict[str, bool], 
                        params: Dict[str, Any]) -> None:
    """Render the Portfolio Analysis tab."""
    if has_flags["has_data"] and not data_dict["trades_data"].empty:
        st.subheader("Portfolio Analysis")
        
        # Calculate current positions
        current_positions = calculate_current_positions(data_dict["trades_data"])
        
        if not current_positions.empty:
            st.subheader("Current Positions")
            st.dataframe(current_positions, hide_index=True)
        
            # Allocation visualization
            if 'position_size' in current_positions.columns:
                st.subheader("Portfolio Allocation")
                allocation = current_positions.groupby('symbol')['position_size'].sum()
                
                # Use pie chart
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Pie(
                    labels=allocation.index,
                    values=allocation.values,
                    hole=.3
                )])
                fig.update_layout(title="Portfolio Allocation by Symbol")
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk exposure metrics
        st.subheader("Risk Exposure")
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            # Value at Risk calculation
            var_value = calculate_var(data_dict["performance_data"])
            st.metric("1-Day Value at Risk (95%)", var_value)
        
        with risk_col2:
            # Leverage calculation
            leverage = calculate_leverage(current_positions)
            st.metric("Current Leverage", leverage)
        
        # Open arbitrage spreads if applicable
        if params["strategy"] == "cross_exchange_arbitrage":
            st.subheader("Open Arbitrage Spreads")
            # Would display current open arb opportunities
            st.info("No open arbitrage opportunities at this time.")
    else:
        st.warning("No portfolio data available for the selected parameters.")


def calculate_current_positions(trades_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate current positions from trade data."""
    if trades_data.empty:
        return pd.DataFrame()
    
    # Group by symbol and take the latest trade for each
    try:
        current_positions = trades_data.groupby('symbol').apply(
            lambda x: x.iloc[-1] if len(x) > 0 else None
        ).reset_index(drop=True)
        return current_positions
    except Exception:
        return pd.DataFrame()


def calculate_var(performance_data: pd.DataFrame, confidence: float = 0.95) -> str:
    """Calculate Value at Risk."""
    if performance_data.empty or 'returns' not in performance_data.columns:
        return "N/A"
    
    try:
        import numpy as np
        returns = performance_data['returns'].dropna()
        
        if len(returns) > 30:  # Need sufficient data for VaR
            # Historical VaR
            var = np.percentile(returns, 100 * (1 - confidence))
            return f"{var:.2%}"
        return "Insufficient data"
    except Exception:
        return "N/A"


def calculate_leverage(positions: pd.DataFrame) -> str:
    """Calculate current leverage from positions."""
    if positions.empty or 'notional_value' not in positions.columns or 'margin' not in positions.columns:
        return "N/A"
    
    try:
        total_notional = positions['notional_value'].sum()
        total_margin = positions['margin'].sum()
        
        if total_margin > 0:
            return f"{total_notional / total_margin:.2f}x"
        return "N/A"
    except Exception:
        return "N/A"


def main() -> None:
    """Main application entry point."""
    setup_page()
    
    # Load application state
    if 'config' not in st.session_state:
        st.session_state.config = load_configuration()
    
    # Application header
    st.title("Quant Research Lab Dashboard")
    
    # Setup sidebar and get parameters
    params = setup_sidebar()
    
    # Load data based on parameters
    data_dict, has_flags = load_data(params)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", 
        "Price Analysis", 
        "Signals & Alpha", 
        "Performance", 
        "Portfolio"
    ])
    
    # Render each tab
    with tab1:
        render_overview_tab(data_dict, has_flags, params)
    
    with tab2:
        render_price_tab(data_dict, has_flags, params)
    
    with tab3:
        render_signals_tab(data_dict, has_flags, params)
    
    with tab4:
        render_performance_tab(data_dict, has_flags, params)
    
    with tab5:
        render_portfolio_tab(data_dict, has_flags, params)


if __name__ == "__main__":
    main()