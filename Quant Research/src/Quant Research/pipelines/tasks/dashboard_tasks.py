"""
Dashboard tasks for pipeline workflows.

This module contains tasks for dashboard and reporting operations in the pipeline.
"""

from typing import Dict, Any, Optional, List, Union
import os
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

from quant_research.pipelines.core.task import dashboard_task
from quant_research.core.storage import load_dataframe
from quant_research.backtest.utils import calculate_metrics

@dashboard_task(
    name="generate_performance_report",
    description="Generate a performance report from backtest metrics"
)
def generate_performance_report(
    metrics: Dict[str, Any],
    output_path: str,
    title: Optional[str] = None,
    include_timestamp: bool = True,
    include_market_data: bool = False,
    market_data_path: Optional[str] = None
) -> str:
    """
    Generate a performance report from backtest metrics.
    
    Args:
        metrics: Backtest metrics
        output_path: Path to save the report
        title: Optional title for the report
        include_timestamp: Whether to include generation timestamp
        include_market_data: Whether to include market data summary
        market_data_path: Optional path to market data for context
        
    Returns:
        Path to the saved report
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize report structure
    report = {
        "title": title or "Performance Report",
        "metrics": metrics
    }
    
    # Add report generation timestamp
    if include_timestamp:
        report["generated_at"] = datetime.datetime.now().isoformat()
    
    # Add market data summary if requested
    if include_market_data and market_data_path:
        try:
            market_df = load_dataframe(market_data_path)
            if not market_df.empty:
                # Calculate basic market stats
                market_summary = {
                    "start_date": market_df['timestamp'].min().isoformat(),
                    "end_date": market_df['timestamp'].max().isoformat(),
                    "symbols": market_df['symbol'].nunique() if 'symbol' in market_df.columns else 1,
                    "data_points": len(market_df)
                }
                
                # Calculate market returns if possible
                if 'close' in market_df.columns:
                    try:
                        # Group by symbol if multiple symbols
                        if 'symbol' in market_df.columns and market_df['symbol'].nunique() > 1:
                            for symbol, data in market_df.groupby('symbol'):
                                data = data.sort_values('timestamp')
                                first_price = data['close'].iloc[0]
                                last_price = data['close'].iloc[-1]
                                market_return = (last_price / first_price) - 1
                                market_summary[f"{symbol}_return"] = market_return
                        else:
                            # Single symbol data
                            market_df = market_df.sort_values('timestamp')
                            first_price = market_df['close'].iloc[0]
                            last_price = market_df['close'].iloc[-1]
                            market_return = (last_price / first_price) - 1
                            market_summary["market_return"] = market_return
                    except Exception as e:
                        pass  # Skip if calculation fails
                
                report["market_summary"] = market_summary
        except Exception as e:
            # Skip market data if loading fails
            pass
    
    # Save report as JSON
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return output_path

@dashboard_task(
    name="plot_performance_metrics",
    description="Create performance metric visualizations"
)
def plot_performance_metrics(
    metrics: Dict[str, Any],
    output_path: str,
    figsize: tuple = (10, 8),
    include_detailed_metrics: bool = True,
    custom_title: Optional[str] = None
) -> str:
    """
    Create performance metric visualizations.
    
    Args:
        metrics: Backtest metrics
        output_path: Path to save the plot
        figsize: Figure size
        include_detailed_metrics: Whether to include additional metrics
        custom_title: Optional custom title for the plot
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract key metrics
    key_metrics = {
        "Total Return": metrics.get("total_return", 0) * 100,
        "Annualized Return": metrics.get("annualized_return", 0) * 100,
        "Sharpe Ratio": metrics.get("sharpe_ratio", 0),
        "Sortino Ratio": metrics.get("sortino_ratio", 0),
        "Max Drawdown": abs(metrics.get("max_drawdown", 0)) * 100,  # Absolute value for visualization
        "Win Rate": metrics.get("win_rate", 0) * 100
    }
    
    # Add additional metrics if requested
    if include_detailed_metrics:
        detailed_metrics = {
            "Volatility": metrics.get("volatility", 0) * 100,
            "Downside Deviation": metrics.get("downside_deviation", 0) * 100,
            "Avg Win": metrics.get("avg_win", 0) * 100,
            "Avg Loss": abs(metrics.get("avg_loss", 0)) * 100  # Absolute value
        }
        key_metrics.update(detailed_metrics)
    
    # Create visualization with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Key performance metrics
    metrics_plot = axes[0]
    
    # Sort metrics by type for better visualization
    return_metrics = {k: v for k, v in key_metrics.items() if "Return" in k or "Win" in k or "Loss" in k}
    risk_metrics = {k: v for k, v in key_metrics.items() if "Drawdown" in k or "Volatility" in k or "Deviation" in k}
    ratio_metrics = {k: v for k, v in key_metrics.items() if "Ratio" in k}
    
    # Define colors
    returns_color = 'green'
    risk_color = 'red'
    ratio_color = 'blue'
    
    # Plot returns
    bar_positions = range(len(return_metrics))
    metrics_plot.bar(bar_positions, return_metrics.values(), color=returns_color, alpha=0.7, label='Returns')
    
    # Add labels
    for i, (key, value) in enumerate(return_metrics.items()):
        metrics_plot.text(i, value + 0.5, f"{value:.2f}%", ha='center')
    
    # Plot risk metrics (with offset)
    offset = len(return_metrics)
    bar_positions = range(offset, offset + len(risk_metrics))
    metrics_plot.bar(bar_positions, risk_metrics.values(), color=risk_color, alpha=0.7, label='Risk')
    
    # Add labels
    for i, (key, value) in enumerate(risk_metrics.items()):
        metrics_plot.text(i + offset, value + 0.5, f"{value:.2f}%", ha='center')
    
    # Plot ratio metrics (with offset)
    offset += len(risk_metrics)
    bar_positions = range(offset, offset + len(ratio_metrics))
    metrics_plot.bar(bar_positions, ratio_metrics.values(), color=ratio_color, alpha=0.7, label='Ratios')
    
    # Add labels
    for i, (key, value) in enumerate(ratio_metrics.items()):
        metrics_plot.text(i + offset, value + 0.5, f"{value:.2f}", ha='center')
    
    # Set x-axis labels
    all_labels = list(return_metrics.keys()) + list(risk_metrics.keys()) + list(ratio_metrics.keys())
    metrics_plot.set_xticks(range(len(all_labels)))
    metrics_plot.set_xticklabels(all_labels, rotation=45, ha='right')
    
    # Add title and legend
    title = custom_title or "Performance Metrics Summary"
    metrics_plot.set_title(title)
    metrics_plot.legend()
    
    # Plot 2: Additional context if available
    context_plot = axes[1]
    
    # If equity curve data is available, plot it
    if 'equity_curve' in metrics:
        equity_data = metrics['equity_curve']
        timestamps = [datetime.datetime.fromisoformat(ts) for ts in equity_data['timestamps']]
        values = equity_data['values']
        
        context_plot.plot(timestamps, values, label='Equity Curve', color='blue')
        context_plot.set_title('Equity Curve')
        context_plot.set_xlabel('Date')
        context_plot.set_ylabel('Portfolio Value')
        
        # Format x-axis dates
        context_plot.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        context_plot.xaxis.set_major_locator(mdates.AutoDateLocator())
    elif 'drawdowns' in metrics:
        # Plot drawdowns if available
        drawdown_data = metrics['drawdowns']
        timestamps = [datetime.datetime.fromisoformat(ts) for ts in drawdown_data['timestamps']]
        values = [abs(v) * 100 for v in drawdown_data['values']]  # Convert to percentage
        
        context_plot.fill_between(timestamps, values, 0, color='red', alpha=0.3)
        context_plot.set_title('Drawdowns')
        context_plot.set_xlabel('Date')
        context_plot.set_ylabel('Drawdown (%)')
        
        # Format x-axis dates
        context_plot.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        context_plot.xaxis.set_major_locator(mdates.AutoDateLocator())
    else:
        # If no time series data, show summary statistics
        context_plot.axis('off')  # Hide axes
        
        summary_text = (
            f"Strategy: {metrics.get('strategy', 'Unknown')}\n"
            f"Period: {metrics.get('start_date', 'Unknown')} to {metrics.get('end_date', 'Unknown')}\n"
            f"Initial Capital: ${metrics.get('initial_value', 0):,.2f}\n"
            f"Final Value: ${metrics.get('final_value', 0):,.2f}\n"
            f"Days: {metrics.get('days', 0)}"
        )
        
        context_plot.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

@dashboard_task(
    name="generate_signals_dashboard",
    description="Generate a dashboard visualization of trading signals"
)
def generate_signals_dashboard(
    signals_path: str,
    prices_path: str,
    output_path: str,
    top_n_signals: int = 10,
    include_price_chart: bool = True
) -> str:
    """
    Generate a dashboard visualization of trading signals.
    
    Args:
        signals_path: Path to signals data
        prices_path: Path to price data for context
        output_path: Path to save the dashboard
        top_n_signals: Number of top signals to highlight
        include_price_chart: Whether to include price chart
        
    Returns:
        Path to the saved dashboard visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load data
    signals_df = load_dataframe(signals_path)
    prices_df = load_dataframe(prices_path)
    
    # Create figure with subplots
    if include_price_chart:
        fig, (price_ax, signal_ax) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    else:
        fig, signal_ax = plt.subplots(1, 1, figsize=(12, 6))
        price_ax = None
    
    # Plot price chart if requested
    if include_price_chart and not prices_df.empty and 'close' in prices_df.columns:
        # Group by symbol if multiple symbols
        if 'symbol' in prices_df.columns and prices_df['symbol'].nunique() > 1:
            for symbol, data in prices_df.groupby('symbol'):
                data = data.sort_values('timestamp')
                price_ax.plot(data['timestamp'], data['close'], label=symbol)
            price_ax.legend(loc='best')
        else:
            # Single symbol data
            prices_df = prices_df.sort_values('timestamp')
            price_ax.plot(prices_df['timestamp'], prices_df['close'], label='Price')
        
        price_ax.set_title('Price Chart')
        price_ax.set_ylabel('Price')
        price_ax.grid(True, alpha=0.3)
    
    # Plot signals
    if not signals_df.empty:
        # Group by signal type
        signal_types = signals_df['signal_type'].unique() if 'signal_type' in signals_df.columns else ['Unknown']
        
        for signal_type in signal_types:
            signal_data = signals_df[signals_df['signal_type'] == signal_type] if 'signal_type' in signals_df.columns else signals_df
            
            if not signal_data.empty:
                # Sort by timestamp
                signal_data = signal_data.sort_values('timestamp')
                
                # Get signal values
                signal_ax.plot(signal_data['timestamp'], signal_data['value'], 
                             label=signal_type, alpha=0.7, marker='o', linestyle='-')
        
        signal_ax.set_title('Trading Signals')
        signal_ax.set_xlabel('Date')
        signal_ax.set_ylabel('Signal Value')
        signal_ax.grid(True, alpha=0.3)
        signal_ax.legend(loc='best')
    
    # Format dates on x-axis
    for ax in [price_ax, signal_ax]:
        if ax is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Add title
    fig.suptitle('Trading Signals Dashboard', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path