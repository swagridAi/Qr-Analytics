"""
Dashboard tasks for pipeline workflows.

This module contains tasks for dashboard and reporting operations in the pipeline.
"""

from typing import Dict, Any, Optional
import os
import json
from prefect import task
import pandas as pd
import matplotlib.pyplot as plt

from quant_research.core.storage import load_dataframe
from quant_research.backtest.utils import calculate_metrics

@task
def generate_performance_report(
    metrics: Dict[str, Any],
    output_path: str,
    title: Optional[str] = None
) -> str:
    """
    Generate a performance report from backtest metrics.
    
    Args:
        metrics: Backtest metrics
        output_path: Path to save the report
        title: Optional title for the report
        
    Returns:
        Path to the saved report
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add report generation timestamp
    report = {
        "title": title or "Performance Report",
        "generated_at": pd.Timestamp.now().isoformat(),
        "metrics": metrics
    }
    
    # Save report as JSON
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return output_path

@task
def plot_performance_metrics(
    metrics: Dict[str, Any],
    output_path: str,
    figsize=(10, 8)
) -> str:
    """
    Create performance metric visualizations.
    
    Args:
        metrics: Backtest metrics
        output_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create summary visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract key metrics
    key_metrics = {
        "Total Return": metrics.get("total_return", 0) * 100,
        "Annualized Return": metrics.get("annualized_return", 0) * 100,
        "Sharpe Ratio": metrics.get("sharpe_ratio", 0),
        "Max Drawdown": metrics.get("max_drawdown", 0) * 100,
        "Win Rate": metrics.get("win_rate", 0) * 100
    }
    
    # Create bar chart
    ax.bar(key_metrics.keys(), key_metrics.values())
    ax.set_title("Backtest Performance Metrics")
    ax.set_ylabel("Value")
    
    # Add value labels
    for i, (key, value) in enumerate(key_metrics.items()):
        unit = "%" if key in ["Total Return", "Annualized Return", "Max Drawdown", "Win Rate"] else ""
        ax.text(i, value + 0.5, f"{value:.2f}{unit}", ha='center')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    return output_path