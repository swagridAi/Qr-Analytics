"""
Statistical Visualization Module

This module provides functions for creating statistical visualizations such as
distribution plots, correlation matrices, quantile-quantile plots, box plots,
scatter matrices, and heatmaps.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

# Local imports
from .utils import add_legend

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.visualization.statistical")

# Default color palette
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

#------------------------------------------------------------------------
# Statistical Visualizations
#------------------------------------------------------------------------

def plot_distribution(
    data: Union[pd.Series, np.ndarray],
    bins: int = 50,
    kde: bool = True,
    normal_overlay: bool = False,
    figsize: Tuple[float, float] = (10, 6),
    color: str = '#1f77b4',
    title: Optional[str] = None,
    stats_box: bool = True,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot the distribution of data with optional overlays.
    
    Args:
        data: Data for distribution analysis
        bins: Number of histogram bins
        kde: Whether to include KDE overlay
        normal_overlay: Whether to overlay normal distribution
        figsize: Figure size (width, height) in inches
        color: Color for the histogram
        title: Plot title
        stats_box: Whether to include statistics box
        ax: Existing axes to plot on
        **kwargs: Additional arguments for histogram
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert to Series if numpy array
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # Remove NaN values
    clean_data = data.dropna()
    
    # Plot histogram
    ax.hist(
        clean_data,
        bins=bins,
        alpha=0.6,
        color=color,
        density=True,
        label='Histogram',
        **kwargs
    )
    
    # Add KDE overlay if requested
    if kde:
        sns.kdeplot(
            clean_data,
            ax=ax,
            color='navy',
            linewidth=2,
            label='KDE'
        )
    
    # Add normal distribution overlay if requested
    if normal_overlay:
        mean = clean_data.mean()
        std = clean_data.std()
        x = np.linspace(
            max(clean_data.min(), mean - 4*std),
            min(clean_data.max(), mean + 4*std),
            100
        )
        y = stats.norm.pdf(x, mean, std)
        ax.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
    
    # Add statistics box if requested
    if stats_box:
        mean = clean_data.mean()
        median = clean_data.median()
        std = clean_data.std()
        skew = stats.skew(clean_data)
        kurt = stats.kurtosis(clean_data)
        
        # Create stats text
        stats_text = (
            f"Mean: {mean:.4f}\n"
            f"Median: {median:.4f}\n"
            f"Std Dev: {std:.4f}\n"
            f"Skewness: {skew:.4f}\n"
            f"Kurtosis: {kurt:.4f}"
        )
        
        # Add text box
        ax.text(
            0.97, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox={'boxstyle': 'round', 'alpha': 0.8, 'facecolor': 'white'}
        )
    
    # Set labels and title
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Distribution')
    
    # Add legend
    legend_elements = [mpatches.Patch(color=color, alpha=0.6, label='Histogram')]
    if kde:
        legend_elements.append(Line2D([0], [0], color='navy', linewidth=2, label='KDE'))
    if normal_overlay:
        legend_elements.append(Line2D([0], [0], color='red', linewidth=2, label='Normal'))
    
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    return fig, ax


def plot_qq(
    data: Union[pd.Series, np.ndarray],
    dist: str = 'norm',
    figsize: Tuple[float, float] = (8, 8),
    color: str = '#1f77b4',
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create a Q-Q plot to compare data to a theoretical distribution.
    
    Args:
        data: Data for Q-Q analysis
        dist: Theoretical distribution to compare against
        figsize: Figure size (width, height) in inches
        color: Color for the points
        title: Plot title
        ax: Existing axes to plot on
        **kwargs: Additional arguments for scatter plot
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert to numpy array if Series
    if isinstance(data, pd.Series):
        data = data.dropna().values
    elif isinstance(data, np.ndarray):
        data = data[~np.isnan(data)]
    
    # Create Q-Q plot
    osm, osr = stats.probplot(data, dist=dist, fit=True, plot=ax)
    
    # Get fit parameters
    slope, intercept, r_value = osr
    
    # Customize plot appearance
    # Replace default points with our own
    ax.clear()
    
    # Recreate the plot
    x = osm[0]
    y = osm[1]
    
    # Plot points
    ax.scatter(x, y, color=color, alpha=0.7, **kwargs)
    
    # Plot reference line
    ax.plot([min(x), max(x)], [slope * min(x) + intercept, slope * max(x) + intercept], 
           'r-', linewidth=2, label=f'R² = {r_value**2:.4f}')
    
    # Set labels and title
    dist_name = dist.capitalize() if dist in ['norm', 'exp'] else dist
    ax.set_xlabel(f'Theoretical {dist_name} Quantiles')
    ax.set_ylabel('Sample Quantiles')
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'{dist_name} Q-Q Plot')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    
    return fig, ax


def plot_correlation_matrix(
    data: pd.DataFrame,
    method: str = 'pearson',
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'coolwarm',
    annot: bool = True,
    mask_upper: bool = False,
    title: Optional[str] = None,
    vmin: Optional[float] = -1.0,
    vmax: Optional[float] = 1.0,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create a correlation matrix heatmap.
    
    Args:
        data: DataFrame with data to correlate
        method: Correlation method ('pearson', 'spearman', 'kendall')
        figsize: Figure size (width, height) in inches
        cmap: Colormap for heatmap
        annot: Whether to annotate cells with correlation values
        mask_upper: Whether to mask the upper triangle
        title: Plot title
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        **kwargs: Additional arguments for heatmap
        
    Returns:
        Tuple of (figure, axes)
    """
    # Calculate correlation matrix
    corr = data.corr(method=method)
    
    # Create mask for upper triangle if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        annot=annot,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax,
        **kwargs
    )
    
    # Set title
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'{method.capitalize()} Correlation Matrix')
    
    plt.tight_layout()
    
    return fig, ax


def plot_scatter_matrix(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    hist_kwds: Optional[Dict[str, Any]] = None,
    density_kwds: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Tuple[Figure, np.ndarray]:
    """
    Create a scatter matrix (pairs plot) for data exploration.
    
    Args:
        data: DataFrame with data to plot
        columns: Columns to include (None for all)
        figsize: Figure size (width, height) in inches
        hist_kwds: Keywords for histogram plots
        density_kwds: Keywords for density plots
        **kwargs: Additional arguments for scatter plots
        
    Returns:
        Tuple of (figure, array of axes)
    """
    # Select columns to include
    if columns is not None:
        plot_data = data[columns]
    else:
        plot_data = data
    
    # Determine figure size if not provided
    if figsize is None:
        n = len(plot_data.columns)
        figsize = (2 * n, 2 * n)
    
    # Set up default hist_kwds if not provided
    if hist_kwds is None:
        hist_kwds = {'bins': 20, 'alpha': 0.6}
    
    # Set up default density_kwds if not provided
    if density_kwds is None:
        density_kwds = {'alpha': 0.6}
    
    # Create scatter matrix
    axes = pd.plotting.scatter_matrix(
        plot_data,
        figsize=figsize,
        hist_kwds=hist_kwds,
        density_kwds=density_kwds,
        **kwargs
    )
    
    # Access figure from axes
    fig = plt.gcf()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, axes


def plot_box(
    data: Union[pd.DataFrame, pd.Series],
    columns: Optional[List[str]] = None,
    by: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    vert: bool = True,
    notch: bool = False,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create a box plot for distribution comparison.
    
    Args:
        data: Data to plot
        columns: Columns to include (if DataFrame)
        by: Column to group by
        figsize: Figure size (width, height) in inches
        vert: Whether to create vertical box plots
        notch: Whether to add notches
        title: Plot title
        ax: Existing axes to plot on
        **kwargs: Additional arguments for boxplot
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert Series to DataFrame if necessary
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    
    # Determine columns to plot
    if columns is None:
        if data.shape[1] <= 10:  # Plot all columns if 10 or fewer
            columns = data.columns.tolist()
        else:
            columns = data.columns[:10].tolist()  # Plot first 10 columns if more
            logger.warning(f"DataFrame has {data.shape[1]} columns, plotting only the first 10")
    
    # Prepare data for plotting
    if by is not None:
        # Grouped box plot
        plot_data = []
        labels = []
        
        for col in columns:
            grouped = data.groupby(by)[col]
            for group_name, group_data in grouped:
                plot_data.append(group_data.values)
                labels.append(f"{col} - {group_name}")
    else:
        # Simple box plot
        plot_data = [data[col].dropna() for col in columns]
        labels = columns
    
    # Create box plot
    ax.boxplot(
        plot_data,
        labels=labels,
        vert=vert,
        notch=notch,
        patch_artist=True,
        **kwargs
    )
    
    # Set labels and title
    if vert:
        ax.set_xlabel('Variable')
        ax.set_ylabel('Value')
    else:
        ax.set_xlabel('Value')
        ax.set_ylabel('Variable')
    
    if title is not None:
        ax.set_title(title)
    else:
        if by is not None:
            ax.set_title(f'Box Plot Grouped by {by}')
        else:
            ax.set_title('Box Plot')
    
    # Rotate x-axis labels for better readability if needed
    if len(labels) > 4 or any(len(str(label)) > 10 for label in labels):
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig, ax


def plot_heatmap(
    data: pd.DataFrame,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'viridis',
    annot: bool = True,
    cbar: bool = True,
    title: Optional[str] = None,
    transpose: bool = False,
    fmt: str = '.2f',
    linewidths: float = 0.5,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create a heatmap to visualize matrix data.
    
    Args:
        data: DataFrame with matrix data
        figsize: Figure size (width, height) in inches
        cmap: Colormap for heatmap
        annot: Whether to annotate cells with values
        cbar: Whether to add a colorbar
        title: Plot title
        transpose: Whether to transpose the data
        fmt: String formatting code for annotations
        linewidths: Width of lines between cells
        ax: Existing axes to plot on
        **kwargs: Additional arguments for heatmap
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Transpose data if requested
    if transpose:
        plot_data = data.T
    else:
        plot_data = data
    
    # Create heatmap
    sns.heatmap(
        plot_data,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        linewidths=linewidths,
        cbar=cbar,
        ax=ax,
        **kwargs
    )
    
    # Set title
    if title is not None:
        ax.set_title(title)
    
    # Rotate x-axis labels for better readability if needed
    if plot_data.shape[1] > 4 or any(len(str(col)) > 10 for col in plot_data.columns):
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig, ax