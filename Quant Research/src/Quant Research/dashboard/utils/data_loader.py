"""
Data loading utilities for the Quant Research dashboard.

This module provides functions for loading data from storage and
handling configuration files.
"""

import yaml
from datetime import date
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import pandas as pd
import streamlit as st
from pandas.core.frame import DataFrame

from quant_research.core.storage import get_connection, load_dataframe
from quant_research.dashboard.config.app_config import DEFAULT_CONFIG_PATH


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_cached_data(
    table_name: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    symbols: Optional[List[str]] = None,
    limit: int = 1000
) -> DataFrame:
    """
    Load data from storage with caching.
    
    Args:
        table_name: Name of the table to load
        start_date: Optional start date filter
        end_date: Optional end date filter
        symbols: Optional list of symbols to filter by
        limit: Maximum number of rows to return
        
    Returns:
        DataFrame with the requested data
        
    Raises:
        ValueError: If table_name is invalid
        ConnectionError: If unable to connect to storage
    """
    if not table_name:
        raise ValueError("Table name is required")
    
    try:
        conn = get_connection()
        return load_dataframe(conn, table_name, start_date, end_date, symbols, limit)
    except Exception as e:
        st.error(f"Error loading data from {table_name}: {str(e)}")
        # Re-raise with more context
        raise ConnectionError(f"Failed to load {table_name}") from e


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_configuration(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if isinstance(config_path, str):
        config_path = Path(config_path)
        
    if not config_path.exists():
        st.error(f"Configuration file not found: {config_path}")
        return {}
    
    try:    
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        st.error(f"Error parsing configuration file: {str(e)}")
        return {}


def get_available_symbols() -> List[str]:
    """
    Get list of available symbols from all data sources.
    
    Returns:
        List of available symbol strings
    """
    symbols = set()
    
    try:
        # Try to load symbols from price data
        price_data = load_cached_data("price_bars", limit=1)
        if 'symbol' in price_data.columns:
            symbols.update(price_data['symbol'].unique())
            
        # Add symbols from other sources
        signal_data = load_cached_data("signals", limit=1)
        if 'symbol' in signal_data.columns:
            symbols.update(signal_data['symbol'].unique())
    except Exception:
        # Fallback to default symbols if data loading fails
        return ["BTC/USD", "ETH/USD", "SOL/USD"]
    
    return sorted(list(symbols)) if symbols else ["BTC/USD", "ETH/USD", "SOL/USD"]


def get_data_freshness() -> Dict[str, Any]:
    """
    Check data freshness for each table.
    
    Returns:
        Dictionary with table names as keys and freshness info as values
    """
    tables = ["price_bars", "signals", "performance", "trades", "sentiment", "regimes"]
    freshness = {}
    
    for table in tables:
        try:
            data = load_cached_data(table, limit=1)
            if not data.empty and 'timestamp' in data.columns:
                latest = data['timestamp'].max()
                freshness[table] = {
                    "latest": latest,
                    "is_fresh": (pd.Timestamp.now() - latest).total_seconds() < 86400  # Within last day
                }
            else:
                freshness[table] = {"latest": None, "is_fresh": False}
        except Exception:
            freshness[table] = {"latest": None, "is_fresh": False}
    
    return freshness