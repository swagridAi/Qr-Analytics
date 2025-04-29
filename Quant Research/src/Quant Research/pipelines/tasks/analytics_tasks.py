"""
Analytics tasks for pipeline workflows.

This module contains tasks for analytics and signal generation in the pipeline.
"""

from typing import Dict, Any, List, Optional
import os
import pandas as pd
import logging

from quant_research.pipelines.core.task import analytics_task
from quant_research.core.storage import load_dataframe, save_dataframe
from quant_research.analytics.common.base import SignalGenerator, SignalPipeline, SignalGeneratorRegistry

logger = logging.getLogger(__name__)

@analytics_task(
    name="generate_signals",
    description="Generate trading signals from market data using analytics modules",
    retries=2,
    retry_delay_seconds=30
)
async def generate_signals(
    data_path: str,
    analytics_config: List[Dict[str, Any]],
    output_path: str,
    combine_signals: bool = True
) -> str:
    """
    Generate trading signals from market data using analytics modules.
    
    Args:
        data_path: Path to market data
        analytics_config: List of analytics module configurations
        output_path: Path to save generated signals
        combine_signals: Whether to combine signals from different modules
        
    Returns:
        Path to the saved signals file
    """
    # Load market data
    data_df = load_dataframe(data_path)
    
    if data_df.empty:
        logger.warning("Input data is empty, no signals will be generated")
        # Create empty signals DataFrame with appropriate columns
        signals_df = pd.DataFrame(columns=[
            'timestamp', 'symbol', 'signal_type', 'value', 'generator'
        ])
        save_dataframe(signals_df, output_path)
        return output_path
    
    # Create directory for output if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize signal generators
    generators = []
    for module_config in analytics_config:
        try:
            name = module_config["name"]
            params = module_config.get("params", {})
            
            # Create generator instance
            generator = SignalGeneratorRegistry.create(name, **params)
            generators.append(generator)
            
        except Exception as e:
            logger.error(f"Error creating signal generator '{module_config.get('name', 'unknown')}': {e}")
            # Continue with other generators
    
    if not generators:
        logger.warning("No valid signal generators configured")
        signals_df = pd.DataFrame(columns=[
            'timestamp', 'symbol', 'signal_type', 'value', 'generator'
        ])
        save_dataframe(signals_df, output_path)
        return output_path
    
    # Create pipeline with all generators
    pipeline = SignalPipeline(generators)
    
    # Run pipeline
    results = pipeline.run(data_df)
    
    # Process results
    if combine_signals and results:
        # Combine all signals into a single DataFrame
        signals_list = []
        for generator_name, signals_df in results.items():
            # Add generator name if not already present
            if 'generator' not in signals_df.columns:
                signals_df['generator'] = generator_name
            signals_list.append(signals_df)
        
        # Concatenate all signals
        if signals_list:
            signals_df = pd.concat(signals_list, ignore_index=True)
        else:
            signals_df = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'signal_type', 'value', 'generator'
            ])
    else:
        # Just use the first generator's results
        signals_df = next(iter(results.values())) if results else pd.DataFrame(columns=[
            'timestamp', 'symbol', 'signal_type', 'value', 'generator'
        ])
    
    # Save signals
    save_dataframe(signals_df, output_path)
    
    return output_path

@analytics_task(
    name="filter_signals",
    description="Filter generated signals based on criteria"
)
def filter_signals(
    signals_path: str,
    filter_criteria: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """
    Filter generated signals based on specified criteria.
    
    Args:
        signals_path: Path to signals data
        filter_criteria: Dictionary of filter criteria
        output_path: Optional path to save filtered signals
        
    Returns:
        Path to the filtered signals file
    """
    # Load signals data
    signals_df = load_dataframe(signals_path)
    
    if signals_df.empty:
        logger.warning("Input signals data is empty")
        
        if output_path:
            save_dataframe(signals_df, output_path)
            return output_path
        
        return signals_path
    
    # Apply filters
    filtered_df = signals_df.copy()
    
    for column, criteria in filter_criteria.items():
        if column not in filtered_df.columns:
            logger.warning(f"Filter column '{column}' not found in signals data")
            continue
            
        if isinstance(criteria, (list, tuple)):
            # Include only values in list
            filtered_df = filtered_df[filtered_df[column].isin(criteria)]
        elif isinstance(criteria, dict):
            # Handle operators
            for op, value in criteria.items():
                if op == "gt":
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif op == "lt":
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif op == "gte":
                    filtered_df = filtered_df[filtered_df[column] >= value]
                elif op == "lte":
                    filtered_df = filtered_df[filtered_df[column] <= value]
                elif op == "eq":
                    filtered_df = filtered_df[filtered_df[column] == value]
                elif op == "neq":
                    filtered_df = filtered_df[filtered_df[column] != value]
        else:
            # Simple equality
            filtered_df = filtered_df[filtered_df[column] == criteria]
    
    # Save filtered signals if output path provided
    if output_path:
        save_dataframe(filtered_df, output_path)
        return output_path
    
    # Otherwise, overwrite original file
    save_dataframe(filtered_df, signals_path)
    return signals_path

@analytics_task(
    name="combine_signals",
    description="Combine signals from multiple sources"
)
def combine_signals(
    signals_paths: List[str],
    output_path: str,
    weights: Optional[Dict[str, float]] = None
) -> str:
    """
    Combine signals from multiple sources, optionally with weights.
    
    Args:
        signals_paths: List of paths to signal files
        output_path: Path to save combined signals
        weights: Optional dictionary mapping generator names to weights
        
    Returns:
        Path to the combined signals file
    """
    # Load and combine all signal files
    combined_signals = []
    
    for path in signals_paths:
        try:
            signals_df = load_dataframe(path)
            if not signals_df.empty:
                # Add source path as metadata if not present
                if 'source_path' not in signals_df.columns:
                    signals_df['source_path'] = path
                
                # Apply weights if provided
                if weights and 'generator' in signals_df.columns:
                    def apply_weight(row):
                        generator = row['generator']
                        if generator in weights:
                            row['value'] = row['value'] * weights[generator]
                        return row
                    
                    signals_df = signals_df.apply(apply_weight, axis=1)
                
                combined_signals.append(signals_df)
        except Exception as e:
            logger.error(f"Error loading signals from {path}: {e}")
    
    if not combined_signals:
        logger.warning("No valid signals to combine")
        combined_df = pd.DataFrame(columns=[
            'timestamp', 'symbol', 'signal_type', 'value', 'generator'
        ])
    else:
        # Concatenate all signals
        combined_df = pd.concat(combined_signals, ignore_index=True)
    
    # Save combined signals
    save_dataframe(combined_df, output_path)
    
    return output_path