# pipelines/tasks/analytics_tasks.py
from typing import Dict, List, Any, Optional
import pandas as pd
from prefect import task

from quant_research.analytics.common.base import SignalPipeline
from quant_research.analytics.common.base.signal_registry import SignalGeneratorRegistry
from quant_research.core.storage import load_dataframe, save_dataframe

@task
def generate_signals(
    data_path: str,
    analytics_config: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    Generate signals using the analytics module.
    
    Args:
        data_path: Path to input data
        analytics_config: List of analytics configurations
        output_path: Path to save signals
        
    Returns:
        Path to the saved signals
    """
    # Load data
    df = load_dataframe(data_path)
    
    # Create generators
    generators = []
    for config in analytics_config:
        generator_name = config.pop("name")
        generator = SignalGeneratorRegistry.create(generator_name, **config)
        generators.append(generator)
    
    # Create pipeline
    pipeline = SignalPipeline(generators)
    
    # Run pipeline
    results = pipeline.run(df)
    
    # Combine results
    all_signals = []
    for generator_name, signals_df in results.items():
        if not signals_df.empty:
            all_signals.append(signals_df)
    
    # Save signals
    if all_signals:
        combined_signals = pd.concat(all_signals, ignore_index=True)
        save_dataframe(combined_signals, output_path)
    else:
        # Save empty DataFrame if no signals
        save_dataframe(pd.DataFrame(), output_path)
    
    return output_path