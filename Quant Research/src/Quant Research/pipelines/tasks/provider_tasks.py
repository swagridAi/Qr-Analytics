"""
Provider tasks for pipeline workflows.

This module contains tasks for data provider operations in the pipeline.
"""

from typing import Dict, Any
import pandas as pd

from quant_research.pipelines.core.task import provider_task
from quant_research.providers import ProviderFactory
from quant_research.core.models import PriceBar
from quant_research.core.storage import save_dataframe

@provider_task(
    name="fetch_market_data",
    description="Fetch market data from a provider and save to storage",
    retries=3,
    retry_delay_seconds=60
)
async def fetch_market_data(
    provider_id: str,
    config: Dict[str, Any],
    output_path: str,
    **fetch_params
) -> str:
    """
    Fetch market data from a provider and save to storage.
    
    Args:
        provider_id: ID of the registered provider
        config: Provider configuration
        output_path: Path to save the data
        **fetch_params: Parameters for the fetch_data method
        
    Returns:
        Path to the saved data file
    """
    # Create provider
    provider = ProviderFactory.create(provider_id, config)
    
    try:
        # Connect to provider
        await provider.connect()
        
        # Fetch data
        data_points = []
        async for data_point in provider.fetch_data(**fetch_params):
            data_points.append(data_point)
        
        # Convert to DataFrame
        if len(data_points) > 0:
            if isinstance(data_points[0], PriceBar):
                df = pd.DataFrame([p.dict() for p in data_points])
            else:
                df = pd.DataFrame(data_points)
            
            # Save data
            save_dataframe(df, output_path)
        else:
            # Save empty DataFrame with appropriate schema
            empty_df = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'
            ])
            save_dataframe(empty_df, output_path)
        
        return output_path
    finally:
        # Always disconnect
        await provider.disconnect()