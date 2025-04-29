# pipelines/tasks/provider_tasks.py
import asyncio
from typing import Dict, List, Any
import pandas as pd
from prefect import task

from quant_research.providers import ProviderFactory
from quant_research.core.models import PriceBar
from quant_research.core.storage import save_dataframe

@task
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
        if isinstance(data_points[0], PriceBar):
            df = pd.DataFrame([p.dict() for p in data_points])
        else:
            df = pd.DataFrame(data_points)
        
        # Save data
        save_dataframe(df, output_path)
        
        return output_path
    finally:
        # Always disconnect
        await provider.disconnect()