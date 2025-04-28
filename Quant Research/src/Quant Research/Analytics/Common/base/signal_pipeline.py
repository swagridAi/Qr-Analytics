"""
Signal Pipeline

This module provides a pipeline for executing multiple signal generators in sequence.
It implements the pipeline pattern for sequential data processing, allowing for
composable signal generation workflows.

Features:
    - Execute multiple signal generators as a single unit
    - Collect results from all generators
    - Continue execution even if individual generators fail
    - Dynamically add or remove generators from pipeline
"""

# Standard library imports
import logging
import time
from typing import Dict, List

# Third-party imports
import pandas as pd

# Local imports
from .signal_generator import SignalGenerator
from quant_research.core.models import Signal

# Configure logger
logger = logging.getLogger("quant_research.analytics")


class SignalPipeline:
    """
    Pipeline for executing multiple signal generators.
    
    This class allows combining multiple signal generators into a processing
    pipeline that can be executed as a single unit. It implements the pipeline
    pattern for sequential data processing.
    
    Usage:
        # Create generators
        vol_gen = VolatilitySignalGenerator(window=21)
        regime_gen = RegimeDetectorSignalGenerator(n_states=3)
        
        # Create pipeline
        pipeline = SignalPipeline([vol_gen, regime_gen])
        
        # Run pipeline
        results = pipeline.run(data_df)
    """
    
    def __init__(self, generators: List[SignalGenerator]):
        """
        Initialize the signal pipeline.
        
        Args:
            generators: List of signal generators to include in the pipeline
        """
        self.generators = generators
        self.logger = logging.getLogger("quant_research.analytics.pipeline")
    
    def run(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run all signal generators in the pipeline.
        
        Executes each signal generator in sequence and collects their results.
        Errors in individual generators do not stop the pipeline.
        
        Args:
            df: Input DataFrame with market data
            
        Returns:
            Dictionary mapping generator names to signal DataFrames
        """
        self.logger.info(f"Running signal pipeline with {len(self.generators)} generators")
        
        results = {}
        start_time = time.time()
        
        for generator in self.generators:
            try:
                name = generator.name
                self.logger.info(f"Running generator: {name}")
                
                signals = generator.generate_signal(df)
                
                # Skip None results
                if signals is None:
                    self.logger.warning(f"Generator {name} returned None")
                    continue
                    
                # Convert Signal objects to DataFrame if needed
                if isinstance(signals, list) and signals and isinstance(signals[0], Signal):
                    signals_df = pd.DataFrame([s.__dict__ for s in signals])
                else:
                    signals_df = signals
                
                results[name] = signals_df
                
            except Exception as e:
                self.logger.error(f"Error in generator {generator.name}: {e}", exc_info=True)
                # Continue with other generators instead of stopping the pipeline
        
        elapsed = time.time() - start_time
        self.logger.info(f"Pipeline completed in {elapsed:.2f} seconds with {len(results)} successful generators")
        
        return results
    
    def add_generator(self, generator: SignalGenerator) -> None:
        """
        Add a signal generator to the pipeline.
        
        Args:
            generator: Signal generator to add
            
        Returns:
            None
        """
        self.generators.append(generator)
        self.logger.debug(f"Added generator {generator.name} to pipeline")
    
    def remove_generator(self, name: str) -> bool:
        """
        Remove a signal generator from the pipeline by name.
        
        Args:
            name: Name of the generator to remove
            
        Returns:
            True if a generator was removed, False otherwise
        """
        initial_count = len(self.generators)
        self.generators = [g for g in self.generators if g.name != name]
        removed = len(self.generators) < initial_count
        
        if removed:
            self.logger.debug(f"Removed generator {name} from pipeline")
        
        return removed