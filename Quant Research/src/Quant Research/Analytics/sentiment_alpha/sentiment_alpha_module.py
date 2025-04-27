"""
Sentiment Alpha Module

This module implements sentiment analysis on social media data and converts it into
tradeable signals for quantitative trading strategies.

The module follows the project's analytics interface pattern by exposing a
`generate_signal(df, **params)` function that processes input data and writes
standardized signals to the signal store.
"""

# Standard library imports
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local imports
from quant_research.core.models import Signal
from quant_research.core.storage import save_to_duckdb
from quant_research.analytics.common.base import SignalGenerator
from quant_research.analytics.common.data_prep import (
    ensure_datetime_index, 
    add_technical_features,
    normalize_data
)
from quant_research.analytics.common.validation import (
    validate_dataframe,
    validate_string_param,
    validate_list_param
)
from quant_research.analytics.common.statistics import (
    calculate_correlation,
    calculate_information_coefficient,
    bootstrap_statistic
)
from quant_research.analytics.common.visualization import (
    plot_correlation_matrix,
    plot_time_series,
    create_multi_panel
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"
DEFAULT_WINDOW_SIZES = [1, 3, 5, 7, 14]  # Days for feature calculation


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis and signal generation."""
    
    # Model settings
    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 16
    
    # Feature generation settings
    window_sizes: List[int] = None
    
    # Signal generation settings
    zscore_threshold: float = 1.5
    signal_threshold: float = 0.5
    
    # Output settings
    output_dir: Optional[str] = None
    signal_output_path: str = "data/signals.parquet"
    
    def __post_init__(self):
        """Set default values for optional fields."""
        if self.window_sizes is None:
            self.window_sizes = DEFAULT_WINDOW_SIZES.copy()


class SentimentAnalyzer:
    """
    Analyzes sentiment in text data using transformer models.
    
    This class handles the NLP aspect of sentiment extraction from raw text,
    using pre-trained transformer models from the HuggingFace library.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initialize the sentiment analyzer with a pre-trained model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Initializing SentimentAnalyzer with model: {model_name}")
        
        # Validate input
        model_name = validate_string_param(model_name, "model_name")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Clean and normalize text for sentiment analysis.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            Preprocessed text strings
        """
        # Validate input
        texts = validate_list_param(texts, "texts", item_type=str, allow_none=True)
        
        processed = []
        
        for text in texts:
            # Skip empty texts
            if not text or not isinstance(text, str):
                processed.append("")
                continue
                
            # Basic cleaning
            text = text.replace('\n', ' ')  # Remove newlines
            
            # Add more sophisticated preprocessing as needed
            processed.append(text)
            
        return processed
    
    def analyze_batch(
        self, texts: List[str], batch_size: int = 16
    ) -> List[Dict[str, float]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            List of sentiment dictionaries with scores for each class
        """
        # Validate inputs
        texts = validate_list_param(texts, "texts", item_type=str, allow_none=True)
        batch_size = max(1, int(batch_size))
        
        results = []
        preprocessed_texts = self.preprocess_texts(texts)
        
        # Process in batches to manage memory
        for i in range(0, len(preprocessed_texts), batch_size):
            batch = preprocessed_texts[i:i+batch_size]
            
            # Skip empty batches
            if not any(batch):
                results.extend([{"positive": 0.0, "neutral": 1.0, "negative": 0.0} 
                               for _ in batch])
                continue
                
            # Tokenize and prepare batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert predictions to dictionaries
            for pred in predictions:
                sentiment_dict = {
                    "positive": pred[2].item(),
                    "neutral": pred[1].item(),
                    "negative": pred[0].item()
                }
                results.append(sentiment_dict)
        
        return results
    
    def get_sentiment_score(self, sentiment_dict: Dict[str, float]) -> float:
        """
        Convert sentiment probabilities to a single score.
        
        Args:
            sentiment_dict: Dictionary with class probabilities
            
        Returns:
            Score between -1 (negative) and 1 (positive)
        """
        return sentiment_dict["positive"] - sentiment_dict["negative"]


class TextPreprocessor:
    """
    Preprocesses raw social media text data for sentiment analysis.
    """
    
    @staticmethod
    def clean_tweet_text(text: str) -> str:
        """
        Clean a single tweet text.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = text.replace(r'http\S+', '', regex=True)
        
        # Remove user mentions
        text = text.replace(r'@\w+', '', regex=True)
        
        # Remove hashtag symbols (keep the text)
        text = text.replace(r'#', '', regex=True)
        
        # Remove extra whitespace
        text = text.strip()
        
        return text
    
    @classmethod
    def preprocess_dataframe(
        cls,
        df: pd.DataFrame,
        text_col: str = "text",
        time_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Preprocess a DataFrame containing tweet data.
        
        Args:
            df: Raw tweet DataFrame
            text_col: Column containing tweet text
            time_col: Column containing timestamp
            
        Returns:
            Preprocessed DataFrame
        """
        # Validate input
        df, errors = validate_dataframe(
            df, 
            required_columns=[text_col, time_col],
            raise_exceptions=False
        )
        
        if errors:
            logger.warning(f"DataFrame validation had issues: {'; '.join(errors)}")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure timestamp is datetime
        df = ensure_datetime_index(df, timestamp_col=time_col, inplace=True)
        
        # Sort by time
        df = df.sort_values(by=time_col)
        
        # Clean text
        df[text_col] = df[text_col].apply(cls.clean_tweet_text)
        
        # Filter out empty tweets
        df = df[df[text_col].str.len() > 0].reset_index(drop=True)
        
        return df


class SentimentSignalGenerator(SignalGenerator):
    """
    Generates trading signals from social media sentiment data.
    
    This class extends the base SignalGenerator class from the analytics
    common framework to provide sentiment-specific signal generation.
    """
    
    def __init__(self, config: Optional[SentimentConfig] = None, **kwargs):
        """
        Initialize the sentiment signal generator.
        
        Args:
            config: Configuration for sentiment analysis
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        # Use provided config or create a default one
        self.config = config or SentimentConfig()
        
        # Initialize components
        self.analyzer = SentimentAnalyzer(self.config.model_name)
        
        # Create output directory if specified
        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation-specific signal generation logic.
        
        This method is called by the base class's generate_signal method.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with signals
        """
        # Extract parameters
        text_col = self.params.get("text_col", "text")
        time_col = self.params.get("time_col", "timestamp")
        
        # Preprocess text data
        preprocessed_df = TextPreprocessor.preprocess_dataframe(
            df, text_col=text_col, time_col=time_col
        )
        
        # Extract sentiment from text
        sentiment_scores = self.analyzer.analyze_batch(
            preprocessed_df[text_col].tolist(), 
            batch_size=self.config.batch_size
        )
        
        # Create daily aggregated sentiment features
        daily_sentiment = self._aggregate_daily_sentiment(
            preprocessed_df, sentiment_scores, time_col
        )
        
        # Generate features from daily sentiment
        features_df = self._create_sentiment_features(daily_sentiment)
        
        # Analyze performance with price data if available
        price_df = self.params.get("price_df")
        feature_cols = self._analyze_performance(features_df, price_df)
        
        # Generate signals from features
        signals_df = self._generate_signals_from_features(features_df, feature_cols)
        
        return signals_df
    
    def _aggregate_daily_sentiment(
        self,
        df: pd.DataFrame,
        sentiment_scores: List[Dict[str, float]],
        time_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Aggregate tweet-level sentiment to daily metrics.
        
        Args:
            df: DataFrame with preprocessed tweets
            sentiment_scores: List of sentiment dictionaries
            time_col: Column containing timestamps
            
        Returns:
            DataFrame with daily sentiment metrics
        """
        # Add sentiment scores to the dataframe
        df = df.copy()
        df["positive"] = [score["positive"] for score in sentiment_scores]
        df["neutral"] = [score["neutral"] for score in sentiment_scores]
        df["negative"] = [score["negative"] for score in sentiment_scores]
        df["sentiment_score"] = [
            self.analyzer.get_sentiment_score(score) for score in sentiment_scores
        ]
        
        # Convert to date for grouping
        df["date"] = df[time_col].dt.date
        
        # Group by date and calculate various metrics
        aggregations = {
            "sentiment_score": ["mean", "median", "std", "count"],
            "positive": "mean",
            "negative": "mean",
            "neutral": "mean"
        }
        
        daily = df.groupby("date").agg(aggregations)
        
        # Flatten column names
        daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
        
        # Reset index and convert date to datetime
        daily = daily.reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        
        return daily
    
    def _create_sentiment_features(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-series features from daily sentiment.
        
        This method uses the common add_technical_features function
        to create standardized features from the daily sentiment data.
        
        Args:
            daily_df: DataFrame with daily sentiment metrics
            
        Returns:
            DataFrame with sentiment features
        """
        df = daily_df.sort_values(by="date").copy()
        df = df.set_index("date")
        
        # Generate features using common utilities
        feature_cols = []
        
        # Create features for each window size
        for window in self.config.window_sizes:
            # Use the common add_technical_features function with sentiment data
            window_features = add_technical_features(
                df, 
                price_col="sentiment_score_mean",
                window=window,
                include=["ma", "zscore", "roc", "ema", "momentum"]
            )
            
            # Merge with original dataframe
            df = pd.concat([df, window_features], axis=1)
            
            # Track feature columns
            feature_cols.extend([
                f"sentiment_score_mean_ma_{window}",
                f"sentiment_score_mean_zscore_{window}",
                f"sentiment_score_mean_roc_{window}",
                f"sentiment_score_mean_ema_{window}",
                f"sentiment_score_mean_momentum_{window}"
            ])
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def _analyze_performance(
        self,
        sentiment_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Analyze correlation with price movements and select features.
        
        Args:
            sentiment_df: DataFrame with sentiment features
            price_df: Optional DataFrame with price data
            
        Returns:
            List of selected feature columns
        """
        # Default feature columns if no price data
        default_features = [
            f"sentiment_score_mean_zscore_{window}" 
            for window in self.config.window_sizes[:3]
        ]
        
        if price_df is None:
            return default_features
        
        feature_cols = []
        for window in self.config.window_sizes:
            feature_cols.extend([
                f"sentiment_score_mean_ma_{window}",
                f"sentiment_score_mean_zscore_{window}", 
                f"sentiment_score_mean_roc_{window}",
                f"sentiment_score_mean_ema_{window}",
                f"sentiment_score_mean_momentum_{window}"
            ])
        
        # Ensure price_df has a returns column
        if "returns" not in price_df.columns and "close" in price_df.columns:
            price_df = price_df.copy()
            price_df["returns"] = price_df["close"].pct_change()
        
        # Reset index to make sure align works
        sentiment_df = sentiment_df.reset_index()
        price_df = price_df.reset_index()
        
        # Calculate correlation with next-day returns
        next_day_returns = price_df["returns"].shift(-1)
        
        # Calculate Information Coefficient for each feature
        ic_results = []
        
        for col in feature_cols:
            if col not in sentiment_df.columns:
                continue
                
            # Align data
            feature = sentiment_df.set_index("date")[col]
            returns = next_day_returns.set_index(price_df["date"])
            
            # Align series and drop NAs
            feature, returns = feature.align(returns, join="inner")
            
            if len(feature) < 10:
                continue
                
            # Calculate IC using common statistics function
            ic = calculate_information_coefficient(feature, returns, method="rank")
            
            # Calculate statistics via bootstrap
            def calc_corr(data):
                x, y = data[:, 0], data[:, 1]
                return stats.spearmanr(x, y)[0]
            
            combined_data = np.column_stack([feature.values, returns.values])
            bootstrap_results = bootstrap_statistic(
                combined_data,
                calc_corr,
                n_samples=1000,
                confidence_level=0.95
            )
            
            # Calculate p-value
            t_stat = ic * np.sqrt((len(feature) - 2) / (1 - ic**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(feature) - 2))
            
            ic_results.append({
                "feature": col,
                "ic": ic,
                "lower_bound": bootstrap_results["lower_bound"],
                "upper_bound": bootstrap_results["upper_bound"],
                "p_value": p_value,
                "significant": p_value < 0.05
            })
        
        if not ic_results:
            return default_features
        
        # Create DataFrame and sort by absolute IC
        ic_df = pd.DataFrame(ic_results)
        ic_df["abs_ic"] = ic_df["ic"].abs()
        ic_df = ic_df.sort_values("abs_ic", ascending=False)
        
        # Plot results if output directory specified
        if self.config.output_dir:
            self._plot_ic_results(ic_df)
        
        # Select best features
        top_n = 5
        significant = ic_df[ic_df["significant"]].copy()
        
        if len(significant) >= top_n:
            return significant.head(top_n)["feature"].tolist()
        else:
            return ic_df.head(top_n)["feature"].tolist()
    
    def _plot_ic_results(self, ic_df: pd.DataFrame) -> None:
        """
        Create visualizations of Information Coefficient results.
        
        Args:
            ic_df: DataFrame with IC results
        """
        if ic_df.empty:
            return
        
        # Take top features
        top_ic = ic_df.head(15).copy()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors based on significance and sign
        colors = []
        for _, row in top_ic.iterrows():
            if row["significant"]:
                colors.append("forestgreen" if row["ic"] > 0 else "firebrick")
            else:
                colors.append("lightgreen" if row["ic"] > 0 else "lightcoral")
        
        # Create horizontal bar chart
        bars = ax.barh(
            top_ic["feature"],
            top_ic["ic"],
            color=colors,
            height=0.6
        )
        
        # Save using common visualization function
        output_path = os.path.join(self.config.output_dir, "sentiment_ic_table.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved IC table plot to {output_path}")
    
    def _generate_signals_from_features(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Generate trading signals from sentiment features.
        
        Args:
            features_df: DataFrame with sentiment features
            feature_cols: List of features to use
            
        Returns:
            DataFrame with signals
        """
        df = features_df.copy().reset_index()
        
        # Filter to valid features
        valid_features = [col for col in feature_cols if col in df.columns]
        
        if not valid_features:
            logger.warning("No valid features for signal generation")
            df["signal"] = 0
            return df
        
        # Generate individual signals for each feature
        for feature in valid_features:
            values = df[feature].dropna()
            
            # Set thresholds
            mean = values.mean()
            std = values.std()
            z = self.config.zscore_threshold
            lower = mean - z * std
            upper = mean + z * std
            
            # Generate discrete signals
            signal_col = f"{feature}_signal"
            df[signal_col] = 0
            df.loc[df[feature] > upper, signal_col] = 1
            df.loc[df[feature] < lower, signal_col] = -1
        
        # Generate composite signal
        signal_cols = [f"{feature}_signal" for feature in valid_features]
        
        # Equal weighting for now (could be improved)
        df["composite_score"] = df[signal_cols].mean(axis=1)
        
        # Discretize composite score
        threshold = self.config.signal_threshold
        df["signal"] = 0
        df.loc[df["composite_score"] > threshold, "signal"] = 1
        df.loc[df["composite_score"] < -threshold, "signal"] = -1
        
        return df
    
    def _create_signal_objects(
        self, signals_df: pd.DataFrame
    ) -> List[Signal]:
        """
        Create Signal objects from the DataFrame.
        
        Args:
            signals_df: DataFrame with signals
            
        Returns:
            List of Signal objects
        """
        signal_records = []
        
        for _, row in signals_df.iterrows():
            if pd.notnull(row["signal"]) and row["signal"] != 0:
                signal_records.append(
                    Signal(
                        timestamp=row["date"],
                        source="sentiment_alpha",
                        signal_value=float(row["signal"]),
                        confidence=float(abs(row["composite_score"])),
                        metadata={
                            "sentiment_mean": float(row["sentiment_score_mean"]),
                            "tweet_count": int(row["sentiment_score_count"])
                        }
                    )
                )
        
        return signal_records
    
    def save_signals(self, signals: List[Signal]) -> bool:
        """
        Save signals to the signal store.
        
        Args:
            signals: List of Signal objects
            
        Returns:
            True if successful, False otherwise
        """
        if not signals:
            logger.warning("No signals to save")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config.signal_output_path), exist_ok=True)
            
            # Save to parquet using core storage utility
            save_to_parquet(signals, self.config.signal_output_path)
            
            # Optional: Save to DuckDB
            try:
                save_to_duckdb(signals, "signals", mode="append")
            except Exception as e:
                logger.warning(f"Failed to save to DuckDB: {e}")
            
            logger.info(f"Saved {len(signals)} signals to {self.config.signal_output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving signals: {e}")
            return False


def generate_signal(df: pd.DataFrame, price_df: pd.DataFrame = None, **params) -> pd.DataFrame:
    """
    Generate sentiment-based trading signals from social media data.
    
    This is the main entry point function for the sentiment alpha module.
    It processes raw social media data, extracts sentiment, creates features,
    analyzes correlations with price (if available), and generates signals.
    
    Args:
        df: DataFrame containing social media data
        price_df: Optional DataFrame containing price data for correlation analysis
        params: Additional parameters:
            - text_col: Column name with text content (default: "text")
            - time_col: Column name with timestamps (default: "timestamp")
            - model_name: HuggingFace model name (default: from SentimentConfig)
            - window_sizes: List of window sizes for features (default: from SentimentConfig)
            - output_dir: Directory to save analysis plots (default: None)
            - signal_output_path: Path for signal output (default: from SentimentConfig)
            - save_signals: Whether to save signals (default: True)
        
    Returns:
        DataFrame containing features and signals
    """
    # Extract key parameters
    model_name = params.get("model_name", DEFAULT_MODEL_NAME)
    window_sizes = params.get("window_sizes", DEFAULT_WINDOW_SIZES)
    output_dir = params.get("output_dir", None)
    signal_output_path = params.get("signal_output_path", "data/signals.parquet")
    
    # Create configuration
    config = SentimentConfig(
        model_name=model_name,
        window_sizes=window_sizes,
        output_dir=output_dir,
        signal_output_path=signal_output_path
    )
    
    # Add price_df to params if provided
    if price_df is not None:
        params["price_df"] = price_df
    
    # Initialize signal generator
    generator = SentimentSignalGenerator(config=config, **params)
    
    # Generate signals
    return generator.generate_signal(df=df)