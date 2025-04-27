"""
Sentiment Alpha Module

This module implements sentiment analysis on social media data and converts it into
tradeable signals for quantitative trading strategies.

The module follows the project's analytics interface pattern by exposing a
`generate_signal(df, **params)` function that processes input data and writes
standardized signals to the signal store.

Classes:
    SentimentAnalyzer: Core transformer-based sentiment extraction
    SentimentFeatureEngine: Creates time-series features from raw sentiment
    SignalGenerator: Converts sentiment features into trading signals
    PerformanceAnalyzer: Analyzes signal efficacy through various metrics
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import ccf
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from quant_research.core.models import Signal
from quant_research.core.storage import save_to_duckdb

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"
DEFAULT_WINDOW_SIZES = [1, 3, 5, 7, 14]  # Days for feature calculation
DEFAULT_LAG_RANGE = range(-10, 11)  # Days from -10 to +10 for correlation
DEFAULT_PLOT_DIR = "data/plots/sentiment"


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
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure timestamp is datetime
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Sort by time
        df = df.sort_values(by=time_col)
        
        # Clean text
        df[text_col] = df[text_col].apply(cls.clean_tweet_text)
        
        # Filter out empty tweets
        df = df[df[text_col].str.len() > 0].reset_index(drop=True)
        
        return df


class SentimentFeatureEngine:
    """
    Creates time-series features from raw sentiment data.
    
    This class handles the transformation of raw tweet-level sentiment
    into aggregated time-series features suitable for signal generation.
    """
    
    def __init__(self, config: SentimentConfig):
        """
        Initialize the feature engine.
        
        Args:
            config: Configuration parameters
        """
        self.config = config
    
    def aggregate_daily(
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
            score["positive"] - score["negative"] for score in sentiment_scores
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
    
    def create_features(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-series features from daily sentiment.
        
        Args:
            daily_df: DataFrame with daily sentiment metrics
            
        Returns:
            DataFrame with sentiment features
        """
        df = daily_df.sort_values(by="date").copy()
        
        # Create features for each window size
        for window in self.config.window_sizes:
            # Simple moving average
            df[f"sentiment_sma_{window}d"] = (
                df["sentiment_score_mean"].rolling(window=window).mean()
            )
            
            # Standard deviation (volatility)
            df[f"sentiment_std_{window}d"] = (
                df["sentiment_score_mean"].rolling(window=window).std()
            )
            
            # Z-score (normalized deviation from mean)
            mean_col = f"sentiment_sma_{window}d"
            std_col = f"sentiment_std_{window}d"
            
            df[f"sentiment_zscore_{window}d"] = (
                df["sentiment_score_mean"] - df[mean_col]
            ) / df[std_col].replace(0, np.nan)
            
            # Rate of change
            df[f"sentiment_roc_{window}d"] = (
                df["sentiment_score_mean"].pct_change(periods=window)
            )
            
            # Exponential moving average
            df[f"sentiment_ema_{window}d"] = (
                df["sentiment_score_mean"].ewm(span=window).mean()
            )
            
            # Momentum (difference between current value and n periods ago)
            df[f"sentiment_momentum_{window}d"] = (
                df["sentiment_score_mean"] - 
                df["sentiment_score_mean"].shift(window)
            )
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df


class PerformanceAnalyzer:
    """
    Analyzes the efficacy of sentiment features for predicting returns.
    """
    
    def __init__(self, config: SentimentConfig):
        """
        Initialize the performance analyzer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.lag_range = DEFAULT_LAG_RANGE
        
        # Create output directory if needed
        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)
    
    def compute_lag_correlations(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        price_col: str = "close",
        return_col: str = "returns"
    ) -> pd.DataFrame:
        """
        Compute correlations between sentiment and returns at various lags.
        
        Args:
            sentiment_df: DataFrame with sentiment data
            price_df: DataFrame with price data
            price_col: Column containing price
            return_col: Column containing returns
            
        Returns:
            DataFrame with lag correlations
        """
        # Prepare sentiment and price data
        sentiment_df = sentiment_df.copy().set_index("date")
        price_df = price_df.copy()
        
        # Calculate returns if not present
        if return_col not in price_df.columns:
            price_df[return_col] = price_df[price_col].pct_change()
        
        price_df = price_df.set_index("date")
        
        # Extract time series
        sentiment = sentiment_df["sentiment_score_mean"]
        returns = price_df[return_col]
        
        # Align indices
        sentiment, returns = sentiment.align(returns, join="inner")
        
        if len(sentiment) < 10:
            logger.warning(
                "Insufficient data for lag correlation analysis "
                f"(only {len(sentiment)} matching dates)"
            )
            return pd.DataFrame(columns=["lag", "correlation", "label", "abs_correlation"])
        
        # Compute correlations at different lags
        results = []
        
        for lag in self.lag_range:
            if lag < 0:
                # Sentiment leads returns
                shifted_sentiment = sentiment.shift(-lag)
                corr = shifted_sentiment.corr(returns)
                label = f"Sentiment(t-{abs(lag)}) → Returns(t)"
            elif lag > 0:
                # Returns lead sentiment
                shifted_returns = returns.shift(-lag)
                corr = sentiment.corr(shifted_returns)
                label = f"Returns(t-{lag}) → Sentiment(t)"
            else:
                # Contemporaneous
                corr = sentiment.corr(returns)
                label = "Sentiment(t) ~ Returns(t)"
            
            results.append({
                "lag": lag,
                "correlation": corr,
                "label": label,
                "abs_correlation": abs(corr)
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values("abs_correlation", ascending=False)
        
        return df
    
    def plot_lag_correlations(
        self, correlation_df: pd.DataFrame
    ) -> Optional[str]:
        """
        Plot lag correlations between sentiment and returns.
        
        Args:
            correlation_df: DataFrame with lag correlations
            
        Returns:
            Path to saved plot if output_dir is specified, otherwise None
        """
        if correlation_df.empty:
            logger.warning("Cannot create plot: empty correlation data")
            return None
        
        plt.figure(figsize=(12, 6))
        
        # Plot bars
        bars = plt.bar(
            correlation_df["lag"],
            correlation_df["correlation"],
            color=correlation_df["correlation"].apply(
                lambda x: "forestgreen" if x > 0 else "firebrick"
            )
        )
        
        # Reference lines
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        # Annotate strongest correlations
        top_corrs = correlation_df.nlargest(3, "abs_correlation")
        for _, row in top_corrs.iterrows():
            plt.annotate(
                f"{row['correlation']:.3f}",
                xy=(row["lag"], row["correlation"]),
                xytext=(0, 10 if row["correlation"] > 0 else -20),
                textcoords="offset points",
                ha='center',
                fontweight='bold'
            )
        
        # Labels and title
        plt.xlabel("Lag (Days)")
        plt.ylabel("Correlation Coefficient")
        plt.title("Sentiment-Return Correlations at Different Lags")
        plt.grid(True, alpha=0.3)
        
        # Explanation text
        textstr = '\n'.join([
            "Negative lag: Sentiment leads Returns",
            "Positive lag: Returns lead Sentiment",
            "Zero lag: Contemporaneous correlation"
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(
            0.05, 0.95, textstr, transform=plt.gca().transAxes,
            verticalalignment='top', bbox=props
        )
        
        plt.tight_layout()
        
        # Save if output directory specified
        if self.config.output_dir:
            output_path = os.path.join(
                self.config.output_dir, "sentiment_lag_correlations.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved lag correlation plot to {output_path}")
            return output_path
        
        return None
    
    def compute_information_coefficient(
        self,
        sentiment_df: pd.DataFrame,
        return_df: pd.DataFrame,
        sentiment_cols: List[str],
        return_col: str = "returns",
        ic_type: str = "rank"
    ) -> pd.DataFrame:
        """
        Compute Information Coefficient for sentiment features.
        
        Args:
            sentiment_df: DataFrame with sentiment features
            return_df: DataFrame with returns
            sentiment_cols: List of sentiment feature columns
            return_col: Column containing returns
            ic_type: Type of correlation ('rank' or 'pearson')
            
        Returns:
            DataFrame with IC analysis
        """
        # Prepare data
        sentiment_df = sentiment_df.set_index("date")
        return_df = return_df.set_index("date")
        
        # Get next-day returns (forward returns)
        next_day_returns = return_df[return_col].shift(-1)
        
        # Calculate IC for each feature
        results = []
        
        for col in sentiment_cols:
            if col not in sentiment_df.columns:
                continue
            
            feature = sentiment_df[col].dropna()
            returns = next_day_returns.loc[feature.index].dropna()
            
            # Align data and drop NaNs
            feature, returns = feature.align(returns, join="inner")
            
            if len(feature) < 10:
                logger.debug(f"Skipping {col}: insufficient data")
                continue
            
            # Calculate correlation
            if ic_type == "rank":
                ic = stats.spearmanr(feature, returns)[0]
            else:  # pearson
                ic = stats.pearsonr(feature, returns)[0]
            
            # Calculate statistics
            t_stat = ic * np.sqrt((len(feature) - 2) / (1 - ic**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(feature) - 2))
            
            results.append({
                "feature": col,
                "ic": ic,
                "t_stat": t_stat,
                "p_value": p_value,
                "observations": len(feature),
                "significant": p_value < 0.05
            })
        
        if not results:
            logger.warning("No valid features for IC calculation")
            return pd.DataFrame()
        
        ic_df = pd.DataFrame(results)
        
        # Sort by absolute IC
        ic_df["abs_ic"] = ic_df["ic"].abs()
        ic_df = ic_df.sort_values("abs_ic", ascending=False)
        
        return ic_df
    
    def plot_ic_table(self, ic_df: pd.DataFrame) -> Optional[str]:
        """
        Create a visualization of the Information Coefficient results.
        
        Args:
            ic_df: DataFrame with IC results
            
        Returns:
            Path to saved plot if output_dir is specified, otherwise None
        """
        if ic_df.empty:
            logger.warning("Cannot create plot: empty IC data")
            return None
        
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
        
        # Reference line
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Labels and title
        ax.set_xlabel("Information Coefficient (IC)")
        ax.set_title("Predictive Power of Sentiment Features")
        ax.grid(True, alpha=0.3, axis='x')
        
        # Annotate significant features
        for i, (_, row) in enumerate(top_ic.iterrows()):
            if row["significant"]:
                ax.text(
                    row["ic"] + (0.01 if row["ic"] > 0 else -0.01),
                    i,
                    f"p={row['p_value']:.3f}*",
                    va='center',
                    fontweight='bold'
                )
        
        # Explanation text
        textstr = '\n'.join([
            "* p < 0.05 (statistically significant)",
            "Positive IC: Higher sentiment → Higher next-day returns",
            "Negative IC: Higher sentiment → Lower next-day returns"
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(
            0.05, 0.05, textstr, transform=ax.transAxes,
            verticalalignment='bottom', bbox=props
        )
        
        plt.tight_layout()
        
        # Save if output directory specified
        if self.config.output_dir:
            output_path = os.path.join(
                self.config.output_dir, "sentiment_ic_table.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved IC table plot to {output_path}")
            return output_path
        
        return None
    
    def get_best_features(self, ic_df: pd.DataFrame, top_n: int = 5) -> List[str]:
        """
        Get the best performing features based on IC analysis.
        
        Args:
            ic_df: DataFrame with IC results
            top_n: Number of top features to return
            
        Returns:
            List of feature column names
        """
        if ic_df.empty:
            # Fallback to default features
            return [
                f"sentiment_zscore_{window}d" 
                for window in self.config.window_sizes[:3]
            ]
        
        # Filter to significant features if possible
        significant = ic_df[ic_df["significant"]].copy()
        
        if len(significant) >= top_n:
            df = significant
        else:
            df = ic_df
        
        # Get top features by absolute IC
        top_features = df.sort_values("abs_ic", ascending=False).head(top_n)
        
        return top_features["feature"].tolist()


class SignalGenerator:
    """
    Converts sentiment features into trading signals.
    """
    
    def __init__(self, config: SentimentConfig):
        """
        Initialize the signal generator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config
    
    def generate_signals(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        thresholds: Dict[str, Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals from sentiment features.
        
        Args:
            features_df: DataFrame with sentiment features
            feature_cols: List of features to use
            thresholds: Custom thresholds for features
            
        Returns:
            DataFrame with signals
        """
        df = features_df.copy()
        
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
            if thresholds and feature in thresholds:
                lower, upper = thresholds[feature]
            else:
                # Default: mean ± z_threshold * std
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
    
    def create_signal_objects(
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
            
            # Convert to pyarrow table
            signals_table = pa.Table.from_pylist([s.dict() for s in signals])
            
            # Write to parquet
            pq.write_table(signals_table, self.config.signal_output_path)
            
            # Optional: Save to DuckDB
            try:
                save_to_duckdb(signals_table, "signals", mode="append")
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
    # Extract parameters
    text_col = params.get("text_col", "text")
    time_col = params.get("time_col", "timestamp")
    model_name = params.get("model_name", DEFAULT_MODEL_NAME)
    window_sizes = params.get("window_sizes", DEFAULT_WINDOW_SIZES)
    output_dir = params.get("output_dir", None)
    signal_output_path = params.get("signal_output_path", "data/signals.parquet")
    save_signals = params.get("save_signals", True)
    
    # Create configuration
    config = SentimentConfig(
        model_name=model_name,
        window_sizes=window_sizes,
        output_dir=output_dir,
        signal_output_path=signal_output_path
    )
    
    logger.info("Starting sentiment signal generation pipeline")
    
    # 1. Preprocess tweet data
    logger.info("Preprocessing social media data")
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.preprocess_dataframe(
        df, text_col=text_col, time_col=time_col
    )
    
    # 2. Extract sentiment
    logger.info(f"Extracting sentiment using {model_name}")
    analyzer = SentimentAnalyzer(model_name=model_name)
    sentiment_scores = analyzer.analyze_batch(
        processed_df[text_col].tolist(), batch_size=16
    )
    
    # 3. Create features
    logger.info("Creating sentiment features")
    feature_engine = SentimentFeatureEngine(config)
    
    # 3.1. Aggregate to daily sentiment
    daily_sentiment = feature_engine.aggregate_daily(
        processed_df, sentiment_scores, time_col=time_col
    )
    
    # 3.2. Create time-series features
    features_df = feature_engine.create_features(daily_sentiment)
    
    # 4. Analyze performance if price data is available
    feature_cols = []
    if price_df is not None:
        logger.info("Analyzing correlation with price movements")
        performance = PerformanceAnalyzer(config)
        
        # 4.1. Compute lag correlations
        lag_corrs = performance.compute_lag_correlations(features_df, price_df)
        performance.plot_lag_correlations(lag_corrs)
        
        # 4.2. Identify potential features
        for window in config.window_sizes:
            feature_cols.extend([
                f"sentiment_sma_{window}d",
                f"sentiment_zscore_{window}d", 
                f"sentiment_roc_{window}d",
                f"sentiment_ema_{window}d",
                f"sentiment_momentum_{window}d"
            ])
        
        # 4.3. Compute information coefficients
        ic_df = performance.compute_information_coefficient(
            features_df, price_df, feature_cols
        )
        performance.plot_ic_table(ic_df)
        
        # 4.4. Get best features for signal generation
        feature_cols = performance.get_best_features(ic_df, top_n=5)
    else:
        # Default features if no price data
        feature_cols = [
            f"sentiment_zscore_{window}d" for window in config.window_sizes[:3]
        ]
    
    # 5. Generate signals
    logger.info("Generating trading signals")
    generator = SignalGenerator(config)
    signals_df = generator.generate_signals(features_df, feature_cols)
    
    # 6. Save signals if requested
    if save_signals:
        signal_objects = generator.create_signal_objects(signals_df)
        generator.save_signals(signal_objects)
    
    logger.info("Sentiment signal generation complete")
    return signals_df


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Run this module through the analytics pipeline or import functions separately.")