# src/quant_research/providers/sentiment_twitter.py
import asyncio
import logging
from datetime import datetime, timedelta
import json
from typing import Dict, Any, AsyncIterator, Optional, List, Union, Tuple
import os
import re
import time

import nest_asyncio
import twint
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from pydantic import Field, validator

from ..core.config import ProviderConfig, ProviderType
from ..core.models import SentimentData
from ..core.errors import (
    ConnectionError, DataFetchError, 
    RateLimitError, AuthenticationError
)
from .base import BaseProvider
from .connection_manager import ConnectionManager


# Enable nested asyncio to work with twint's event loop requirements
nest_asyncio.apply()

logger = logging.getLogger(__name__)


class TwitterSentimentConfig(ProviderConfig):
    """Configuration for Twitter Sentiment Provider"""
    
    # Override defaults from base
    name: str = "sentiment_twitter"
    type: ProviderType = ProviderType.SENTIMENT
    env_prefix: str = "TWITTER"
    
    # Twitter-specific settings
    keywords: List[str] = Field(..., description="Keywords or cashtags to track (e.g., 'bitcoin', '$BTC')")
    max_tweets: int = Field(default=100, description="Maximum number of tweets to fetch per request")
    since_days: int = Field(default=1, description="Fetch tweets from last N days")
    language: str = Field(default="en", description="Language filter for tweets")
    verified_only: bool = Field(default=False, description="Only fetch tweets from verified accounts")
    
    # Sentiment analysis settings
    sentiment_model: str = Field(
        default="finiteautomata/bertweet-base-sentiment-analysis", 
        description="Hugging Face model for sentiment analysis"
    )
    batch_size: int = Field(default=32, description="Batch size for sentiment analysis")
    cache_duration: int = Field(default=3600, description="Cache duration in seconds")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=5, description="Requests per minute")
    
    @validator('keywords')
    def validate_keywords(cls, v):
        """Validate keywords format"""
        if not v or len(v) == 0:
            raise ValueError("At least one keyword must be provided")
        
        # Cashtags should start with $
        for keyword in v:
            if keyword.startswith("$") and not re.match(r'^\$[A-Za-z]+', keyword):
                raise ValueError(f"Invalid cashtag format: {keyword}. Expected format: '$BTC'")
        
        return v
    
    class Config:
        arbitrary_types_allowed = True


class TwitterSentimentProvider(BaseProvider[TwitterSentimentConfig]):
    """
    Provider for Twitter sentiment data using Twint and Hugging Face transformers.
    
    Features:
    - Fetch tweets based on keywords, cashtags, or users
    - Analyze sentiment using pre-trained models
    - Compute aggregate sentiment metrics
    - Cache results to minimize API calls
    """
    
    def __init__(self, config: TwitterSentimentConfig):
        """Initialize the Twitter sentiment provider"""
        self.config = config
        self._sentiment_pipeline = None
        self._cache = {}
        self._cache_timestamps = {}
        self._initialized = False
        self._last_request_time = 0
        self._connection_manager = None
    
    async def _create_connection(self) -> Any:
        """Create connection to sentiment analysis model"""
        try:
            # Initialize sentiment analysis pipeline using transformers
            tokenizer = AutoTokenizer.from_pretrained(self.config.sentiment_model)
            model = AutoModelForSequenceClassification.from_pretrained(self.config.sentiment_model)
            
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=-1  # CPU, use device=0 for GPU if available
            )
            
            # Test the pipeline with a sample text
            _ = sentiment_pipeline("Test message for sentiment analysis")[0]
            
            return sentiment_pipeline
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analysis pipeline: {e}")
            raise ConnectionError(f"Failed to initialize sentiment model: {e}")
    
    async def _check_connection(self, conn: Any) -> bool:
        """Check if sentiment pipeline is working"""
        try:
            result = conn("Connection test message")
            return isinstance(result, list) and len(result) > 0
        except Exception:
            return False
    
    async def connect(self) -> None:
        """Establish connection to the sentiment analysis pipeline"""
        if self._initialized:
            return
        
        # Initialize connection manager
        self._connection_manager = ConnectionManager(
            connection_factory=self._create_connection,
            config=self.config.connection,
            health_check=self._check_connection,
            cleanup=lambda _: asyncio.sleep(0)  # No need for cleanup
        )
        
        await self._connection_manager.initialize()
        self._initialized = True
        logger.info(f"Connected to sentiment analysis pipeline: {self.config.sentiment_model}")
    
    async def is_connected(self) -> bool:
        """Check if provider is connected"""
        if not self._initialized or not self._connection_manager:
            return False
        
        try:
            async with self._connection_manager.acquire() as sentiment_pipeline:
                return await self._check_connection(sentiment_pipeline)
        except Exception:
            return False
    
    async def disconnect(self) -> None:
        """Close connection and release resources"""
        if self._connection_manager:
            await self._connection_manager.close()
            self._initialized = False
            logger.info("Disconnected from sentiment analysis pipeline")
    
    async def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the provider and available data"""
        if not self._initialized:
            await self.connect()
        
        metadata = {
            "keywords": self.config.keywords,
            "language": self.config.language,
            "sentiment_model": self.config.sentiment_model,
            "connection_stats": {}
        }
        
        try:
            metadata["connection_stats"] = self._connection_manager.get_stats()
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
        
        return metadata
    
    async def _fetch_tweets(self, keyword: str, since_date: str) -> List[Dict[str, Any]]:
        """
        Fetch tweets using twint
        
        Args:
            keyword: Keyword or cashtag to search
            since_date: Date to fetch tweets from (format: YYYY-MM-DD)
            
        Returns:
            List of tweet dictionaries
        """
        # Apply rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < (60 / self.config.rate_limit_requests):
            await asyncio.sleep((60 / self.config.rate_limit_requests) - time_since_last_request)
        
        self._last_request_time = time.time()
        
        # Configure twint
        c = twint.Config()
        c.Search = keyword
        c.Lang = self.config.language
        c.Limit = self.config.max_tweets
        c.Since = since_date
        c.Pandas = True
        c.Hide_output = True
        
        if self.config.verified_only:
            c.Verified = True
        
        # Run search
        twint.run.Search(c)
        
        # Get tweets from Pandas dataframe
        try:
            tweets_df = twint.storage.panda.Tweets_df
            if tweets_df.empty:
                return []
            
            # Convert to list of dictionaries
            tweets = tweets_df.to_dict('records')
            return tweets
        except Exception as e:
            logger.error(f"Error fetching tweets for {keyword}: {e}")
            raise DataFetchError(
                f"Failed to fetch tweets: {e}",
                provider_id=self.config.name,
                provider_type=self.config.type.value,
                keyword=keyword,
                original_error=e
            )
    
    async def _analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of texts using transformer model
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment results
        """
        if not texts:
            return []
        
        async with self._connection_manager.acquire() as sentiment_pipeline:
            # Process in batches to avoid memory issues
            results = []
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                try:
                    batch_results = sentiment_pipeline(batch)
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Error analyzing sentiment for batch: {e}")
                    # Return neutral sentiment for failed batches
                    results.extend([{"label": "neutral", "score": 0.5} for _ in batch])
            
            return results
    
    async def _compute_aggregate_sentiment(self, 
                                          keyword: str, 
                                          tweets: List[Dict[str, Any]], 
                                          sentiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregate sentiment metrics
        
        Args:
            keyword: Keyword or cashtag
            tweets: List of tweets
            sentiment_results: List of sentiment analysis results
            
        Returns:
            Aggregate sentiment metrics
        """
        if not tweets or not sentiment_results:
            return {
                "keyword": keyword,
                "timestamp": datetime.now(),
                "positive_ratio": 0,
                "negative_ratio": 0,
                "neutral_ratio": 0,
                "sentiment_score": 0,
                "tweet_count": 0,
                "source": "twitter"
            }
        
        # Map sentiment labels
        sentiment_map = {
            'POS': 1,  # Positive
            'NEG': -1,  # Negative
            'NEU': 0,   # Neutral
            'positive': 1,
            'negative': -1,
            'neutral': 0
        }
        
        pos_count = sum(1 for res in sentiment_results if res['label'].lower() == 'positive')
        neg_count = sum(1 for res in sentiment_results if res['label'].lower() == 'negative')
        neu_count = sum(1 for res in sentiment_results if res['label'].lower() == 'neutral')
        
        total_count = len(sentiment_results)
        
        # Calculate ratios
        pos_ratio = pos_count / total_count if total_count > 0 else 0
        neg_ratio = neg_count / total_count if total_count > 0 else 0
        neu_ratio = neu_count / total_count if total_count > 0 else 0
        
        # Calculate weighted sentiment score
        sentiment_scores = [
            sentiment_map.get(res['label'].upper(), 0) * res['score']
            for res in sentiment_results
        ]
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Create SentimentData instance
        sentiment_data = SentimentData(
            keyword=keyword,
            timestamp=datetime.now(),
            positive_ratio=pos_ratio,
            negative_ratio=neg_ratio, 
            neutral_ratio=neu_ratio,
            sentiment_score=avg_sentiment,
            tweet_count=total_count,
            source="twitter"
        )
        
        return sentiment_data.dict()
    
    async def fetch_data(self, **params) -> AsyncIterator[Dict[str, Any]]:
        """
        Fetch tweets and analyze sentiment.
        
        Parameters:
            keywords (List[str], optional): Keywords to search (defaults to config)
            since_days (int, optional): Days to look back (defaults to config)
            max_tweets (int, optional): Maximum tweets (defaults to config)
            
        Yields:
            Dict[str, Any]: Sentiment data points
        """
        if not self._initialized:
            await self.connect()
        
        # Get parameters with defaults from config
        keywords = params.get('keywords', self.config.keywords)
        since_days = params.get('since_days', self.config.since_days)
        max_tweets = params.get('max_tweets', self.config.max_tweets)
        
        # Calculate since date
        since_date = (datetime.now() - timedelta(days=since_days)).strftime('%Y-%m-%d')
        
        for keyword in keywords:
            # Check cache for recent data
            cache_key = f"{keyword}_{since_date}_{max_tweets}"
            
            if (
                cache_key in self._cache and
                cache_key in self._cache_timestamps and
                (datetime.now() - self._cache_timestamps[cache_key]).total_seconds() < self.config.cache_duration
            ):
                # Use cached data
                yield self._cache[cache_key]
                continue
            
            try:
                # Fetch tweets
                tweets = await self._fetch_tweets(keyword, since_date)
                
                if not tweets:
                    logger.warning(f"No tweets found for keyword: {keyword}")
                    continue
                
                # Extract text content
                tweet_texts = [tweet.get('tweet', '') for tweet in tweets if tweet.get('tweet')]
                
                # Analyze sentiment
                sentiment_results = await self._analyze_sentiment(tweet_texts)
                
                # Compute aggregate metrics
                aggregate_sentiment = await self._compute_aggregate_sentiment(
                    keyword, tweets, sentiment_results
                )
                
                # Store in cache
                self._cache[cache_key] = aggregate_sentiment
                self._cache_timestamps[cache_key] = datetime.now()
                
                # Yield result
                yield aggregate_sentiment
                
            except RateLimitError as e:
                # Re-raise with provider info
                raise RateLimitError(
                    str(e),
                    provider_id=self.config.name,
                    provider_type=self.config.type.value,
                    retry_after=60  # Default retry after 1 minute
                )
            except Exception as e:
                logger.error(f"Error processing sentiment for {keyword}: {e}")
                raise DataFetchError(
                    f"Failed to process sentiment data: {e}",
                    provider_id=self.config.name,
                    provider_type=self.config.type.value,
                    keyword=keyword,
                    original_error=e
                )
    
    async def fetch_tweet_details(self, tweet_id: str) -> Dict[str, Any]:
        """
        Fetch details for a specific tweet
        
        Args:
            tweet_id: Twitter ID
            
        Returns:
            Tweet details with sentiment analysis
        """
        if not self._initialized:
            await self.connect()
        
        try:
            # Configure twint for specific tweet
            c = twint.Config()
            c.Search = f"from:{tweet_id}"
            c.Pandas = True
            c.Hide_output = True
            
            # Run search
            twint.run.Search(c)
            
            # Get tweet from Pandas dataframe
            tweets_df = twint.storage.panda.Tweets_df
            if tweets_df.empty:
                raise DataFetchError(
                    f"Tweet not found: {tweet_id}",
                    provider_id=self.config.name,
                    provider_type=self.config.type.value,
                    tweet_id=tweet_id
                )
            
            tweet = tweets_df.iloc[0].to_dict()
            
            # Analyze sentiment
            sentiment_results = await self._analyze_sentiment([tweet.get('tweet', '')])
            
            # Add sentiment to tweet details
            tweet['sentiment'] = sentiment_results[0] if sentiment_results else {"label": "neutral", "score": 0.5}
            
            return tweet
        except Exception as e:
            logger.error(f"Error fetching tweet details for {tweet_id}: {e}")
            raise DataFetchError(
                f"Failed to fetch tweet details: {e}",
                provider_id=self.config.name,
                provider_type=self.config.type.value,
                tweet_id=tweet_id,
                original_error=e
            )
    
    async def search_users(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search Twitter users
        
        Args:
            query: Search query for users
            limit: Maximum number of users to return
            
        Returns:
            List of user information
        """
        if not self._initialized:
            await self.connect()
        
        try:
            # Configure twint for user search
            c = twint.Config()
            c.Username = query
            c.Limit = limit
            c.Pandas = True
            c.Hide_output = True
            
            # Run search
            twint.run.SearchUsers(c)
            
            # Get users from Pandas dataframe
            try:
                users_df = twint.storage.panda.User_df
                if users_df.empty:
                    return []
                
                # Convert to list of dictionaries
                users = users_df.to_dict('records')
                return users
            except Exception:
                return []
        except Exception as e:
            logger.error(f"Error searching users for {query}: {e}")
            raise DataFetchError(
                f"Failed to search users: {e}",
                provider_id=self.config.name,
                provider_type=self.config.type.value,
                query=query,
                original_error=e
            )