# src/quant_research/core/models.py
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class DataType(str, Enum):
    """Types of data that can be provided"""
    PRICE = "price"
    OHLCV = "ohlcv"
    TICKER = "ticker"
    ORDER_BOOK = "order_book"
    TRADE = "trade"
    SENTIMENT = "sentiment"
    ONCHAIN = "onchain"
    CUSTOM = "custom"


class PriceBar(BaseModel):
    """OHLCV price bar data"""
    
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Bar timestamp")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: float = Field(..., description="Trading volume")
    source: str = Field(..., description="Data source")
    
    # Optional fields
    trades: Optional[int] = Field(None, description="Number of trades")
    vwap: Optional[float] = Field(None, description="Volume-weighted average price")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate symbol format"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Symbol cannot be empty")
        return v
    
    @validator('high')
    def validate_high(cls, v, values):
        """Validate high price is not less than low/open/close"""
        if 'open' in values and v < values['open']:
            raise ValueError(f"High price ({v}) cannot be less than open price ({values['open']})")
        if 'low' in values and v < values['low']:
            raise ValueError(f"High price ({v}) cannot be less than low price ({values['low']})")
        return v
    
    @validator('low')
    def validate_low(cls, v, values):
        """Validate low price is not greater than open/close/high"""
        if 'open' in values and v > values['open']:
            raise ValueError(f"Low price ({v}) cannot be greater than open price ({values['open']})")
        if 'high' in values and v > values['high']:
            raise ValueError(f"Low price ({v}) cannot be greater than high price ({values['high']})")
        return v
    
    class Config:
        allow_population_by_field_name = True


class Ticker(BaseModel):
    """Real-time ticker data"""
    
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Ticker timestamp")
    bid: Optional[float] = Field(None, description="Best bid price")
    ask: Optional[float] = Field(None, description="Best ask price")
    last: float = Field(..., description="Last trade price")
    volume: Optional[float] = Field(None, description="24h volume")
    source: str = Field(..., description="Data source")
    
    # Optional fields
    change: Optional[float] = Field(None, description="24h price change")
    change_percent: Optional[float] = Field(None, description="24h price change percent")
    high: Optional[float] = Field(None, description="24h high")
    low: Optional[float] = Field(None, description="24h low")
    vwap: Optional[float] = Field(None, description="Volume-weighted average price")
    
    class Config:
        allow_population_by_field_name = True


class OrderBookLevel(BaseModel):
    """Single level in an order book"""
    
    price: float = Field(..., description="Price level")
    amount: float = Field(..., description="Amount available")
    
    class Config:
        allow_population_by_field_name = True


class OrderBook(BaseModel):
    """Order book snapshot"""
    
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Order book timestamp")
    bids: List[OrderBookLevel] = Field(default_factory=list, description="Bid levels")
    asks: List[OrderBookLevel] = Field(default_factory=list, description="Ask levels")
    source: str = Field(..., description="Data source")
    
    class Config:
        allow_population_by_field_name = True


class Trade(BaseModel):
    """Individual trade data"""
    
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Trade timestamp")
    price: float = Field(..., description="Trade price")
    amount: float = Field(..., description="Trade amount")
    side: str = Field(..., description="Trade side (buy/sell)")
    source: str = Field(..., description="Data source")
    
    # Optional fields
    trade_id: Optional[str] = Field(None, description="Trade ID")
    taker: Optional[bool] = Field(None, description="Whether trade was a taker")
    
    class Config:
        allow_population_by_field_name = True


class SentimentScore(BaseModel):
    """Sentiment analysis score"""
    
    symbol: str = Field(..., description="Asset symbol")
    timestamp: datetime = Field(..., description="Timestamp")
    score: float = Field(..., description="Sentiment score (-1 to 1)")
    source: str = Field(..., description="Data source")
    
    # Optional fields
    volume: Optional[int] = Field(None, description="Volume of mentions")
    magnitude: Optional[float] = Field(None, description="Magnitude of sentiment")
    positive: Optional[float] = Field(None, description="Positive component")
    negative: Optional[float] = Field(None, description="Negative component")
    neutral: Optional[float] = Field(None, description="Neutral component")
    
    @validator('score')
    def validate_score(cls, v):
        """Validate score is between -1 and 1"""
        if v < -1 or v > 1:
            raise ValueError(f"Sentiment score must be between -1 and 1, got {v}")
        return v
    
    class Config:
        allow_population_by_field_name = True


class OnchainMetric(BaseModel):
    """Blockchain on-chain metric"""
    
    blockchain: str = Field(..., description="Blockchain name")
    timestamp: datetime = Field(..., description="Timestamp")
    metric: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    source: str = Field(..., description="Data source")
    
    # Optional fields
    address: Optional[str] = Field(None, description="Associated address")
    token: Optional[str] = Field(None, description="Associated token")
    block_number: Optional[int] = Field(None, description="Block number")
    
    class Config:
        allow_population_by_field_name = True


class Signal(BaseModel):
    """Trading signal generated from analysis"""
    
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Signal timestamp")
    signal_type: str = Field(..., description="Signal type (e.g., 'buy', 'sell', 'hold')")
    strength: float = Field(..., description="Signal strength (0-1)")
    model: str = Field(..., description="Model that generated the signal")
    
    # Optional fields
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    confidence: Optional[float] = Field(None, description="Signal confidence")
    price_target: Optional[float] = Field(None, description="Price target")
    stop_loss: Optional[float] = Field(None, description="Stop loss level")
    time_horizon: Optional[str] = Field(None, description="Time horizon")
    
    @validator('strength')
    def validate_strength(cls, v):
        """Validate strength is between 0 and 1"""
        if v < 0 or v > 1:
            raise ValueError(f"Signal strength must be between 0 and 1, got {v}")
        return v
    
    @validator('signal_type')
    def validate_signal_type(cls, v):
        """Validate signal type"""
        valid_types = {'buy', 'sell', 'hold', 'strong_buy', 'strong_sell'}
        if v.lower() not in valid_types:
            raise ValueError(f"Signal type must be one of {valid_types}, got {v}")
        return v.lower()
    
    class Config:
        allow_population_by_field_name = True


class DataPoint(BaseModel):
    """Generic data point that can contain any type of data"""
    
    timestamp: datetime = Field(..., description="Data timestamp")
    type: DataType = Field(..., description="Type of data")
    source: str = Field(..., description="Data source")
    symbol: Optional[str] = Field(None, description="Symbol if applicable")
    
    # The actual data payload
    data: Dict[str, Any] = Field(..., description="Data payload")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        allow_population_by_field_name = True