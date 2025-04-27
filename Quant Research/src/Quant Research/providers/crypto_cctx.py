# src/quant_research/providers/crypto_ccxt.py
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncIterator, Optional, List, Union

import ccxt.async_support as ccxt
from pydantic import Field, validator

from ..core.config import ProviderConfig, ProviderType
from ..core.models import PriceBar
from ..core.errors import (
    ConnectionError, DataFetchError, 
    RateLimitError, AuthenticationError
)
from .base import BaseProvider
from .connection_manager import ConnectionManager


logger = logging.getLogger(__name__)


class CCXTProviderConfig(ProviderConfig):
    """Configuration for CCXT provider"""
    
    # Override defaults from base
    name: str = "crypto_ccxt"
    type: ProviderType = ProviderType.CRYPTO
    env_prefix: str = "CCXT"
    
    # CCXT-specific settings
    exchange: str = Field(..., description="Exchange ID (e.g., 'binance', 'coinbase')")
    symbols: List[str] = Field(..., description="List of symbols to track (e.g., 'BTC/USDT')")
    timeframe: str = Field(default="1m", description="Timeframe for OHLCV data")
    
    # Authentication (can also be provided via environment variables)
    api_key: Optional[str] = Field(default=None, description="API key")
    api_secret: Optional[str] = Field(default=None, description="API secret")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=10, description="Requests per second")
    
    # Cache settings
    cache_duration: int = Field(default=60, description="Cache duration in seconds")
    
    @validator('exchange')
    def validate_exchange(cls, v):
        """Validate exchange ID"""
        if v not in ccxt.exchanges:
            raise ValueError(f"Invalid exchange: {v}. Supported exchanges: {', '.join(ccxt.exchanges)}")
        return v
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """Validate symbols format"""
        for symbol in v:
            if '/' not in symbol:
                raise ValueError(f"Invalid symbol format: {symbol}. Expected format: 'BTC/USDT'")
        return v
    
    class Config:
        arbitrary_types_allowed = True


class CCXTProvider(BaseProvider[CCXTProviderConfig]):
    """
    Provider for cryptocurrency data using CCXT library.
    
    Features:
    - Historical OHLCV data
    - Real-time ticker data
    - Order book depth
    - Trading pairs information
    """
    
    def __init__(self, config: CCXTProviderConfig):
        """Initialize the CCXT provider"""
        self.config = config
        self._exchange_instance = None
        self._connection_manager = None
        self._cache = {}
        self._cache_timestamps = {}
        self._markets = None
        self._initialized = False
    
    async def _create_exchange(self) -> ccxt.Exchange:
        """Create and initialize a CCXT exchange instance"""
        exchange_class = getattr(ccxt, self.config.exchange)
        
        # Authentication parameters if provided
        params = {}
        if self.config.api_key and self.config.api_secret:
            params.update({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
            })
        
        # Initialize exchange
        exchange = exchange_class(params)
        
        # Set timeouts
        exchange.timeout = self.config.connection.timeout * 1000  # ms
        
        # Set rate limiting
        if hasattr(exchange, 'enableRateLimit'):
            exchange.enableRateLimit = True
        
        # Load markets
        await exchange.load_markets()
        
        return exchange
    
    async def _is_exchange_healthy(self, exchange: ccxt.Exchange) -> bool:
        """Check if exchange connection is healthy"""
        try:
            # Simple test request
            await exchange.fetch_time()
            return True
        except Exception as e:
            logger.warning(f"Exchange health check failed: {e}")
            return False
    
    async def _cleanup_exchange(self, exchange: ccxt.Exchange) -> None:
        """Clean up exchange resources"""
        try:
            await exchange.close()
        except Exception as e:
            logger.warning(f"Error closing exchange connection: {e}")
    
    async def connect(self) -> None:
        """Establish connection to the exchange"""
        if self._initialized:
            return
        
        # Initialize connection manager
        self._connection_manager = ConnectionManager(
            connection_factory=self._create_exchange,
            config=self.config.connection,
            health_check=self._is_exchange_healthy,
            cleanup=self._cleanup_exchange
        )
        
        await self._connection_manager.initialize()
        
        # Fetch markets for later use
        async with self._connection_manager.acquire() as exchange:
            self._markets = exchange.markets
        
        self._initialized = True
        logger.info(f"Connected to exchange: {self.config.exchange}")
    
    async def is_connected(self) -> bool:
        """Check if provider is connected"""
        if not self._initialized or not self._connection_manager:
            return False
        
        try:
            async with self._connection_manager.acquire() as exchange:
                return await self._is_exchange_healthy(exchange)
        except Exception:
            return False
    
    async def disconnect(self) -> None:
        """Close connection and release resources"""
        if self._connection_manager:
            await self._connection_manager.close()
            self._initialized = False
            logger.info(f"Disconnected from exchange: {self.config.exchange}")
    
    async def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the provider and available data"""
        if not self._initialized:
            await self.connect()
        
        metadata = {
            "exchange": self.config.exchange,
            "symbols": self.config.symbols,
            "timeframes": [],
            "has_ohlcv": False,
            "has_ticker": False,
            "has_orderbook": False,
            "connection_stats": {}
        }
        
        try:
            async with self._connection_manager.acquire() as exchange:
                # Get exchange capabilities
                metadata["timeframes"] = list(exchange.timeframes.keys()) if hasattr(exchange, 'timeframes') else []
                metadata["has_ohlcv"] = exchange.has.get('fetchOHLCV', False)
                metadata["has_ticker"] = exchange.has.get('fetchTicker', False)
                metadata["has_orderbook"] = exchange.has.get('fetchOrderBook', False)
                
                # Get connection statistics
                metadata["connection_stats"] = self._connection_manager.get_stats()
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
        
        return metadata
    
    async def fetch_data(self, **params) -> AsyncIterator[Dict[str, Any]]:
        """
        Fetch OHLCV data from the exchange.
        
        Parameters:
            symbols (List[str], optional): Symbols to fetch (defaults to config)
            timeframe (str, optional): Timeframe (defaults to config)
            since (Union[int, datetime], optional): Start time
            limit (int, optional): Number of candles to fetch
            
        Yields:
            Dict[str, Any]: OHLCV data points as PriceBar dict
        """
        if not self._initialized:
            await self.connect()
        
        # Get parameters with defaults from config
        symbols = params.get('symbols', self.config.symbols)
        timeframe = params.get('timeframe', self.config.timeframe)
        since = params.get('since')
        limit = params.get('limit', 100)
        
        # Convert datetime to timestamp if needed
        if isinstance(since, datetime):
            since = int(since.timestamp() * 1000)  # CCXT uses milliseconds
        
        for symbol in symbols:
            # Check cache for recent data
            cache_key = f"{symbol}_{timeframe}_{since}_{limit}"
            
            if (
                cache_key in self._cache and
                cache_key in self._cache_timestamps and
                (datetime.now() - self._cache_timestamps[cache_key]).total_seconds() < self.config.cache_duration
            ):
                # Use cached data
                for bar in self._cache[cache_key]:
                    yield bar
                continue
            
            # Fetch from exchange with retry
            try:
                ohlcv_data = await self._connection_manager.retry(
                    lambda exchange: exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                )
                
                # Format and cache data
                formatted_data = []
                for candle in ohlcv_data:
                    price_bar = self._format_ohlcv(symbol, candle)
                    if price_bar:
                        formatted_data.append(price_bar)
                        yield price_bar
                
                # Update cache
                self._cache[cache_key] = formatted_data
                self._cache_timestamps[cache_key] = datetime.now()
                
            except RateLimitError as e:
                # Re-raise with provider info
                raise RateLimitError(
                    str(e),
                    provider_id=self.config.name,
                    provider_type=self.config.type.value,
                    retry_after=e.retry_after
                )
            except Exception as e:
                # Handle various CCXT exceptions
                if isinstance(e, ccxt.NetworkError):
                    logger.warning(f"Network error fetching {symbol}: {e}")
                    raise DataFetchError(
                        f"Network error: {e}",
                        provider_id=self.config.name,
                        provider_type=self.config.type.value,
                        symbol=symbol,
                        original_error=e
                    )
                elif isinstance(e, ccxt.AuthenticationError):
                    logger.error(f"Authentication error for {symbol}: {e}")
                    raise AuthenticationError(
                        f"Authentication failed: {e}",
                        provider_id=self.config.name,
                        provider_type=self.config.type.value
                    )
                elif isinstance(e, ccxt.ExchangeNotAvailable):
                    logger.warning(f"Exchange not available for {symbol}: {e}")
                    raise ConnectionError(f"Exchange not available: {e}")
                else:
                    logger.error(f"Error fetching {symbol}: {e}")
                    raise DataFetchError(
                        f"Failed to fetch data: {e}",
                        provider_id=self.config.name,
                        provider_type=self.config.type.value,
                        symbol=symbol,
                        original_error=e
                    )
    
    def _format_ohlcv(self, symbol: str, ohlcv: List) -> Dict[str, Any]:
        """Convert CCXT OHLCV format to PriceBar dict"""
        try:
            # CCXT OHLCV format: [timestamp, open, high, low, close, volume]
            timestamp, open_price, high, low, close, volume = ohlcv
            
            # Create PriceBar instance
            price_bar = PriceBar(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(timestamp / 1000),  # Convert ms to seconds
                open=float(open_price),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=float(volume),
                source=self.config.exchange
            )
            
            return price_bar.dict()
        except Exception as e:
            logger.error(f"Error formatting OHLCV data: {e}")
            return None
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict containing ticker information
        """
        if not self._initialized:
            await self.connect()
        
        try:
            ticker = await self._connection_manager.retry(
                lambda exchange: exchange.fetch_ticker(symbol)
            )
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise DataFetchError(
                f"Failed to fetch ticker: {e}",
                provider_id=self.config.name,
                provider_type=self.config.type.value,
                symbol=symbol,
                original_error=e
            )
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Fetch order book for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Depth of the order book
            
        Returns:
            Dict containing order book data
        """
        if not self._initialized:
            await self.connect()
        
        try:
            order_book = await self._connection_manager.retry(
                lambda exchange: exchange.fetch_order_book(symbol, limit)
            )
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise DataFetchError(
                f"Failed to fetch order book: {e}",
                provider_id=self.config.name,
                provider_type=self.config.type.value,
                symbol=symbol,
                original_error=e
            )