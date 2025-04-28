# src/quant_research/providers/equities_yf.py
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncIterator, Optional, List, Union, Callable

import yfinance as yf
import pandas as pd
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


class YahooFinanceProviderConfig(ProviderConfig):
    """Configuration for Yahoo Finance provider"""
    
    # Override defaults from base
    name: str = "equities_yf"
    type: ProviderType = ProviderType.EQUITY
    env_prefix: str = "YF"
    
    # Yahoo Finance-specific settings
    symbols: List[str] = Field(..., description="List of ticker symbols to track (e.g., 'AAPL', 'MSFT')")
    interval: str = Field(default="1d", description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)")
    include_dividends: bool = Field(default=True, description="Whether to include dividend data")
    include_splits: bool = Field(default=True, description="Whether to include split data")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=2000, description="Requests per hour (Yahoo Finance limit)")
    
    # Cache settings
    cache_duration: int = Field(default=300, description="Cache duration in seconds")
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """Validate symbol format"""
        for symbol in v:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError(f"Invalid symbol: {symbol}. Expected non-empty string.")
        return [s.upper() for s in v]  # Normalize to uppercase
    
    @validator('interval')
    def validate_interval(cls, v):
        """Validate interval format"""
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        if v not in valid_intervals:
            raise ValueError(f"Invalid interval: {v}. Valid options: {', '.join(valid_intervals)}")
        return v
    
    class Config:
        arbitrary_types_allowed = True


class YahooFinanceProvider(BaseProvider[YahooFinanceProviderConfig]):
    """
    Provider for equity market data using Yahoo Finance.
    
    Features:
    - Historical OHLCV data
    - Dividend and split information
    - Company information and fundamentals
    - Multiple timeframes from 1m to 3mo
    """
    
    def __init__(self, config: YahooFinanceProviderConfig):
        """Initialize the Yahoo Finance provider"""
        self.config = config
        self._tickers = {}
        self._connection_manager = None
        self._cache = {}
        self._cache_timestamps = {}
        self._initialized = False
    
    async def _create_connection(self) -> Dict[str, Any]:
        """Create a connection to Yahoo Finance (via yfinance)"""
        # yfinance doesn't require a persistent connection,
        # but we'll create tickers for each symbol
        tickers = {}
        for symbol in self.config.symbols:
            tickers[symbol] = yf.Ticker(symbol)
        
        # Return the ticker objects dictionary as our "connection"
        return {
            "tickers": tickers,
            "session": None  # Could store a requests.Session here if needed
        }
    
    async def _is_connection_healthy(self, connection: Dict[str, Any]) -> bool:
        """Check if YF connection is healthy by making a simple request"""
        try:
            # Pick the first ticker and try to get a basic info
            if not connection["tickers"]:
                return False
            
            symbol = next(iter(connection["tickers"]))
            ticker = connection["tickers"][symbol]
            
            # Run in an executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: ticker.info.get("regularMarketPrice") is not None
            )
            return result
        except Exception as e:
            logger.warning(f"Yahoo Finance health check failed: {e}")
            return False
    
    async def _cleanup_connection(self, connection: Dict[str, Any]) -> None:
        """Clean up YF connection resources"""
        # Not much to clean up for yfinance, but we'll close any session if present
        if connection.get("session"):
            try:
                session = connection["session"]
                if hasattr(session, "close"):
                    await asyncio.get_event_loop().run_in_executor(None, session.close)
            except Exception as e:
                logger.warning(f"Error closing Yahoo Finance session: {e}")
    
    async def connect(self) -> None:
        """Initialize the connection to Yahoo Finance"""
        if self._initialized:
            return
        
        # Initialize connection manager
        self._connection_manager = ConnectionManager(
            connection_factory=self._create_connection,
            config=self.config.connection,
            health_check=self._is_connection_healthy,
            cleanup=self._cleanup_connection
        )
        
        await self._connection_manager.initialize()
        self._initialized = True
        logger.info(f"Connected to Yahoo Finance for {len(self.config.symbols)} symbols")
    
    async def is_connected(self) -> bool:
        """Check if provider is connected and healthy"""
        if not self._initialized or not self._connection_manager:
            return False
        
        try:
            async with self._connection_manager.acquire() as connection:
                return await self._is_connection_healthy(connection)
        except Exception:
            return False
    
    async def disconnect(self) -> None:
        """Release resources"""
        if self._connection_manager:
            await self._connection_manager.close()
            self._initialized = False
            logger.info("Disconnected from Yahoo Finance")
    
    async def _run_yf_operation(self, func: Callable, *args, **kwargs) -> Any:
        """Run a yfinance operation in an executor to avoid blocking the event loop"""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        except Exception as e:
            logger.error(f"Error running Yahoo Finance operation: {e}")
            raise e
    
    async def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the provider and available data"""
        if not self._initialized:
            await self.connect()
        
        metadata = {
            "provider": "Yahoo Finance",
            "symbols": self.config.symbols,
            "interval": self.config.interval,
            "include_dividends": self.config.include_dividends,
            "include_splits": self.config.include_splits,
            "connection_stats": {}
        }
        
        try:
            # Get connection statistics
            if self._connection_manager:
                metadata["connection_stats"] = self._connection_manager.get_stats()
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
        
        return metadata
    
    async def fetch_data(self, **params) -> AsyncIterator[Dict[str, Any]]:
        """
        Fetch historical price data from Yahoo Finance.
        
        Parameters:
            symbols (List[str], optional): Symbols to fetch (defaults to config)
            interval (str, optional): Timeframe (defaults to config)
            start (Union[str, datetime], optional): Start date
            end (Union[str, datetime], optional): End date
            period (str, optional): Period as alternative to start/end (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Yields:
            Dict[str, Any]: OHLCV data points as PriceBar dict
        """
        if not self._initialized:
            await self.connect()
        
        # Get parameters with defaults from config
        symbols = params.get('symbols', self.config.symbols)
        interval = params.get('interval', self.config.interval)
        start = params.get('start')
        end = params.get('end')
        period = params.get('period', "1mo" if not start and not end else None)
        
        # Convert datetime objects to strings if needed
        if isinstance(start, datetime):
            start = start.strftime('%Y-%m-%d')
        if isinstance(end, datetime):
            end = end.strftime('%Y-%m-%d')
        
        for symbol in symbols:
            # Check cache for recent data
            cache_key = f"{symbol}_{interval}_{start}_{end}_{period}"
            
            if (
                cache_key in self._cache and
                cache_key in self._cache_timestamps and
                (datetime.now() - self._cache_timestamps[cache_key]).total_seconds() < self.config.cache_duration
            ):
                # Use cached data
                for bar in self._cache[cache_key]:
                    yield bar
                continue
            
            try:
                # Fetch from Yahoo Finance with retry
                hist_data = await self._connection_manager.retry(
                    lambda conn: self._fetch_history(
                        conn, symbol, interval, start, end, period
                    )
                )
                
                # Format and cache data
                formatted_data = []
                for idx, row in hist_data.iterrows():
                    price_bar = self._format_ohlcv(symbol, idx, row)
                    if price_bar:
                        formatted_data.append(price_bar)
                        yield price_bar
                
                # Update cache
                self._cache[cache_key] = formatted_data
                self._cache_timestamps[cache_key] = datetime.now()
                
            except Exception as e:
                # Handle exceptions
                if "Rate limit" in str(e):
                    logger.warning(f"Rate limit exceeded for {symbol}: {e}")
                    raise RateLimitError(
                        str(e),
                        provider_id=self.config.name,
                        provider_type=self.config.type.value,
                        retry_after=60  # Yahoo Finance doesn't provide specific retry-after
                    )
                elif "Invalid API call" in str(e) or "Not found" in str(e):
                    logger.error(f"API error for {symbol}: {e}")
                    raise DataFetchError(
                        f"API error: {e}",
                        provider_id=self.config.name,
                        provider_type=self.config.type.value,
                        symbol=symbol,
                        original_error=e
                    )
                else:
                    logger.error(f"Error fetching {symbol}: {e}")
                    raise DataFetchError(
                        f"Failed to fetch data: {e}",
                        provider_id=self.config.name,
                        provider_type=self.config.type.value,
                        symbol=symbol,
                        original_error=e
                    )
    
    async def _fetch_history(
        self, connection: Dict[str, Any], symbol: str, 
        interval: str, start: Optional[str], end: Optional[str], 
        period: Optional[str]
    ) -> pd.DataFrame:
        """Fetch historical data for a symbol"""
        # Get the ticker object
        ticker = connection["tickers"].get(symbol)
        if not ticker:
            # Create ticker if not in connection
            ticker = yf.Ticker(symbol)
            connection["tickers"][symbol] = ticker
        
        # Run the history fetch in an executor
        return await self._run_yf_operation(
            ticker.history,
            period=period,
            interval=interval,
            start=start,
            end=end,
            auto_adjust=True,  # Adjust for splits
            actions=True,      # Include dividends and splits
        )
    
    def _format_ohlcv(self, symbol: str, timestamp, row) -> Dict[str, Any]:
        """Convert Yahoo Finance data to PriceBar dict"""
        try:
            # Create PriceBar instance
            price_bar = PriceBar(
                symbol=symbol,
                timestamp=timestamp if isinstance(timestamp, datetime) else pd.Timestamp(timestamp).to_pydatetime(),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume']),
                source="yahoo_finance"
            )
            
            # Add additional fields if available
            result = price_bar.dict()
            if 'Dividends' in row and row['Dividends'] > 0:
                result['dividends'] = float(row['Dividends'])
            if 'Stock Splits' in row and row['Stock Splits'] > 0:
                result['splits'] = float(row['Stock Splits'])
            
            return result
        except Exception as e:
            logger.error(f"Error formatting OHLCV data for {symbol}: {e}")
            return None
    
    async def fetch_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch company information for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dict containing company information
        """
        if not self._initialized:
            await self.connect()
        
        symbol = symbol.upper()
        
        try:
            info = await self._connection_manager.retry(
                lambda conn: self._fetch_info(conn, symbol)
            )
            return info
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            raise DataFetchError(
                f"Failed to fetch company info: {e}",
                provider_id=self.config.name,
                provider_type=self.config.type.value,
                symbol=symbol,
                original_error=e
            )
    
    async def _fetch_info(self, connection: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Fetch company info for a symbol"""
        ticker = connection["tickers"].get(symbol)
        if not ticker:
            ticker = yf.Ticker(symbol)
            connection["tickers"][symbol] = ticker
        
        # Run the info fetch in an executor
        return await self._run_yf_operation(lambda: ticker.info)
    
    async def fetch_fundamentals(self, symbol: str, statement_type: str = "income") -> Dict[str, Any]:
        """
        Fetch fundamental financial data for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            statement_type: Type of financial statement ("income", "balance", "cash")
            
        Returns:
            Dict containing financial statement data
        """
        if not self._initialized:
            await self.connect()
        
        symbol = symbol.upper()
        statement_map = {
            "income": "income_stmt",
            "balance": "balance_sheet",
            "cash": "cashflow"
        }
        
        method_name = statement_map.get(statement_type)
        if not method_name:
            raise ValueError(f"Invalid statement type: {statement_type}. Valid options: income, balance, cash")
        
        try:
            financials = await self._connection_manager.retry(
                lambda conn: self._fetch_financials(conn, symbol, method_name)
            )
            return financials
        except Exception as e:
            logger.error(f"Error fetching {statement_type} for {symbol}: {e}")
            raise DataFetchError(
                f"Failed to fetch {statement_type}: {e}",
                provider_id=self.config.name,
                provider_type=self.config.type.value,
                symbol=symbol,
                original_error=e
            )
    
    async def _fetch_financials(self, connection: Dict[str, Any], symbol: str, method_name: str) -> Dict[str, Any]:
        """Fetch financial data for a symbol"""
        ticker = connection["tickers"].get(symbol)
        if not ticker:
            ticker = yf.Ticker(symbol)
            connection["tickers"][symbol] = ticker
        
        # Get the method to call
        method = getattr(ticker, method_name)
        
        # Run the financials fetch in an executor
        df = await self._run_yf_operation(method)
        
        # Convert DataFrame to dict format
        if isinstance(df, pd.DataFrame):
            return df.to_dict()
        return {}