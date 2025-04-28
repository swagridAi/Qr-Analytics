# src/quant_research/providers/config_validators.py
import re
import ipaddress
import logging
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime
import ccxt
from pydantic import validator, root_validator, Field

from ..core.config import ProviderConfig, ProviderType


logger = logging.getLogger(__name__)


class CCXTProviderConfig(ProviderConfig):
    """Enhanced configuration for CCXT provider with rigorous validation"""
    
    # Override defaults from base
    name: str = "crypto_ccxt"
    type: ProviderType = ProviderType.CRYPTO
    env_prefix: str = "CCXT_"
    require_auth: bool = True
    
    # CCXT-specific settings
    exchange: str = Field(..., description="Exchange ID (e.g., 'binance', 'coinbase')")
    symbols: List[str] = Field(..., description="List of symbols to track (e.g., 'BTC/USDT')")
    timeframe: str = Field(default="1m", description="Timeframe for OHLCV data")
    
    # Authentication
    api_keys: List[str] = Field(
        default_factory=lambda: ["API_KEY", "API_SECRET"],
        description="Required API key environment variables"
    )
    
    # Rate limiting
    rate_limit: Dict[str, Any] = Field(
        default_factory=lambda: {
            "requests_per_second": 5.0, 
            "burst": 10
        },
        description="Rate limiting configuration"
    )
    
    # Additional settings
    sandbox_mode: bool = Field(
        default=False, 
        description="Whether to use exchange sandbox mode"
    )
    default_limit: int = Field(
        default=100, 
        ge=1, 
        le=1000, 
        description="Default limit for data fetching"
    )
    fetch_ticker: bool = Field(
        default=True, 
        description="Whether to fetch ticker data"
    )
    fetch_ohlcv: bool = Field(
        default=True, 
        description="Whether to fetch OHLCV data"
    )
    fetch_order_book: bool = Field(
        default=False, 
        description="Whether to fetch order book data"
    )
    
    @validator('exchange')
    def validate_exchange(cls, v):
        """Validate exchange ID against CCXT supported exchanges"""
        if v not in ccxt.exchanges:
            supported = ", ".join(sorted(ccxt.exchanges)[:10]) + "..."
            raise ValueError(
                f"Invalid exchange: '{v}'. Not in list of supported CCXT exchanges. "
                f"Examples of supported exchanges: {supported}"
            )
        return v
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """Validate trading pair symbols format"""
        invalid_symbols = []
        
        for symbol in v:
            # Check basic format (e.g., 'BTC/USDT')
            if '/' not in symbol:
                invalid_symbols.append(f"{symbol} (missing '/')")
                continue
            
            base, quote = symbol.split('/')
            
            # Check that base and quote are non-empty
            if not base or not quote:
                invalid_symbols.append(f"{symbol} (empty base or quote)")
                continue
            
            # Check for common cryptocurrency format errors
            if not re.match(r'^[A-Za-z0-9._-]+$', base) or not re.match(r'^[A-Za-z0-9._-]+$', quote):
                invalid_symbols.append(f"{symbol} (invalid characters)")
                
        if invalid_symbols:
            raise ValueError(
                f"Invalid symbol formats: {', '.join(invalid_symbols)}. "
                f"Expected format: 'BTC/USDT'"
            )
        
        return v
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        """Validate timeframe format"""
        # Common timeframes in CCXT
        valid_timeframes = {
            '1m', '3m', '5m', '15m', '30m',  # minutes
            '1h', '2h', '4h', '6h', '8h', '12h',  # hours
            '1d', '3d',  # days
            '1w', '1M'  # week, month
        }
        
        if v not in valid_timeframes:
            raise ValueError(
                f"Invalid timeframe: '{v}'. "
                f"Valid timeframes: {', '.join(sorted(valid_timeframes))}"
            )
        
        return v
    
    @root_validator
    def validate_exchange_capabilities(cls, values):
        """Validate that requested operations are supported by the exchange"""
        exchange_id = values.get('exchange')
        fetch_ohlcv = values.get('fetch_ohlcv', True)
        fetch_order_book = values.get('fetch_order_book', False)
        
        # Skip validation if exchange is not specified
        if not exchange_id:
            return values
        
        try:
            # Check exchange capabilities without creating a full instance
            exchange_class = getattr(ccxt, exchange_id)
            exchange_has = exchange_class.has if hasattr(exchange_class, 'has') else {}
            
            # Validate OHLCV capability
            if fetch_ohlcv and not exchange_has.get('fetchOHLCV', False):
                logger.warning(
                    f"Exchange '{exchange_id}' does not support OHLCV data fetching "
                    f"but fetch_ohlcv is set to True. This operation will fail."
                )
            
            # Validate order book capability
            if fetch_order_book and not exchange_has.get('fetchOrderBook', False):
                logger.warning(
                    f"Exchange '{exchange_id}' does not support order book fetching "
                    f"but fetch_order_book is set to True. This operation will fail."
                )
                
        except AttributeError:
            # This should never happen if exchange validation passed, but just in case
            logger.warning(f"Could not verify capabilities for exchange '{exchange_id}'")
        
        return values
    
    @root_validator
    def validate_symbols_availability(cls, values):
        """Check that symbols exist on the exchange (if possible without making API calls)"""
        # This is a light validation that doesn't require API calls
        # A more thorough validation would happen when the provider connects
        
        exchange_id = values.get('exchange')
        symbols = values.get('symbols', [])
        
        if not exchange_id or not symbols:
            return values
            
        # Just log a reminder that symbols will be validated on connection
        logger.info(
            f"Symbols {symbols} will be validated against exchange '{exchange_id}' "
            f"when the provider connects"
        )
        
        return values


class YahooFinanceConfig(ProviderConfig):
    """Configuration for Yahoo Finance provider with enhanced validation"""
    
    # Override defaults from base
    name: str = "equities_yf"
    type: ProviderType = ProviderType.EQUITIES
    env_prefix: str = "YF_"
    
    # YF-specific settings
    symbols: List[str] = Field(..., description="List of symbols to track (e.g., 'AAPL')")
    period: str = Field(default="1d", description="Data interval period")
    start_date: Optional[Union[str, datetime]] = Field(
        default=None, 
        description="Start date for historical data"
    )
    end_date: Optional[Union[str, datetime]] = Field(
        default=None, 
        description="End date for historical data"
    )
    
    # Additional settings
    adjust_prices: bool = Field(
        default=True, 
        description="Whether to adjust prices for splits and dividends"
    )
    include_dividends: bool = Field(
        default=False, 
        description="Whether to include dividend information"
    )
    include_fundamentals: bool = Field(
        default=False, 
        description="Whether to include fundamental data"
    )
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """Validate stock symbols format"""
        invalid_symbols = []
        
        for symbol in v:
            # Check basic format for stock symbols
            if not re.match(r'^[A-Za-z0-9.^-]+$', symbol):
                invalid_symbols.append(f"{symbol} (invalid characters)")
                continue
                
            # Common error: URLs instead of symbols
            if '/' in symbol or ':' in symbol or '.com' in symbol:
                invalid_symbols.append(f"{symbol} (appears to be a URL, not a symbol)")
                continue
                
        if invalid_symbols:
            raise ValueError(
                f"Invalid symbol formats: {', '.join(invalid_symbols)}. "
                f"Expected format: 'AAPL', 'MSFT', etc."
            )
        
        return v
    
    @validator('period')
    def validate_period(cls, v):
        """Validate data period format"""
        valid_periods = {'1d', '5d', '1wk', '1mo', '3mo'}
        
        if v not in valid_periods:
            raise ValueError(
                f"Invalid period: '{v}'. "
                f"Valid periods: {', '.join(sorted(valid_periods))}"
            )
        
        return v
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        """Validate and normalize date formats"""
        if v is None:
            return v
            
        if isinstance(v, datetime):
            return v
            
        if isinstance(v, str):
            try:
                # Try ISO format first
                return datetime.fromisoformat(v)
            except ValueError:
                try:
                    # Try common date formats
                    formats = ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d-%m-%Y']
                    for fmt in formats:
                        try:
                            return datetime.strptime(v, fmt)
                        except ValueError:
                            continue
                    
                    # If we get here, none of the formats worked
                    raise ValueError()
                except ValueError:
                    raise ValueError(
                        f"Invalid date format: '{v}'. "
                        f"Please use ISO format (YYYY-MM-DD) or common date formats"
                    )
        
        raise ValueError(f"Date must be a string or datetime object: {v}")
    
    @root_validator
    def validate_date_range(cls, values):
        """Validate that start_date is before end_date"""
        start_date = values.get('start_date')
        end_date = values.get('end_date')
        
        if start_date and end_date and start_date > end_date:
            raise ValueError(
                f"Start date ({start_date}) must be before end date ({end_date})"
            )
        
        return values


class OnChainConfig(ProviderConfig):
    """Configuration for blockchain data provider with enhanced validation"""
    
    # Override defaults from base
    name: str = "onchain"
    type: ProviderType = ProviderType.ONCHAIN
    env_prefix: str = "ONCHAIN_"
    require_auth: bool = True
    
    # API sources
    api_source: str = Field(
        default="etherscan", 
        description="API source (etherscan, glassnode, etc.)"
    )
    
    # Authentication
    api_keys: List[str] = Field(
        default_factory=lambda: ["API_KEY"],
        description="Required API key environment variables"
    )
    
    # Blockchain settings
    network: str = Field(
        default="ethereum", 
        description="Blockchain network"
    )
    
    # Data types to fetch
    fetch_transactions: bool = Field(
        default=True, 
        description="Whether to fetch transaction data"
    )
    fetch_contracts: bool = Field(
        default=False, 
        description="Whether to fetch contract data"
    )
    fetch_metrics: bool = Field(
        default=False, 
        description="Whether to fetch on-chain metrics"
    )
    
    # Addresses to track
    addresses: List[str] = Field(
        default_factory=list, 
        description="List of blockchain addresses to track"
    )
    
    @validator('api_source')
    def validate_api_source(cls, v):
        """Validate API source"""
        valid_sources = {'etherscan', 'glassnode', 'blockchair', 'blockstream', 'mempool'}
        
        if v not in valid_sources:
            raise ValueError(
                f"Invalid API source: '{v}'. "
                f"Valid sources: {', '.join(sorted(valid_sources))}"
            )
        
        return v
    
    @validator('network')
    def validate_network(cls, v):
        """Validate blockchain network"""
        valid_networks = {
            'ethereum', 'bitcoin', 'solana', 'arbitrum',
            'optimism', 'polygon', 'avalanche', 'bsc'
        }
        
        if v not in valid_networks:
            raise ValueError(
                f"Invalid blockchain network: '{v}'. "
                f"Valid networks: {', '.join(sorted(valid_networks))}"
            )
        
        return v
    
    @validator('addresses')
    def validate_addresses(cls, v, values):
        """Validate blockchain addresses"""
        if not v:
            return v
        
        network = values.get('network', 'ethereum')
        invalid_addresses = []
        
        for addr in v:
            # Basic format validation based on network
            if network == 'ethereum':
                # Ethereum address validation
                if not re.match(r'^0x[a-fA-F0-9]{40}$', addr):
                    invalid_addresses.append(f"{addr} (invalid format for Ethereum)")
            elif network == 'bitcoin':
                # Bitcoin address validation (basic)
                if not (re.match(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$', addr) or 
                       re.match(r'^bc1[ac-hj-np-z02-9]{39,59}$', addr)):
                    invalid_addresses.append(f"{addr} (invalid format for Bitcoin)")
            else:
                # Basic check for other networks
                if not re.match(r'^[a-zA-Z0-9]{30,}$', addr):
                    invalid_addresses.append(f"{addr} (invalid format)")
        
        if invalid_addresses:
            raise ValueError(
                f"Invalid blockchain addresses for {network}: {', '.join(invalid_addresses)}"
            )
        
        return v
    
    @root_validator
    def validate_api_compatibility(cls, values):
        """Validate that requested operations are compatible with the API source"""
        api_source = values.get('api_source', 'etherscan')
        network = values.get('network', 'ethereum')
        fetch_metrics = values.get('fetch_metrics', False)
        
        # Check API source and network compatibility
        ethereum_sources = {'etherscan', 'glassnode'}
        bitcoin_sources = {'blockchair', 'blockstream', 'mempool', 'glassnode'}
        
        if network == 'ethereum' and api_source not in ethereum_sources:
            logger.warning(
                f"API source '{api_source}' may not support Ethereum network. "
                f"Consider using one of: {', '.join(ethereum_sources)}"
            )
        
        if network == 'bitcoin' and api_source not in bitcoin_sources:
            logger.warning(
                f"API source '{api_source}' may not support Bitcoin network. "
                f"Consider using one of: {', '.join(bitcoin_sources)}"
            )
        
        # Check metrics availability
        if fetch_metrics and api_source != 'glassnode':
            logger.warning(
                f"On-chain metrics are primarily available through Glassnode, "
                f"but api_source is set to '{api_source}'. This operation may fail."
            )
        
        return values


class TwitterSentimentConfig(ProviderConfig):
    """Configuration for Twitter sentiment provider with enhanced validation"""
    
    # Override defaults from base
    name: str = "sentiment_twitter"
    type: ProviderType = ProviderType.SENTIMENT
    env_prefix: str = "TWITTER_"
    require_auth: bool = True
    
    # Authentication
    api_keys: List[str] = Field(
        default_factory=lambda: ["API_KEY", "API_SECRET", "ACCESS_TOKEN", "ACCESS_SECRET"],
        description="Required API key environment variables"
    )
    
    # Query settings
    keywords: List[str] = Field(
        default_factory=list, 
        description="Keywords to track"
    )
    hashtags: List[str] = Field(
        default_factory=list, 
        description="Hashtags to track"
    )
    accounts: List[str] = Field(
        default_factory=list, 
        description="Twitter accounts to track"
    )
    
    # Filter settings
    languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Languages to include (ISO 639-1 codes)"
    )
    exclude_retweets: bool = Field(
        default=True, 
        description="Whether to exclude retweets"
    )
    min_followers: int = Field(
        default=0, 
        ge=0, 
        description="Minimum followers for included accounts"
    )
    
    # Sentiment analysis
    sentiment_model: str = Field(
        default="transformers",
        description="Sentiment analysis model to use"
    )
    batch_size: int = Field(
        default=32, 
        ge=1, 
        le=256, 
        description="Batch size for sentiment analysis"
    )
    
    @validator('keywords', 'hashtags')
    def validate_query_terms(cls, v, values, **kwargs):
        """Validate search terms"""
        field_name = kwargs.get('field_name', 'terms')
        
        if not v:
            return v
        
        invalid_terms = []
        for term in v:
            # Check for empty or overly short terms
            if not term or len(term) < 2:
                invalid_terms.append(f"'{term}' (too short)")
                continue
            
            # Check for potentially problematic characters
            if re.search(r'[\'"\\]', term):
                invalid_terms.append(f"'{term}' (contains problematic characters)")
        
        if invalid_terms:
            raise ValueError(
                f"Invalid {field_name}: {', '.join(invalid_terms)}. "
                f"Terms should be at least 2 characters and avoid special characters."
            )
        
        return v
    
    @validator('accounts')
    def validate_accounts(cls, v):
        """Validate Twitter account handles"""
        if not v:
            return v
        
        invalid_accounts = []
        for account in v:
            # Remove @ prefix if present
            account_clean = account[1:] if account.startswith('@') else account
            
            # Check for valid Twitter handle format
            if not re.match(r'^[A-Za-z0-9_]{1,15}$', account_clean):
                invalid_accounts.append(
                    f"'{account}' (invalid format, should be 1-15 alphanumeric or underscore characters)"
                )
        
        if invalid_accounts:
            raise ValueError(
                f"Invalid Twitter handles: {', '.join(invalid_accounts)}. "
                f"Handles should match Twitter's format requirements."
            )
        
        # Normalize handles by adding @ prefix if missing
        return [f"@{a}" if not a.startswith('@') else a for a in v]
    
    @validator('languages')
    def validate_languages(cls, v):
        """Validate language codes"""
        valid_languages = {
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi', 'nl',
            'no', 'sv', 'fi', 'da', 'pl', 'hu', 'fa', 'he', 'ur', 'th'
        }
        
        invalid_langs = [lang for lang in v if lang not in valid_languages]
        
        if invalid_langs:
            raise ValueError(
                f"Invalid language codes: {', '.join(invalid_langs)}. "
                f"Please use ISO 639-1 codes (e.g., 'en' for English)."
            )
        
        return v
    
    @validator('sentiment_model')
    def validate_sentiment_model(cls, v):
        """Validate sentiment analysis model"""
        valid_models = {'transformers', 'vader', 'textblob', 'flair', 'custom'}
        
        if v not in valid_models:
            raise ValueError(
                f"Invalid sentiment model: '{v}'. "
                f"Valid models: {', '.join(sorted(valid_models))}"
            )
        
        return v
    
    @root_validator
    def validate_search_criteria(cls, values):
        """Validate that at least one search criterion is provided"""
        keywords = values.get('keywords', [])
        hashtags = values.get('hashtags', [])
        accounts = values.get('accounts', [])
        
        if not keywords and not hashtags and not accounts:
            raise ValueError(
                "At least one search criterion (keywords, hashtags, or accounts) must be provided"
            )
        
        return values