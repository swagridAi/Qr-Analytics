# src/quant_research/providers/onchain.py
import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, AsyncIterator, Optional, List, Union, Callable

import aiohttp
from pydantic import Field, validator

from ..core.config import ProviderConfig, ProviderType
from ..core.models import BlockchainMetric
from ..core.errors import (
    ConnectionError, DataFetchError, 
    RateLimitError, AuthenticationError
)
from .base import BaseProvider
from .connection_manager import ConnectionManager


logger = logging.getLogger(__name__)


class OnChainDataSource(str, Enum):
    """Supported blockchain data sources"""
    ETHERSCAN = "etherscan"
    GLASSNODE = "glassnode"


class BlockchainNetwork(str, Enum):
    """Supported blockchain networks"""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"


class MetricType(str, Enum):
    """Types of on-chain metrics"""
    TRANSACTIONS = "transactions"
    FEES = "fees"
    ADDRESSES = "addresses"
    SUPPLY = "supply"
    DEFI = "defi"
    NFT = "nft"
    MINING = "mining"
    EXCHANGE = "exchange"


class OnChainProviderConfig(ProviderConfig):
    """Configuration for blockchain on-chain data provider"""
    
    # Override defaults from base
    name: str = "onchain"
    type: ProviderType = ProviderType.BLOCKCHAIN
    env_prefix: str = "ONCHAIN"
    
    # On-chain specific settings
    data_source: OnChainDataSource = Field(
        default=OnChainDataSource.ETHERSCAN,
        description="Blockchain data source to use"
    )
    networks: List[BlockchainNetwork] = Field(
        default=[BlockchainNetwork.ETHEREUM],
        description="Blockchain networks to fetch data from"
    )
    metrics: List[MetricType] = Field(
        default=[MetricType.TRANSACTIONS],
        description="Types of metrics to collect"
    )
    
    # Authentication
    api_key: Optional[str] = Field(
        default=None, 
        description="API key for the data source"
    )
    
    # Rate limiting
    rate_limit_requests: int = Field(
        default=5, 
        description="Requests per second"
    )
    
    # Cache settings
    cache_duration: int = Field(
        default=300,  # 5 minutes
        description="Cache duration in seconds"
    )
    
    @validator('data_source')
    def validate_data_source(cls, v):
        """Validate data source"""
        if not isinstance(v, OnChainDataSource):
            try:
                return OnChainDataSource(v)
            except ValueError:
                valid_sources = [s.value for s in OnChainDataSource]
                raise ValueError(f"Invalid data source: {v}. Supported sources: {', '.join(valid_sources)}")
        return v

    @validator('networks')
    def validate_networks(cls, v):
        """Validate networks"""
        result = []
        for network in v:
            if not isinstance(network, BlockchainNetwork):
                try:
                    network = BlockchainNetwork(network)
                except ValueError:
                    valid_networks = [n.value for n in BlockchainNetwork]
                    raise ValueError(f"Invalid network: {network}. Supported networks: {', '.join(valid_networks)}")
            result.append(network)
        return result
    
    class Config:
        arbitrary_types_allowed = True


class OnChainProvider(BaseProvider[OnChainProviderConfig]):
    """
    Provider for blockchain on-chain data.
    
    Features:
    - Transaction metrics (count, volume, gas used)
    - Active addresses
    - Network hash rates
    - Fee metrics
    - DeFi protocol metrics
    - Exchange inflows/outflows
    
    Supports multiple data sources:
    - Etherscan API
    - Glassnode API
    """
    
    def __init__(self, config: OnChainProviderConfig):
        """Initialize the on-chain data provider"""
        self.config = config
        self._connection_manager = None
        self._cache = {}
        self._cache_timestamps = {}
        self._initialized = False
        
        # API endpoints
        self._endpoints = {
            OnChainDataSource.ETHERSCAN: "https://api.etherscan.io/api",
            OnChainDataSource.GLASSNODE: "https://api.glassnode.com/v1",
        }
        
        # Metric to endpoint mapping for each data source
        self._metric_endpoints = {
            OnChainDataSource.ETHERSCAN: {
                MetricType.TRANSACTIONS: "?module=stats&action=dailytx",
                MetricType.FEES: "?module=proxy&action=eth_gasPrice",
                MetricType.SUPPLY: "?module=stats&action=ethsupply",
                MetricType.ADDRESSES: "?module=stats&action=dailyaddresscount",
            },
            OnChainDataSource.GLASSNODE: {
                MetricType.TRANSACTIONS: "metrics/transactions/count",
                MetricType.FEES: "metrics/fees/volume_sum",
                MetricType.ADDRESSES: "metrics/addresses/active_count",
                MetricType.SUPPLY: "metrics/supply/current",
                MetricType.DEFI: "metrics/protocols/total_value_locked",
                MetricType.MINING: "metrics/mining/hash_rate_mean",
                MetricType.EXCHANGE: "metrics/distribution/exchange_inflow",
            }
        }
    
    async def _create_session(self) -> aiohttp.ClientSession:
        """Create an HTTP session for API requests"""
        timeout = aiohttp.ClientTimeout(total=self.config.connection.timeout)
        return aiohttp.ClientSession(timeout=timeout)
    
    async def _is_session_healthy(self, session: aiohttp.ClientSession) -> bool:
        """Check if the session is healthy"""
        try:
            # Test request to check connectivity
            if self.config.data_source == OnChainDataSource.ETHERSCAN:
                url = f"{self._endpoints[self.config.data_source]}?module=proxy&action=eth_blockNumber&apikey={self.config.api_key}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("status") == "1" or "result" in data
            else:  # Glassnode
                url = f"{self._endpoints[self.config.data_source]}/metrics/market/price_usd_close"
                params = {
                    "a": "BTC",
                    "api_key": self.config.api_key,
                    "i": "24h"
                }
                async with session.get(url, params=params) as response:
                    return response.status == 200
            
            return False
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def _cleanup_session(self, session: aiohttp.ClientSession) -> None:
        """Clean up session resources"""
        try:
            await session.close()
        except Exception as e:
            logger.warning(f"Error closing session: {e}")
    
    async def connect(self) -> None:
        """Establish connection to the data source"""
        if self._initialized:
            return
        
        # Check if API key is provided
        if not self.config.api_key:
            raise AuthenticationError(
                "API key is required for on-chain data sources",
                provider_id=self.config.name,
                provider_type=self.config.type.value
            )
        
        # Initialize connection manager
        self._connection_manager = ConnectionManager(
            connection_factory=self._create_session,
            config=self.config.connection,
            health_check=self._is_session_healthy,
            cleanup=self._cleanup_session
        )
        
        await self._connection_manager.initialize()
        
        self._initialized = True
        logger.info(f"Connected to {self.config.data_source} on-chain data source")
    
    async def is_connected(self) -> bool:
        """Check if provider is connected"""
        if not self._initialized or not self._connection_manager:
            return False
        
        try:
            async with self._connection_manager.acquire() as session:
                return await self._is_session_healthy(session)
        except Exception:
            return False
    
    async def disconnect(self) -> None:
        """Close connection and release resources"""
        if self._connection_manager:
            await self._connection_manager.close()
            self._initialized = False
            logger.info(f"Disconnected from {self.config.data_source} on-chain data source")
    
    async def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the provider and available data"""
        if not self._initialized:
            await self.connect()
        
        metadata = {
            "data_source": self.config.data_source,
            "networks": [n.value for n in self.config.networks],
            "available_metrics": {},
            "connection_stats": {}
        }
        
        # Get available metrics for the data source
        for metric_type in MetricType:
            if metric_type.value in self._metric_endpoints.get(self.config.data_source, {}):
                metadata["available_metrics"][metric_type.value] = True
            else:
                metadata["available_metrics"][metric_type.value] = False
        
        # Get connection statistics
        if self._connection_manager:
            metadata["connection_stats"] = self._connection_manager.get_stats()
        
        return metadata
    
    async def fetch_data(self, **params) -> AsyncIterator[Dict[str, Any]]:
        """
        Fetch on-chain data from the provider.
        
        Parameters:
            network (str, optional): Blockchain network (defaults to config)
            metric_type (str, optional): Type of metric to fetch (defaults to config)
            start_time (Union[int, datetime], optional): Start time
            end_time (Union[int, datetime], optional): End time
            limit (int, optional): Number of data points to fetch
            
        Yields:
            Dict[str, Any]: On-chain data points
        """
        if not self._initialized:
            await self.connect()
        
        # Get parameters with defaults from config
        network = params.get('network')
        if network is None and self.config.networks:
            network = self.config.networks[0].value
        
        metric_type = params.get('metric_type')
        if metric_type is None and self.config.metrics:
            metric_type = self.config.metrics[0].value
        
        # Parse metric type
        try:
            metric_enum = MetricType(metric_type)
        except ValueError:
            valid_metrics = [m.value for m in MetricType]
            raise ValueError(f"Invalid metric type: {metric_type}. Supported metrics: {', '.join(valid_metrics)}")
        
        # Parse network
        try:
            network_enum = BlockchainNetwork(network)
        except ValueError:
            valid_networks = [n.value for n in BlockchainNetwork]
            raise ValueError(f"Invalid network: {network}. Supported networks: {', '.join(valid_networks)}")
        
        # Get other parameters
        start_time = params.get('start_time')
        end_time = params.get('end_time', datetime.now())
        limit = params.get('limit', 100)
        
        # Convert datetime to timestamp if needed
        if isinstance(start_time, datetime):
            start_time = int(start_time.timestamp())
        if isinstance(end_time, datetime):
            end_time = int(end_time.timestamp())
        
        # Default start time if not provided (24 hours ago)
        if start_time is None:
            start_time = int((datetime.now() - timedelta(days=1)).timestamp())
        
        # Check cache for recent data
        cache_key = f"{network}_{metric_type}_{start_time}_{end_time}_{limit}"
        
        if (
            cache_key in self._cache and
            cache_key in self._cache_timestamps and
            (datetime.now() - self._cache_timestamps[cache_key]).total_seconds() < self.config.cache_duration
        ):
            # Use cached data
            for item in self._cache[cache_key]:
                yield item
            return
        
        # Determine the fetch method based on data source and metric
        if self.config.data_source == OnChainDataSource.ETHERSCAN:
            data = await self._fetch_etherscan_data(network_enum, metric_enum, start_time, end_time, limit)
        elif self.config.data_source == OnChainDataSource.GLASSNODE:
            data = await self._fetch_glassnode_data(network_enum, metric_enum, start_time, end_time, limit)
        else:
            raise ValueError(f"Unsupported data source: {self.config.data_source}")
        
        # Format and cache data
        formatted_data = []
        for item in data:
            blockchain_metric = self._format_metric(network_enum, metric_enum, item)
            if blockchain_metric:
                formatted_data.append(blockchain_metric)
                yield blockchain_metric
        
        # Update cache
        self._cache[cache_key] = formatted_data
        self._cache_timestamps[cache_key] = datetime.now()
    
    async def _fetch_etherscan_data(
        self, 
        network: BlockchainNetwork,
        metric_type: MetricType,
        start_time: int,
        end_time: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Fetch data from Etherscan API"""
        # Only support Ethereum network for Etherscan
        if network != BlockchainNetwork.ETHEREUM:
            raise ValueError(f"Etherscan only supports Ethereum network, got {network}")
        
        # Get the endpoint for the metric
        if metric_type.value not in self._metric_endpoints.get(OnChainDataSource.ETHERSCAN, {}):
            raise ValueError(f"Metric type {metric_type} not supported by Etherscan")
        
        endpoint = self._metric_endpoints[OnChainDataSource.ETHERSCAN][metric_type]
        base_url = self._endpoints[OnChainDataSource.ETHERSCAN]
        
        # Add date parameters if the endpoint supports them
        if metric_type in [MetricType.TRANSACTIONS, MetricType.ADDRESSES]:
            endpoint += f"&startdate={datetime.fromtimestamp(start_time).strftime('%Y-%m-%d')}"
            endpoint += f"&enddate={datetime.fromtimestamp(end_time).strftime('%Y-%m-%d')}"
        
        # Add API key
        endpoint += f"&apikey={self.config.api_key}"
        
        url = f"{base_url}{endpoint}"
        
        # Fetch data with retry
        try:
            response_data = await self._connection_manager.retry(
                lambda session: self._make_request(session, url)
            )
            
            if response_data.get("status") != "1" and "result" not in response_data:
                error_msg = response_data.get("message", "Unknown error")
                raise DataFetchError(
                    f"Etherscan API error: {error_msg}",
                    provider_id=self.config.name,
                    provider_type=self.config.type.value
                )
            
            result = response_data.get("result", [])
            
            # Handle different response formats
            if isinstance(result, list):
                return result[:limit]
            elif isinstance(result, dict):
                return [result]
            else:
                return [{"value": result}]
            
        except Exception as e:
            logger.error(f"Error fetching Etherscan data: {e}")
            if "rate limit" in str(e).lower():
                raise RateLimitError(
                    str(e),
                    provider_id=self.config.name,
                    provider_type=self.config.type.value,
                    retry_after=60  # Default retry after 1 minute
                )
            raise DataFetchError(
                f"Failed to fetch Etherscan data: {e}",
                provider_id=self.config.name,
                provider_type=self.config.type.value,
                original_error=e
            )
    
    async def _fetch_glassnode_data(
        self, 
        network: BlockchainNetwork,
        metric_type: MetricType,
        start_time: int,
        end_time: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Fetch data from Glassnode API"""
        # Get the endpoint for the metric
        if metric_type.value not in self._metric_endpoints.get(OnChainDataSource.GLASSNODE, {}):
            raise ValueError(f"Metric type {metric_type} not supported by Glassnode")
        
        endpoint = self._metric_endpoints[OnChainDataSource.GLASSNODE][metric_type]
        base_url = self._endpoints[OnChainDataSource.GLASSNODE]
        
        # Map network to Glassnode asset
        network_map = {
            BlockchainNetwork.BITCOIN: "BTC",
            BlockchainNetwork.ETHEREUM: "ETH",
            BlockchainNetwork.POLYGON: "MATIC",
            BlockchainNetwork.ARBITRUM: "ARB",
            BlockchainNetwork.OPTIMISM: "OP"
        }
        
        if network not in network_map:
            raise ValueError(f"Network {network} not supported by Glassnode")
        
        asset = network_map[network]
        url = f"{base_url}/{endpoint}"
        
        # Prepare parameters
        params = {
            "a": asset,
            "s": start_time,
            "u": end_time,
            "i": "24h",  # Daily interval
            "api_key": self.config.api_key
        }
        
        # Fetch data with retry
        try:
            response_data = await self._connection_manager.retry(
                lambda session: self._make_request(session, url, params=params)
            )
            
            # Glassnode returns data as an array
            if isinstance(response_data, list):
                return response_data[:limit]
            else:
                raise DataFetchError(
                    f"Unexpected Glassnode response format",
                    provider_id=self.config.name,
                    provider_type=self.config.type.value
                )
            
        except Exception as e:
            logger.error(f"Error fetching Glassnode data: {e}")
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                raise RateLimitError(
                    str(e),
                    provider_id=self.config.name,
                    provider_type=self.config.type.value,
                    retry_after=60  # Default retry after 1 minute
                )
            raise DataFetchError(
                f"Failed to fetch Glassnode data: {e}",
                provider_id=self.config.name,
                provider_type=self.config.type.value,
                original_error=e
            )
    
    async def _make_request(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request to the API"""
        try:
            async with session.get(url, params=params) as response:
                # Check for rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise RateLimitError(
                        f"Rate limit exceeded",
                        provider_id=self.config.name,
                        provider_type=self.config.type.value,
                        retry_after=retry_after
                    )
                
                # Check for auth errors
                if response.status == 401 or response.status == 403:
                    raise AuthenticationError(
                        f"Authentication failed with status {response.status}",
                        provider_id=self.config.name,
                        provider_type=self.config.type.value
                    )
                
                # Check for other errors
                if response.status != 200:
                    raise DataFetchError(
                        f"API request failed with status {response.status}",
                        provider_id=self.config.name,
                        provider_type=self.config.type.value
                    )
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request error: {e}")
            raise ConnectionError(f"HTTP request failed: {e}")
    
    def _format_metric(
        self, 
        network: BlockchainNetwork, 
        metric_type: MetricType, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert API response to BlockchainMetric dict"""
        try:
            timestamp = None
            value = None
            
            # Extract timestamp and value based on data source and format
            if self.config.data_source == OnChainDataSource.ETHERSCAN:
                if "timeStamp" in data:
                    timestamp = datetime.fromtimestamp(int(data["timeStamp"]))
                elif "UTCDate" in data:
                    timestamp = datetime.strptime(data["UTCDate"], "%Y-%m-%d")
                else:
                    timestamp = datetime.now()
                
                if "value" in data:
                    value = float(data["value"])
                elif "result" in data:
                    value = float(data["result"])
                else:
                    # Find the first numeric field
                    for key, val in data.items():
                        try:
                            value = float(val)
                            break
                        except (ValueError, TypeError):
                            continue
            
            elif self.config.data_source == OnChainDataSource.GLASSNODE:
                if "t" in data:
                    timestamp = datetime.fromtimestamp(int(data["t"]))
                elif "timestamp" in data:
                    timestamp = datetime.fromtimestamp(int(data["timestamp"]))
                
                if "v" in data:
                    value = float(data["v"])
                elif "value" in data:
                    value = float(data["value"])
            
            if timestamp is None or value is None:
                logger.warning(f"Could not extract timestamp or value from data: {data}")
                return None
            
            # Create BlockchainMetric instance
            blockchain_metric = BlockchainMetric(
                network=network.value,
                metric_type=metric_type.value,
                timestamp=timestamp,
                value=value,
                source=self.config.data_source.value,
                additional_data=data
            )
            
            return blockchain_metric.dict()
            
        except Exception as e:
            logger.error(f"Error formatting metric data: {e}")
            return None
    
    async def fetch_transactions(
        self, 
        network: Optional[Union[str, BlockchainNetwork]] = None,
        start_time: Optional[Union[int, datetime]] = None,
        end_time: Optional[Union[int, datetime]] = None,
        limit: int = 100
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Fetch transaction count metrics.
        
        Args:
            network: Blockchain network
            start_time: Start time
            end_time: End time
            limit: Number of data points to fetch
            
        Yields:
            Dict[str, Any]: Transaction metrics
        """
        async for metric in self.fetch_data(
            network=network,
            metric_type=MetricType.TRANSACTIONS.value,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        ):
            yield metric
    
    async def fetch_fees(
        self, 
        network: Optional[Union[str, BlockchainNetwork]] = None,
        start_time: Optional[Union[int, datetime]] = None,
        end_time: Optional[Union[int, datetime]] = None,
        limit: int = 100
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Fetch fee metrics.
        
        Args:
            network: Blockchain network
            start_time: Start time
            end_time: End time
            limit: Number of data points to fetch
            
        Yields:
            Dict[str, Any]: Fee metrics
        """
        async for metric in self.fetch_data(
            network=network,
            metric_type=MetricType.FEES.value,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        ):
            yield metric
    
    async def fetch_addresses(
        self, 
        network: Optional[Union[str, BlockchainNetwork]] = None,
        start_time: Optional[Union[int, datetime]] = None,
        end_time: Optional[Union[int, datetime]] = None,
        limit: int = 100
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Fetch address metrics.
        
        Args:
            network: Blockchain network
            start_time: Start time
            end_time: End time
            limit: Number of data points to fetch
            
        Yields:
            Dict[str, Any]: Address metrics
        """
        async for metric in self.fetch_data(
            network=network,
            metric_type=MetricType.ADDRESSES.value,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        ):
            yield metric
    
    async def fetch_defi_metrics(
        self, 
        network: Optional[Union[str, BlockchainNetwork]] = None,
        start_time: Optional[Union[int, datetime]] = None,
        end_time: Optional[Union[int, datetime]] = None,
        limit: int = 100
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Fetch DeFi protocol metrics.
        
        Args:
            network: Blockchain network
            start_time: Start time
            end_time: End time
            limit: Number of data points to fetch
            
        Yields:
            Dict[str, Any]: DeFi metrics
        """
        # This is only available in Glassnode
        if self.config.data_source != OnChainDataSource.GLASSNODE:
            raise ValueError(f"DeFi metrics are only available from Glassnode, not {self.config.data_source}")
        
        async for metric in self.fetch_data(
            network=network,
            metric_type=MetricType.DEFI.value,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        ):
            yield metric