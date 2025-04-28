# src/quant_research/providers/telemetry.py
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable
import asyncio
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, field
import json
import statistics

from ..core.config import ProviderConfig
from ..core.errors import RateLimitError


logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single data point for a metric"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuotaInfo:
    """Information about API quota usage"""
    limit: int
    remaining: int
    reset_time: Optional[datetime] = None
    

class ProviderMetrics:
    """Tracks metrics for a specific provider"""
    
    def __init__(self, provider_id: str, max_history: int = 10000):
        """
        Initialize metrics tracking for a provider
        
        Args:
            provider_id: Identifier for the provider
            max_history: Maximum number of metric points to keep in history
        """
        self.provider_id = provider_id
        self.max_history = max_history
        
        # Metric histories
        self._latencies: List[MetricPoint] = []
        self._errors: List[MetricPoint] = []
        self._requests: List[MetricPoint] = []
        self._data_points: List[MetricPoint] = []
        
        # Summary statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        
        # Rate limit tracking
        self._quota_info: Optional[QuotaInfo] = None
        self._rate_limit_hits = 0
        
        # Last metrics update time
        self._last_update = datetime.now()
    
    def record_request(self, endpoint: str, success: bool, latency_ms: float, 
                      data_points: int = 0, error: Optional[Exception] = None) -> None:
        """
        Record metrics for a request
        
        Args:
            endpoint: API endpoint or method called
            success: Whether the request was successful
            latency_ms: Request latency in milliseconds
            data_points: Number of data points retrieved (if applicable)
            error: Exception that occurred (if failed)
        """
        now = datetime.now()
        self._last_update = now
        
        # Update summary stats
        self._total_requests += 1
        if success:
            self._successful_requests += 1
        else:
            self._failed_requests += 1
            if isinstance(error, RateLimitError):
                self._rate_limit_hits += 1
        
        # Record metrics with metadata
        metadata = {
            "endpoint": endpoint,
            "success": success,
        }
        
        if error:
            metadata["error_type"] = type(error).__name__
            metadata["error_msg"] = str(error)
        
        # Add metrics to histories (with size limits)
        self._latencies.append(MetricPoint(now, latency_ms, metadata.copy()))
        if len(self._latencies) > self.max_history:
            self._latencies = self._latencies[-self.max_history:]
        
        self._requests.append(MetricPoint(now, 1 if success else 0, metadata.copy()))
        if len(self._requests) > self.max_history:
            self._requests = self._requests[-self.max_history:]
        
        if not success and error:
            self._errors.append(MetricPoint(now, 1, metadata.copy()))
            if len(self._errors) > self.max_history:
                self._errors = self._errors[-self.max_history:]
        
        if data_points > 0:
            self._data_points.append(MetricPoint(now, data_points, metadata.copy()))
            if len(self._data_points) > self.max_history:
                self._data_points = self._data_points[-self.max_history:]
    
    def update_quota(self, limit: int, remaining: int, 
                    reset_time: Optional[datetime] = None) -> None:
        """
        Update quota information
        
        Args:
            limit: Total request limit
            remaining: Remaining requests allowed
            reset_time: When the quota will reset
        """
        self._quota_info = QuotaInfo(limit, remaining, reset_time)
        
        # Log if we're getting close to quota limits
        usage_percent = (limit - remaining) / limit * 100 if limit > 0 else 0
        if usage_percent > 80:
            logger.warning(
                f"Provider {self.provider_id} approaching quota limit: "
                f"{remaining}/{limit} remaining ({usage_percent:.1f}% used)"
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this provider"""
        success_rate = (
            self._successful_requests / self._total_requests * 100 
            if self._total_requests > 0 else 0
        )
        
        # Calculate recent latency statistics (last 100 requests)
        recent_latencies = [p.value for p in self._latencies[-100:]] if self._latencies else []
        
        latency_stats = {}
        if recent_latencies:
            try:
                latency_stats = {
                    "avg_ms": statistics.mean(recent_latencies),
                    "median_ms": statistics.median(recent_latencies),
                    "p95_ms": statistics.quantiles(recent_latencies, n=20)[-1] if len(recent_latencies) >= 20 else None,
                    "min_ms": min(recent_latencies),
                    "max_ms": max(recent_latencies)
                }
            except statistics.StatisticsError:
                # Handle potential statistics errors 
                latency_stats = {"error": "insufficient data for statistics"}
        
        quota_info = {}
        if self._quota_info:
            quota_info = {
                "limit": self._quota_info.limit,
                "remaining": self._quota_info.remaining,
                "usage_percent": (
                    (self._quota_info.limit - self._quota_info.remaining) / 
                    self._quota_info.limit * 100 if self._quota_info.limit > 0 else 0
                ),
                "reset_time": self._quota_info.reset_time.isoformat() 
                    if self._quota_info.reset_time else None
            }
        
        # Calculate request rate (requests per minute)
        recent_requests = [
            p for p in self._requests 
            if datetime.now() - p.timestamp < timedelta(minutes=5)
        ]
        
        requests_per_minute = len(recent_requests) / 5 if recent_requests else 0
        
        return {
            "provider_id": self.provider_id,
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate_percent": success_rate,
            "rate_limit_hits": self._rate_limit_hits,
            "requests_per_minute": requests_per_minute,
            "latency": latency_stats,
            "quota": quota_info,
            "last_update": self._last_update.isoformat(),
        }
    
    def export_metrics(self, time_window: Optional[timedelta] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Export detailed metrics for analysis
        
        Args:
            time_window: Optional time window to filter metrics (e.g., last hour)
            
        Returns:
            Dictionary of metric lists
        """
        now = datetime.now()
        
        def filter_by_time(points: List[MetricPoint]) -> List[Dict[str, Any]]:
            filtered = [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "value": p.value,
                    **p.metadata
                }
                for p in points
                if not time_window or (now - p.timestamp <= time_window)
            ]
            return filtered
        
        return {
            "latencies": filter_by_time(self._latencies),
            "errors": filter_by_time(self._errors),
            "requests": filter_by_time(self._requests),
            "data_points": filter_by_time(self._data_points),
        }


class TelemetryManager:
    """
    Central manager for provider telemetry
    
    This class tracks metrics across all providers and offers
    a unified interface for monitoring and reporting.
    """
    
    def __init__(self):
        """Initialize the telemetry manager"""
        self._metrics: Dict[str, ProviderMetrics] = {}
        self._lock = asyncio.Lock()
        
        # Global stats
        self._global_start_time = datetime.now()
    
    def get_metrics(self, provider_id: str) -> ProviderMetrics:
        """
        Get metrics for a specific provider, creating if necessary
        
        Args:
            provider_id: Provider identifier
            
        Returns:
            Provider metrics instance
        """
        if provider_id not in self._metrics:
            self._metrics[provider_id] = ProviderMetrics(provider_id)
        return self._metrics[provider_id]
    
    async def record_request_async(self, provider_id: str, endpoint: str, 
                                 success: bool, latency_ms: float,
                                 data_points: int = 0, 
                                 error: Optional[Exception] = None) -> None:
        """
        Record request metrics (async version)
        
        Args:
            provider_id: Provider identifier
            endpoint: API endpoint or method
            success: Whether the request succeeded
            latency_ms: Request latency in milliseconds
            data_points: Number of data points retrieved
            error: Exception that occurred (if failed)
        """
        async with self._lock:
            metrics = self.get_metrics(provider_id)
            metrics.record_request(endpoint, success, latency_ms, data_points, error)
    
    async def update_quota_async(self, provider_id: str, limit: int, 
                               remaining: int, 
                               reset_time: Optional[datetime] = None) -> None:
        """
        Update quota information (async version)
        
        Args:
            provider_id: Provider identifier
            limit: Total request limit
            remaining: Remaining requests allowed
            reset_time: When the quota will reset
        """
        async with self._lock:
            metrics = self.get_metrics(provider_id)
            metrics.update_quota(limit, remaining, reset_time)
    
    def get_provider_summary(self, provider_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a specific provider
        
        Args:
            provider_id: Provider identifier
            
        Returns:
            Dictionary of summary statistics
        """
        if provider_id not in self._metrics:
            return {"provider_id": provider_id, "error": "no metrics available"}
        
        return self._metrics[provider_id].get_summary()
    
    def get_all_provider_summaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary statistics for all providers
        
        Returns:
            Dictionary of provider summaries
        """
        return {
            provider_id: metrics.get_summary()
            for provider_id, metrics in self._metrics.items()
        }
    
    def get_global_summary(self) -> Dict[str, Any]:
        """
        Get global summary statistics across all providers
        
        Returns:
            Dictionary of global summary statistics
        """
        total_requests = sum(
            metrics.get_summary()["total_requests"] 
            for metrics in self._metrics.values()
        )
        
        total_successful = sum(
            metrics.get_summary()["successful_requests"] 
            for metrics in self._metrics.values()
        )
        
        total_failed = sum(
            metrics.get_summary()["failed_requests"] 
            for metrics in self._metrics.values()
        )
        
        global_success_rate = (
            total_successful / total_requests * 100 
            if total_requests > 0 else 0
        )
        
        uptime = datetime.now() - self._global_start_time
        uptime_seconds = uptime.total_seconds()
        
        return {
            "total_providers": len(self._metrics),
            "total_requests": total_requests,
            "total_successful_requests": total_successful,
            "total_failed_requests": total_failed,
            "global_success_rate_percent": global_success_rate,
            "start_time": self._global_start_time.isoformat(),
            "uptime_seconds": uptime_seconds,
            "requests_per_second": total_requests / uptime_seconds if uptime_seconds > 0 else 0,
        }
    
    async def export_provider_metrics(self, provider_id: str, 
                                    time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Export detailed metrics for a provider
        
        Args:
            provider_id: Provider identifier
            time_window: Optional time window to filter metrics
            
        Returns:
            Detailed metrics in exportable format
        """
        if provider_id not in self._metrics:
            return {"provider_id": provider_id, "error": "no metrics available"}
        
        async with self._lock:
            metrics = self._metrics[provider_id]
            return {
                "provider_id": provider_id,
                "summary": metrics.get_summary(),
                "metrics": metrics.export_metrics(time_window),
            }
    
    async def save_metrics_to_file(self, filename: str) -> None:
        """
        Save all metrics to a JSON file
        
        Args:
            filename: Path to save the metrics file
        """
        data = {
            "global": self.get_global_summary(),
            "providers": self.get_all_provider_summaries(),
            "export_time": datetime.now().isoformat(),
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Metrics saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save metrics to {filename}: {e}")
    
    async def periodic_metric_saver(self, filename_pattern: str, interval_seconds: int = 3600) -> None:
        """
        Periodically save metrics to files
        
        Args:
            filename_pattern: Pattern for filenames (should include {} for timestamp)
            interval_seconds: Interval between saves in seconds
        """
        while True:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = filename_pattern.format(timestamp)
                await self.save_metrics_to_file(filename)
            except Exception as e:
                logger.error(f"Error in periodic metric saver: {e}")
            
            await asyncio.sleep(interval_seconds)


# Global telemetry manager instance
_telemetry_manager = TelemetryManager()

def get_telemetry_manager() -> TelemetryManager:
    """Get the global telemetry manager instance"""
    return _telemetry_manager


def with_telemetry(func):
    """
    Decorator to add telemetry tracking to provider methods
    
    This decorator will track latency, success/failure, and data points
    for the decorated method.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'config') or not hasattr(self.config, 'name'):
            # Not a provider or missing config
            return await func(self, *args, **kwargs)
        
        provider_id = self.config.name
        endpoint = func.__name__
        telemetry = get_telemetry_manager()
        
        start_time = time.time()
        error = None
        success = False
        data_points = 0
        
        try:
            result = await func(self, *args, **kwargs)
            success = True
            
            # Try to count data points if it's an iterable result
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
                try:
                    data_points = len(result)
                except (TypeError, ValueError):
                    pass
            
            return result
        
        except Exception as e:
            error = e
            raise
        
        finally:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Use asyncio.create_task to avoid blocking
            asyncio.create_task(
                telemetry.record_request_async(
                    provider_id, endpoint, success, latency_ms, 
                    data_points, error
                )
            )
    
    return wrapper


def extract_quota_headers(headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract quota information from response headers
    
    This attempts to handle common API quota header formats.
    Can be extended for specific provider patterns.
    
    Args:
        headers: Response headers
        
    Returns:
        Extracted quota information
    """
    quota_info = {}
    
    # Common header patterns
    limit_keys = ['X-RateLimit-Limit', 'X-Rate-Limit-Limit', 'RateLimit-Limit', 'X-Limit']
    remaining_keys = ['X-RateLimit-Remaining', 'X-Rate-Limit-Remaining', 
                     'RateLimit-Remaining', 'X-Remaining']
    reset_keys = ['X-RateLimit-Reset', 'X-Rate-Limit-Reset', 'RateLimit-Reset', 'X-Reset']
    
    # Case insensitive lookup
    headers_lower = {k.lower(): v for k, v in headers.items()}
    
    # Try to find limit
    for key in limit_keys:
        if key.lower() in headers_lower:
            try:
                quota_info['limit'] = int(headers_lower[key.lower()])
                break
            except (ValueError, TypeError):
                pass
    
    # Try to find remaining
    for key in remaining_keys:
        if key.lower() in headers_lower:
            try:
                quota_info['remaining'] = int(headers_lower[key.lower()])
                break
            except (ValueError, TypeError):
                pass
    
    # Try to find reset time
    for key in reset_keys:
        if key.lower() in headers_lower:
            try:
                # Could be timestamp or seconds until reset
                reset_val = headers_lower[key.lower()]
                
                if isinstance(reset_val, (int, float)) or reset_val.isdigit():
                    reset_val = float(reset_val)
                    
                    # If it's a small number, it's probably seconds until reset
                    if reset_val < 10000:
                        quota_info['reset_time'] = datetime.now() + timedelta(seconds=reset_val)
                    else:
                        # Otherwise it's a timestamp
                        quota_info['reset_time'] = datetime.fromtimestamp(reset_val)
                
                break
            except (ValueError, TypeError):
                pass
    
    return quota_info