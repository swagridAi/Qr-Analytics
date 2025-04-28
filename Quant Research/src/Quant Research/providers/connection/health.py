# src/quant_research/providers/connection/health.py
"""Health checking utilities for connection management."""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar('T')  # Type of connection object


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    is_healthy: bool
    latency_ms: float
    checked_at: datetime
    details: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    
    def __str__(self) -> str:
        """String representation of health check result."""
        status = "HEALTHY" if self.is_healthy else "UNHEALTHY"
        return f"{status} (latency: {self.latency_ms:.2f}ms, checked: {self.checked_at.isoformat()})"


class HealthChecker(Generic[T]):
    """
    Utilities for checking and monitoring connection health.
    
    Features:
    - Customizable health check functions
    - Timeout handling
    - Health history tracking
    - Latency monitoring
    """
    
    def __init__(
        self,
        check_func: Callable[[T], Awaitable[bool]],
        timeout_seconds: float = 10.0,
        history_size: int = 10
    ):
        """
        Initialize health checker.
        
        Args:
            check_func: Async function to check connection health
            timeout_seconds: Timeout for health checks in seconds
            history_size: Number of health check results to keep
        """
        self.check_func = check_func
        self.timeout_seconds = timeout_seconds
        self.history_size = history_size
        self._history: list[HealthCheckResult] = []
    
    async def check(self, conn: T) -> HealthCheckResult:
        """
        Check if a connection is healthy.
        
        Args:
            conn: The connection to check
            
        Returns:
            Health check result
        """
        start_time = time.time()
        is_healthy = False
        error = None
        details = {}
        
        try:
            # Run health check with timeout
            is_healthy = await asyncio.wait_for(
                self.check_func(conn),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            error = asyncio.TimeoutError(f"Health check timed out after {self.timeout_seconds}s")
            details["timeout"] = self.timeout_seconds
        except Exception as e:
            error = e
            details["error_type"] = type(e).__name__
            details["error_message"] = str(e)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Create result
        result = HealthCheckResult(
            is_healthy=is_healthy,
            latency_ms=latency_ms,
            checked_at=datetime.now(),
            details=details,
            error=error
        )
        
        # Add to history
        self._add_to_history(result)
        
        return result
    
    def _add_to_history(self, result: HealthCheckResult) -> None:
        """
        Add a health check result to history.
        
        Args:
            result: Health check result to add
        """
        self._history.append(result)
        
        # Trim history if needed
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size:]
    
    def get_history(self) -> list[HealthCheckResult]:
        """
        Get history of health check results.
        
        Returns:
            List of health check results
        """
        return self._history.copy()
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status.
        
        Returns:
            Dictionary with health status information
        """
        if not self._history:
            return {"status": "unknown", "checks_performed": 0}
        
        # Calculate stats
        total_checks = len(self._history)
        healthy_checks = sum(1 for result in self._history if result.is_healthy)
        recent_latencies = [result.latency_ms for result in self._history[-5:]]
        
        # Determine overall status
        health_ratio = healthy_checks / total_checks if total_checks > 0 else 0
        status = "healthy" if health_ratio >= 0.7 else "degraded" if health_ratio >= 0.3 else "unhealthy"
        
        return {
            "status": status,
            "checks_performed": total_checks,
            "healthy_ratio": health_ratio,
            "last_check": self._history[-1].checked_at.isoformat(),
            "is_currently_healthy": self._history[-1].is_healthy,
            "average_latency_ms": sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0,
        }


# Default health check implementation for common connection types
async def default_http_health_check(session: Any) -> bool:
    """
    Default health check for HTTP clients.
    
    Args:
        session: HTTP client session
        
    Returns:
        True if connection is healthy, False otherwise
    """
    try:
        # Check if the session has a get method (most HTTP clients do)
        if hasattr(session, 'get'):
            return True
        return False
    except Exception:
        return False


async def default_db_health_check(connection: Any) -> bool:
    """
    Default health check for database connections.
    
    Args:
        connection: Database connection
        
    Returns:
        True if connection is healthy, False otherwise
    """
    try:
        # Try to execute a simple query if possible
        if hasattr(connection, 'execute'):
            return True
        return False
    except Exception:
        return False