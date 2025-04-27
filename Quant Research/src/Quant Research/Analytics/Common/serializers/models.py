"""Serializers for domain models."""

from datetime import datetime
from typing import Dict, Any, Type

from quant_research.core.models import Signal
from ..core import TypeSerializer

class SignalSerializer(TypeSerializer[Signal]):
    """Serializer for Signal objects."""
    
    @property
    def target_type(self) -> Type[Signal]:
        return Signal
    
    def serialize(self, obj: Signal) -> Dict[str, Any]:
        """
        Serialize Signal to dictionary.
        
        Args:
            obj: Signal object to serialize
            
        Returns:
            Dictionary representation of the Signal
        """
        return {
            "timestamp": obj.timestamp.isoformat(),
            "asset": obj.asset,
            "signal": obj.signal,
            "strength": float(obj.strength),
            "metadata": obj.metadata
        }
    
    def deserialize(self, data: Dict[str, Any]) -> Signal:
        """
        Deserialize dictionary to Signal.
        
        Args:
            data: Dictionary representation of a Signal
            
        Returns:
            Signal object
        """
        # Convert timestamp string to datetime
        if isinstance(data["timestamp"], str):
            timestamp = datetime.fromisoformat(data["timestamp"])
        else:
            timestamp = data["timestamp"]
            
        return Signal(
            timestamp=timestamp,
            asset=data["asset"],
            signal=data["signal"],
            strength=data["strength"],
            metadata=data.get("metadata", {})
        )

# Add additional domain model serializers here as needed
# For example:
#
# class AnalyticsResultSerializer(TypeSerializer[AnalyticsResult]):
#     """Serializer for AnalyticsResult objects."""
#     
#     @property
#     def target_type(self) -> Type[AnalyticsResult]:
#         return AnalyticsResult
#     
#     def serialize(self, obj: AnalyticsResult) -> Dict[str, Any]:
#         """Serialize AnalyticsResult to dictionary."""
#         return {
#             "timestamp": obj.timestamp.isoformat(),
#             "value": obj.value,
#             "metadata": obj.metadata
#         }
#     
#     def deserialize(self, data: Dict[str, Any]) -> AnalyticsResult:
#         """Deserialize dictionary to AnalyticsResult."""
#         return AnalyticsResult(
#             timestamp=datetime.fromisoformat(data["timestamp"]),
#             value=data["value"],
#             metadata=data.get("metadata", {})
#         )