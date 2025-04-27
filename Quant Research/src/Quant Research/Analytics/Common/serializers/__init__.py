"""Type-specific serializers for different data structures."""

from ..core import register_serializer

# Import serializers
from .primitives import DateTimeSerializer, NumpySerializer
from .models import SignalSerializer
from .dataframes import DataFrameSerializer

# Register standard serializers
register_serializer(DateTimeSerializer())
register_serializer(NumpySerializer())
register_serializer(SignalSerializer())
register_serializer(DataFrameSerializer())

__all__ = [
    "DateTimeSerializer", "NumpySerializer", 
    "SignalSerializer", "DataFrameSerializer"
]