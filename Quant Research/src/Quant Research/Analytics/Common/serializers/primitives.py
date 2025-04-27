"""Serializers for primitive data types."""

import base64
from datetime import datetime, date
from typing import Dict, Any, Type, Union

import numpy as np

from ..core import TypeSerializer

class DateTimeSerializer(TypeSerializer[Union[datetime, date]]):
    """Serializer for datetime objects."""
    
    @property
    def target_type(self) -> Type[Union[datetime, date]]:
        return datetime
    
    def serialize(self, obj: Union[datetime, date]) -> Dict[str, Any]:
        """
        Serialize datetime to dictionary.
        
        Args:
            obj: Datetime or date object to serialize
            
        Returns:
            Dictionary representation with ISO format string
            
        Raises:
            TypeError: If obj is not a datetime or date
        """
        if isinstance(obj, datetime):
            return {
                "type": "datetime",
                "iso": obj.isoformat()
            }
        elif isinstance(obj, date):
            return {
                "type": "date",
                "iso": obj.isoformat()
            }
        raise TypeError(f"Expected datetime or date, got {type(obj)}")
    
    def deserialize(self, data: Dict[str, Any]) -> Union[datetime, date]:
        """
        Deserialize dictionary to datetime.
        
        Args:
            data: Dictionary with type and ISO format string
            
        Returns:
            Datetime or date object
            
        Raises:
            ValueError: If type is unknown
        """
        iso_str = data["iso"]
        if data["type"] == "datetime":
            return datetime.fromisoformat(iso_str)
        elif data["type"] == "date":
            return date.fromisoformat(iso_str)
        raise ValueError(f"Unknown datetime type: {data['type']}")


class NumpySerializer(TypeSerializer[np.ndarray]):
    """Serializer for NumPy arrays."""
    
    @property
    def target_type(self) -> Type[np.ndarray]:
        return np.ndarray
    
    def serialize(self, obj: np.ndarray) -> Dict[str, Any]:
        """
        Serialize ndarray to dictionary.
        
        Args:
            obj: NumPy array to serialize
            
        Returns:
            Dictionary with array data, dtype, and shape
        """
        return {
            "type": "numpy.ndarray",
            "data": base64.b64encode(obj.tobytes()).decode('ascii'),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape)
        }
    
    def deserialize(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Deserialize dictionary to ndarray.
        
        Args:
            data: Dictionary with array data, dtype, and shape
            
        Returns:
            NumPy array
        """
        binary_data = base64.b64decode(data["data"])
        dtype = np.dtype(data["dtype"])
        shape = tuple(data["shape"])
        return np.frombuffer(binary_data, dtype=dtype).reshape(shape)