"""Serializer for pandas DataFrames."""

from typing import Dict, Any, Type, Optional, Union
import io

import pandas as pd
import numpy as np

from ..core import TypeSerializer

class DataFrameSerializer(TypeSerializer[pd.DataFrame]):
    """Serializer for pandas DataFrame objects."""
    
    @property
    def target_type(self) -> Type[pd.DataFrame]:
        return pd.DataFrame
    
    def serialize(self, obj: pd.DataFrame) -> Dict[str, Any]:
        """
        Serialize DataFrame to dictionary.
        
        This method preserves index, columns, and data types as much as possible.
        For complex data types, it converts to simpler representations.
        
        Args:
            obj: DataFrame to serialize
            
        Returns:
            Dictionary representation of the DataFrame
        """
        # Make a copy to avoid modifying the original
        df = obj.copy()
        
        # Handle index
        has_named_index = df.index.name is not None
        has_multiindex = isinstance(df.index, pd.MultiIndex)
        
        # Save index info
        index_info = {
            "name": df.index.name,
            "is_multiindex": has_multiindex
        }
        
        if has_multiindex:
            index_info["names"] = list(df.index.names)
            # Reset to convert MultiIndex to columns
            df = df.reset_index()
        
        # Handle special data types
        for col in df.columns:
            # Convert datetime columns to strings
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
            
            # Convert categorical to strings
            elif pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype(str)
            
            # Convert complex objects to strings
            elif df[col].dtype == 'object' and not pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(str)
        
        # Split data dictionary
        data_dict = {
            "index": index_info,
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        return data_dict
    
    def deserialize(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Deserialize dictionary to DataFrame.
        
        Reconstructs the DataFrame including index and data types.
        
        Args:
            data: Dictionary representation of DataFrame
            
        Returns:
            Reconstructed DataFrame
            
        Raises:
            ValueError: If data dictionary is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Invalid DataFrame data format")
        
        if "data" not in data or "columns" not in data:
            # Try alternate format
            if isinstance(data, dict) and all(isinstance(k, str) for k in data.keys()):
                return pd.DataFrame(data)
            raise ValueError("Missing required keys in DataFrame data")
        
        # Create DataFrame from records
        df = pd.DataFrame(data["data"], columns=data["columns"])
        
        # Restore data types where possible
        if "dtypes" in data:
            for col, dtype_str in data["dtypes"].items():
                if col in df.columns:
                    try:
                        # Handle numeric types
                        if dtype_str.startswith('int') or dtype_str.startswith('uint'):
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype_str)
                        elif dtype_str.startswith('float'):
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype_str)
                        # Handle datetime
                        elif 'datetime' in dtype_str:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        # Handle boolean
                        elif dtype_str == 'bool':
                            df[col] = df[col].astype(bool)
                    except Exception:
                        # If conversion fails, keep as is
                        pass
        
        # Restore index if needed
        if "index" in data:
            index_info = data["index"]
            
            if index_info.get("is_multiindex", False):
                # Restore MultiIndex
                df = df.set_index(index_info["names"])
            elif index_info.get("name") is not None:
                # Single index with name
                index_col = index_info["name"]
                if index_col in df.columns:
                    df = df.set_index(index_col)
                    df.index.name = index_info["name"]
        
        return df