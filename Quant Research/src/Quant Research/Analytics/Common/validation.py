"""
Validation Utilities

This module provides validation functions for parameters, data, and signals used
across all analytics modules. It helps ensure that inputs are valid, consistent, 
and appropriate for the intended use.

Features:
- Parameter validation for various types and constraints
- Data validation for DataFrames and time series
- Signal validation for generated signals
- Validation context managers and decorators
- Schema validation for structured data

Usage:
    ```python
    from quant_research.analytics.common.validation import (
        validate_numeric_param,
        validate_dataframe,
        validate_signals,
        ValidationError
    )
    
    # Validate parameters
    window = validate_numeric_param(window, "window", min_value=1, max_value=1000)
    
    # Validate input DataFrame
    result, errors = validate_dataframe(
        df, 
        required_columns=['timestamp', 'close'],
        min_rows=10
    )
    if errors:
        raise ValidationError(f"Invalid input data: {errors}")
    
    # Validate generated signals
    valid_signals = validate_signals(signals, expected_columns=['timestamp', 'value'])
    ```
"""

# Standard library imports
import inspect
import logging
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Type

# Third-party imports
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError as PydanticValidationError, validator

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.validation")

#------------------------------------------------------------------------
# Exception Classes
#------------------------------------------------------------------------

class ValidationError(Exception):
    """
    Exception raised for validation errors.
    
    This error is raised when input validation fails and should be caught
    by the calling function to handle appropriately.
    
    Attributes:
        message: Explanation of the error
        param_name: Name of the parameter that failed validation
        value: Value that failed validation
        source: Source of the validation error
    """
    def __init__(
        self, 
        message: str, 
        param_name: Optional[str] = None,
        value: Any = None,
        source: Optional[str] = None
    ):
        self.message = message
        self.param_name = param_name
        self.value = value
        self.source = source
        
        # Format the error message
        full_message = message
        if param_name:
            full_message = f"{message} (parameter: {param_name})"
        if source:
            full_message = f"{full_message} [source: {source}]"
            
        super().__init__(full_message)


class DataValidationError(ValidationError):
    """
    Exception raised for data validation errors.
    
    This error is raised when input data validation fails and includes
    information about the specific data issues.
    
    Attributes:
        message: Explanation of the error
        data_info: Dictionary with information about the data
        errors: List of specific validation errors
    """
    def __init__(
        self, 
        message: str, 
        data_info: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None
    ):
        self.data_info = data_info or {}
        self.errors = errors or []
        
        # Format the error message
        full_message = message
        if errors:
            full_message = f"{message}: {'; '.join(errors)}"
            
        super().__init__(full_message, source="data_validation")


class SchemaValidationError(ValidationError):
    """
    Exception raised for schema validation errors.
    
    This error is raised when data does not conform to the expected schema.
    
    Attributes:
        message: Explanation of the error
        schema_errors: Dictionary mapping fields to error messages
    """
    def __init__(
        self, 
        message: str, 
        schema_errors: Optional[Dict[str, str]] = None
    ):
        self.schema_errors = schema_errors or {}
        
        # Format the error message
        full_message = message
        if schema_errors:
            error_details = "; ".join([f"{field}: {error}" for field, error in schema_errors.items()])
            full_message = f"{message}: {error_details}"
            
        super().__init__(full_message, source="schema_validation")


#------------------------------------------------------------------------
# Parameter Validation Functions
#------------------------------------------------------------------------

def validate_numeric_param(
    value: Any, 
    param_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_none: bool = False,
    default: Optional[float] = None,
    integer_only: bool = False
) -> Union[float, int, None]:
    """
    Validate a numeric parameter.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        integer_only: Whether to require integer values
        
    Returns:
        Validated numeric value (float, int, or None)
        
    Raises:
        ValidationError: If validation fails
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise ValidationError(
                f"Parameter '{param_name}' must not be None", 
                param_name=param_name,
                value=value
            )
    
    # Check if value is numeric
    try:
        if integer_only:
            # For integers, convert and check if conversion preserves value
            numeric_value = int(value)
            if float(numeric_value) != float(value):
                raise ValidationError(
                    f"Parameter '{param_name}' must be an integer", 
                    param_name=param_name, 
                    value=value
                )
        else:
            numeric_value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Parameter '{param_name}' must be numeric", 
            param_name=param_name,
            value=value
        )
    
    # Check minimum value
    if min_value is not None and numeric_value < min_value:
        raise ValidationError(
            f"Parameter '{param_name}' must be at least {min_value}", 
            param_name=param_name,
            value=value
        )
    
    # Check maximum value
    if max_value is not None and numeric_value > max_value:
        raise ValidationError(
            f"Parameter '{param_name}' must be at most {max_value}", 
            param_name=param_name,
            value=value
        )
    
    return numeric_value


def validate_string_param(
    value: Any,
    param_name: str,
    allowed_values: Optional[List[str]] = None,
    case_sensitive: bool = True,
    allow_none: bool = False,
    default: Optional[str] = None
) -> Optional[str]:
    """
    Validate a string parameter.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        allowed_values: List of allowed string values
        case_sensitive: Whether validation is case-sensitive
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        
    Returns:
        Validated string value or None
        
    Raises:
        ValidationError: If validation fails
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise ValidationError(
                f"Parameter '{param_name}' must not be None", 
                param_name=param_name,
                value=value
            )
    
    # Convert to string
    try:
        string_value = str(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Parameter '{param_name}' must be convertible to string", 
            param_name=param_name,
            value=value
        )
    
    # Check allowed values if specified
    if allowed_values:
        if case_sensitive:
            if string_value not in allowed_values:
                raise ValidationError(
                    f"Parameter '{param_name}' must be one of: {', '.join(allowed_values)}", 
                    param_name=param_name,
                    value=value
                )
        else:
            if string_value.lower() not in [v.lower() for v in allowed_values]:
                raise ValidationError(
                    f"Parameter '{param_name}' must be one of: {', '.join(allowed_values)} (case-insensitive)", 
                    param_name=param_name,
                    value=value
                )
                
            # Return the properly-cased version
            for allowed in allowed_values:
                if string_value.lower() == allowed.lower():
                    return allowed
    
    return string_value


def validate_bool_param(
    value: Any,
    param_name: str,
    allow_none: bool = False,
    default: Optional[bool] = None
) -> Optional[bool]:
    """
    Validate a boolean parameter.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        
    Returns:
        Validated boolean value or None
        
    Raises:
        ValidationError: If validation fails
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise ValidationError(
                f"Parameter '{param_name}' must not be None", 
                param_name=param_name,
                value=value
            )
    
    # Handle string values for convenience
    if isinstance(value, str):
        lower_value = value.lower()
        if lower_value in ('true', 't', 'yes', 'y', '1'):
            return True
        elif lower_value in ('false', 'f', 'no', 'n', '0'):
            return False
    
    # Handle numeric values
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        elif value == 0:
            return False
    
    # Handle boolean values
    if isinstance(value, bool):
        return value
    
    # If we get here, the value is not convertible to boolean
    raise ValidationError(
        f"Parameter '{param_name}' must be a boolean value", 
        param_name=param_name,
        value=value
    )


def validate_datetime_param(
    value: Any,
    param_name: str,
    min_datetime: Optional[datetime] = None,
    max_datetime: Optional[datetime] = None,
    allow_none: bool = False,
    default: Optional[datetime] = None,
    format_str: Optional[str] = None
) -> Optional[datetime]:
    """
    Validate a datetime parameter.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        min_datetime: Minimum allowed datetime
        max_datetime: Maximum allowed datetime
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        format_str: Format string for parsing string dates
        
    Returns:
        Validated datetime value or None
        
    Raises:
        ValidationError: If validation fails
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise ValidationError(
                f"Parameter '{param_name}' must not be None", 
                param_name=param_name,
                value=value
            )
    
    # Convert to datetime
    try:
        if isinstance(value, datetime):
            dt_value = value
        elif isinstance(value, pd.Timestamp):
            dt_value = value.to_pydatetime()
        elif isinstance(value, str):
            if format_str:
                dt_value = datetime.strptime(value, format_str)
            else:
                dt_value = pd.to_datetime(value).to_pydatetime()
        else:
            dt_value = pd.to_datetime(value).to_pydatetime()
    except (ValueError, TypeError):
        raise ValidationError(
            f"Parameter '{param_name}' must be a valid datetime", 
            param_name=param_name,
            value=value
        )
    
    # Check minimum datetime
    if min_datetime is not None and dt_value < min_datetime:
        raise ValidationError(
            f"Parameter '{param_name}' must be at or after {min_datetime}", 
            param_name=param_name,
            value=value
        )
    
    # Check maximum datetime
    if max_datetime is not None and dt_value > max_datetime:
        raise ValidationError(
            f"Parameter '{param_name}' must be at or before {max_datetime}", 
            param_name=param_name,
            value=value
        )
    
    return dt_value


def validate_list_param(
    value: Any,
    param_name: str,
    item_type: Optional[Type] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_none: bool = False,
    default: Optional[List] = None,
    item_validator: Optional[Callable[[Any, str], Any]] = None
) -> Optional[List]:
    """
    Validate a list parameter.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        item_type: Expected type of list items
        min_length: Minimum allowed list length
        max_length: Maximum allowed list length
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        item_validator: Function to validate each list item
        
    Returns:
        Validated list or None
        
    Raises:
        ValidationError: If validation fails
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise ValidationError(
                f"Parameter '{param_name}' must not be None", 
                param_name=param_name,
                value=value
            )
    
    # Convert to list if possible
    try:
        if isinstance(value, list):
            list_value = value
        elif isinstance(value, (tuple, set)):
            list_value = list(value)
        elif isinstance(value, (pd.Series, np.ndarray)):
            list_value = value.tolist()
        elif isinstance(value, str):
            # Treat string as single item, not a list of characters
            list_value = [value]
        else:
            # Try to convert to list as a last resort
            list_value = list(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Parameter '{param_name}' must be convertible to a list", 
            param_name=param_name,
            value=value
        )
    
    # Check list length
    if min_length is not None and len(list_value) < min_length:
        raise ValidationError(
            f"Parameter '{param_name}' must have at least {min_length} items", 
            param_name=param_name,
            value=value
        )
    
    if max_length is not None and len(list_value) > max_length:
        raise ValidationError(
            f"Parameter '{param_name}' must have at most {max_length} items", 
            param_name=param_name,
            value=value
        )
    
    # Validate list items if type or validator provided
    if item_type is not None or item_validator is not None:
        validated_items = []
        for i, item in enumerate(list_value):
            # Check item type
            if item_type is not None and not isinstance(item, item_type):
                raise ValidationError(
                    f"Item {i} in '{param_name}' must be of type {item_type.__name__}", 
                    param_name=f"{param_name}[{i}]",
                    value=item
                )
            
            # Apply item validator if provided
            if item_validator is not None:
                try:
                    validated_item = item_validator(item, f"{param_name}[{i}]")
                    validated_items.append(validated_item)
                except ValidationError as e:
                    # Re-raise with more context
                    raise ValidationError(
                        f"Item {i} in '{param_name}' failed validation: {e}", 
                        param_name=f"{param_name}[{i}]",
                        value=item
                    )
            else:
                validated_items.append(item)
        
        return validated_items
    
    return list_value


def validate_dict_param(
    value: Any,
    param_name: str,
    key_type: Optional[Type] = None,
    value_type: Optional[Type] = None,
    required_keys: Optional[List[str]] = None,
    allowed_keys: Optional[List[str]] = None,
    allow_none: bool = False,
    default: Optional[Dict] = None,
    value_validator: Optional[Callable[[Any, str], Any]] = None
) -> Optional[Dict]:
    """
    Validate a dictionary parameter.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        key_type: Expected type of dictionary keys
        value_type: Expected type of dictionary values
        required_keys: List of required dictionary keys
        allowed_keys: List of allowed dictionary keys
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        value_validator: Function to validate each dictionary value
        
    Returns:
        Validated dictionary or None
        
    Raises:
        ValidationError: If validation fails
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise ValidationError(
                f"Parameter '{param_name}' must not be None", 
                param_name=param_name,
                value=value
            )
    
    # Convert to dictionary if possible
    try:
        if isinstance(value, dict):
            dict_value = value
        elif isinstance(value, pd.Series):
            dict_value = value.to_dict()
        elif hasattr(value, "__dict__"):
            dict_value = value.__dict__
        else:
            # Try to convert to dict as a last resort
            dict_value = dict(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Parameter '{param_name}' must be convertible to a dictionary", 
            param_name=param_name,
            value=value
        )
    
    # Check required keys
    if required_keys:
        missing_keys = [key for key in required_keys if key not in dict_value]
        if missing_keys:
            raise ValidationError(
                f"Parameter '{param_name}' is missing required keys: {', '.join(missing_keys)}", 
                param_name=param_name,
                value=value
            )
    
    # Check allowed keys
    if allowed_keys:
        invalid_keys = [key for key in dict_value if key not in allowed_keys]
        if invalid_keys:
            raise ValidationError(
                f"Parameter '{param_name}' contains invalid keys: {', '.join(invalid_keys)}", 
                param_name=param_name,
                value=value
            )
    
    # Check key type
    if key_type is not None:
        invalid_keys = [key for key in dict_value if not isinstance(key, key_type)]
        if invalid_keys:
            raise ValidationError(
                f"Keys in '{param_name}' must be of type {key_type.__name__}", 
                param_name=param_name,
                value=value
            )
    
    # Validate dictionary values
    if value_type is not None or value_validator is not None:
        validated_dict = {}
        for key, val in dict_value.items():
            # Check value type
            if value_type is not None and not isinstance(val, value_type):
                raise ValidationError(
                    f"Value for key '{key}' in '{param_name}' must be of type {value_type.__name__}", 
                    param_name=f"{param_name}.{key}",
                    value=val
                )
            
            # Apply value validator if provided
            if value_validator is not None:
                try:
                    validated_val = value_validator(val, f"{param_name}.{key}")
                    validated_dict[key] = validated_val
                except ValidationError as e:
                    # Re-raise with more context
                    raise ValidationError(
                        f"Value for key '{key}' in '{param_name}' failed validation: {e}", 
                        param_name=f"{param_name}.{key}",
                        value=val
                    )
            else:
                validated_dict[key] = val
        
        return validated_dict
    
    return dict_value


def validate_enum_param(
    value: Any,
    param_name: str,
    enum_values: List[Any],
    case_insensitive: bool = False,
    allow_none: bool = False,
    default: Optional[Any] = None
) -> Any:
    """
    Validate a parameter against a list of allowed values (enum).
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        enum_values: List of allowed values
        case_insensitive: Whether to ignore case for string values
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        
    Returns:
        Validated enum value or None
        
    Raises:
        ValidationError: If validation fails
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise ValidationError(
                f"Parameter '{param_name}' must not be None", 
                param_name=param_name,
                value=value
            )
    
    # Case-insensitive comparison for strings
    if case_insensitive and isinstance(value, str):
        for enum_val in enum_values:
            if isinstance(enum_val, str) and value.lower() == enum_val.lower():
                return enum_val
        
        # Value not found in enum values
        raise ValidationError(
            f"Parameter '{param_name}' must be one of: {', '.join(map(str, enum_values))} (case-insensitive)", 
            param_name=param_name,
            value=value
        )
    
    # Direct comparison
    if value not in enum_values:
        raise ValidationError(
            f"Parameter '{param_name}' must be one of: {', '.join(map(str, enum_values))}", 
            param_name=param_name,
            value=value
        )
    
    return value


def validate_callable_param(
    value: Any,
    param_name: str,
    expected_args: Optional[List[str]] = None,
    expected_return_type: Optional[Type] = None,
    allow_none: bool = False,
    default: Optional[Callable] = None
) -> Optional[Callable]:
    """
    Validate a callable parameter (function or method).
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        expected_args: List of expected argument names
        expected_return_type: Expected return type
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        
    Returns:
        Validated callable or None
        
    Raises:
        ValidationError: If validation fails
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise ValidationError(
                f"Parameter '{param_name}' must not be None", 
                param_name=param_name,
                value=value
            )
    
    # Check if value is callable
    if not callable(value):
        raise ValidationError(
            f"Parameter '{param_name}' must be callable", 
            param_name=param_name,
            value=value
        )
    
    # Check expected arguments
    if expected_args is not None:
        sig = inspect.signature(value)
        param_names = list(sig.parameters.keys())
        
        # Check if all expected arguments are present
        for arg in expected_args:
            if arg not in param_names:
                raise ValidationError(
                    f"Callable '{param_name}' is missing expected argument '{arg}'", 
                    param_name=param_name,
                    value=value
                )
    
    # Check return type if possible and expected
    if expected_return_type is not None:
        sig = inspect.signature(value)
        if sig.return_annotation != inspect.Signature.empty:
            if sig.return_annotation != expected_return_type:
                raise ValidationError(
                    f"Callable '{param_name}' should return {expected_return_type.__name__}, but returns {sig.return_annotation.__name__}", 
                    param_name=param_name,
                    value=value
                )
    
    return value


#------------------------------------------------------------------------
# Data Validation Functions
#------------------------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    dtypes: Optional[Dict[str, Type]] = None,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
    check_index_type: Optional[Type] = None,
    check_sorted_index: bool = False,
    check_index_uniqueness: bool = False,
    check_null_columns: Optional[List[str]] = None,
    check_non_negative_columns: Optional[List[str]] = None,
    raise_exceptions: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate a pandas DataFrame against various criteria.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        dtypes: Dictionary mapping columns to expected data types
        min_rows: Minimum allowed number of rows
        max_rows: Maximum allowed number of rows
        check_index_type: Expected type of index
        check_sorted_index: Whether to check if index is sorted
        check_index_uniqueness: Whether to check if index is unique
        check_null_columns: Columns to check for NULL values
        check_non_negative_columns: Columns to check for negative values
        raise_exceptions: Whether to raise exceptions on validation errors
        
    Returns:
        Tuple of (validated DataFrame, list of validation errors)
        
    Raises:
        DataValidationError: If validation fails and raise_exceptions is True
    """
    errors = []
    
    # Basic DataFrame validation
    if not isinstance(df, pd.DataFrame):
        errors.append(f"Input must be a pandas DataFrame, got {type(df).__name__}")
        if raise_exceptions:
            raise DataValidationError("Invalid DataFrame type", errors=errors)
        return df, errors
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df_copy.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check data types
    if dtypes:
        for col, expected_type in dtypes.items():
            if col not in df_copy.columns:
                # Column was already reported as missing
                continue
                
            # Check if column type matches expected type
            actual_type = df_copy[col].dtype
            if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                errors.append(f"Column '{col}' has incorrect type: expected {expected_type}, got {actual_type}")
    
    # Check number of rows
    if min_rows is not None and len(df_copy) < min_rows:
        errors.append(f"DataFrame has too few rows: expected at least {min_rows}, got {len(df_copy)}")
    
    if max_rows is not None and len(df_copy) > max_rows:
        errors.append(f"DataFrame has too many rows: expected at most {max_rows}, got {len(df_copy)}")
    
    # Check index type
    if check_index_type is not None:
        if not isinstance(df_copy.index, check_index_type):
            errors.append(f"Index has incorrect type: expected {check_index_type.__name__}, got {type(df_copy.index).__name__}")
    
    # Check if index is sorted
    if check_sorted_index:
        if not df_copy.index.is_monotonic_increasing:
            errors.append("Index is not sorted in ascending order")
    
    # Check if index is unique
    if check_index_uniqueness:
        if not df_copy.index.is_unique:
            errors.append("Index contains duplicate values")
    
    # Check for NULL values
    if check_null_columns:
        for col in check_null_columns:
            if col not in df_copy.columns:
                # Column was already reported as missing
                continue
                
            null_count = df_copy[col].isna().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' contains {null_count} NULL values")
    
    # Check for negative values
    if check_non_negative_columns:
        for col in check_non_negative_columns:
            if col not in df_copy.columns:
                # Column was already reported as missing
                continue
                
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                errors.append(f"Column '{col}' should be numeric for non-negative check")
                continue
                
            neg_count = (df_copy[col] < 0).sum()
            if neg_count > 0:
                errors.append(f"Column '{col}' contains {neg_count} negative values")
    
    # Raise exception if requested and errors exist
    if raise_exceptions and errors:
        raise DataValidationError(
            "DataFrame validation failed",
            data_info={"shape": df_copy.shape, "columns": list(df_copy.columns)},
            errors=errors
        )
    
    return df_copy, errors


def validate_time_series(
    series: pd.Series,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    check_monotonic: bool = True,
    check_stationarity: bool = False,
    check_gaps: bool = False,
    max_gap: Optional[timedelta] = None,
    raise_exceptions: bool = False
) -> Tuple[pd.Series, List[str]]:
    """
    Validate a time series for common problems.
    
    Args:
        series: Time series to validate
        min_length: Minimum length of the series
        max_length: Maximum length of the series
        check_monotonic: Whether to check if values are monotonically increasing/decreasing
        check_stationarity: Whether to check if series is stationary (requires statsmodels)
        check_gaps: Whether to check for gaps in the time index
        max_gap: Maximum allowed gap between observations
        raise_exceptions: Whether to raise exceptions on validation errors
        
    Returns:
        Tuple of (validated series, list of validation errors)
        
    Raises:
        DataValidationError: If validation fails and raise_exceptions is True
    """
    errors = []
    
    # Basic series validation
    if not isinstance(series, pd.Series):
        errors.append(f"Input must be a pandas Series, got {type(series).__name__}")
        if raise_exceptions:
            raise DataValidationError("Invalid Series type", errors=errors)
        return series, errors
    
    # Make a copy to avoid modifying the original
    series_copy = series.copy()
    
    # Check series length
    if min_length is not None and len(series_copy) < min_length:
        errors.append(f"Series is too short: expected at least {min_length} observations, got {len(series_copy)}")
    
    if max_length is not None and len(series_copy) > max_length:
        errors.append(f"Series is too long: expected at most {max_length} observations, got {len(series_copy)}")
    
    # Check for monotonicity
    if check_monotonic:
        # Check if series is numeric
        if not pd.api.types.is_numeric_dtype(series_copy):
            errors.append("Series must be numeric for monotonicity check")
        else:
            is_increasing = series_copy.is_monotonic_increasing
            is_decreasing = series_copy.is_monotonic_decreasing
            
            if not (is_increasing or is_decreasing):
                # Neither strictly increasing nor decreasing
                errors.append("Series is not monotonically increasing or decreasing")
    
    # Check for stationarity
    if check_stationarity:
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Check if series is numeric
            if not pd.api.types.is_numeric_dtype(series_copy):
                errors.append("Series must be numeric for stationarity check")
            else:
                # Drop NaN values
                clean_series = series_copy.dropna()
                
                if len(clean_series) < 20:  # Need enough data for meaningful test
                    errors.append("Not enough data for stationarity test")
                else:
                    # Run Augmented Dickey-Fuller test
                    adf_result = adfuller(clean_series)
                    p_value = adf_result[1]
                    
                    if p_value > 0.05:
                        errors.append(f"Series is not stationary (ADF p-value: {p_value:.4f})")
        except ImportError:
            errors.append("statsmodels required for stationarity check")
    
    # Check for time index
    is_datetime_index = isinstance(series_copy.index, pd.DatetimeIndex)
    
    # Check for gaps in time series
    if check_gaps and is_datetime_index:
        # Check for gaps in time index
        idx = series_copy.index
        gaps = []
        
        # Calculate differences between consecutive timestamps
        if len(idx) > 1:
            diffs = idx[1:] - idx[:-1]
            
            # Find the most common difference as expected frequency
            if not max_gap:
                # Use 2x the median difference as the threshold
                median_diff = pd.Series(diffs).median()
                max_gap = median_diff * 2
            
            # Find gaps larger than max_gap
            large_diffs = [(i, diff) for i, diff in enumerate(diffs) if diff > max_gap]
            
            if large_diffs:
                for i, diff in large_diffs:
                    start_time = idx[i]
                    end_time = idx[i + 1]
                    gaps.append(f"{start_time} to {end_time} ({diff})")
                
                if len(gaps) > 3:
                    # Too many gaps to list them all
                    errors.append(f"Found {len(gaps)} time gaps larger than {max_gap}. First few: {', '.join(gaps[:3])}")
                else:
                    errors.append(f"Found time gaps: {', '.join(gaps)}")
    elif check_gaps and not is_datetime_index:
        errors.append("Series must have DatetimeIndex for gap checking")
    
    # Raise exception if requested and errors exist
    if raise_exceptions and errors:
        raise DataValidationError(
            "Time series validation failed",
            data_info={"length": len(series_copy), "dtype": str(series_copy.dtype)},
            errors=errors
        )
    
    return series_copy, errors


def validate_missing_data(
    df: pd.DataFrame,
    threshold: float = 0.1,
    columns: Optional[List[str]] = None,
    drop_columns: bool = False,
    fill_method: Optional[str] = None,
    raise_exceptions: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Validate and handle missing data in a DataFrame.
    
    Args:
        df: DataFrame to validate
        threshold: Maximum allowed proportion of missing data
        columns: Columns to check (None for all columns)
        drop_columns: Whether to drop columns with too many missing values
        fill_method: Method to fill missing values ('ffill', 'bfill', 'mean', etc.)
        raise_exceptions: Whether to raise exceptions on validation errors
        
    Returns:
        Tuple of (validated DataFrame, dictionary of missing data proportions)
        
    Raises:
        DataValidationError: If validation fails and raise_exceptions is True
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Determine columns to check
    check_columns = columns or df_copy.columns
    
    # Calculate missing data proportions
    missing_props = {}
    for col in check_columns:
        if col in df_copy.columns:
            missing_props[col] = df_copy[col].isna().mean()
    
    # Find columns with too many missing values
    high_missing_cols = {col: prop for col, prop in missing_props.items() if prop > threshold}
    
    # Handle columns with too many missing values
    if high_missing_cols:
        if drop_columns:
            df_copy = df_copy.drop(columns=list(high_missing_cols.keys()))
            
            # Update missing_props after dropping columns
            missing_props = {col: prop for col, prop in missing_props.items() if col not in high_missing_cols}
        elif fill_method:
            # Fill missing values
            if fill_method == 'ffill':
                df_copy = df_copy.ffill()
            elif fill_method == 'bfill':
                df_copy = df_copy.bfill()
            elif fill_method == 'mean':
                for col in high_missing_cols:
                    if pd.api.types.is_numeric_dtype(df_copy[col]):
                        df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            elif fill_method == 'median':
                for col in high_missing_cols:
                    if pd.api.types.is_numeric_dtype(df_copy[col]):
                        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif fill_method == 'mode':
                for col in high_missing_cols:
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            elif fill_method == 'zero':
                for col in high_missing_cols:
                    df_copy[col] = df_copy[col].fillna(0)
            else:
                warnings.warn(f"Unknown fill method: {fill_method}")
        
        # Raise exception if requested
        if raise_exceptions:
            errors = [f"Column '{col}' has {prop*100:.1f}% missing values (threshold: {threshold*100:.1f}%)" 
                     for col, prop in high_missing_cols.items()]
            
            raise DataValidationError(
                "Too many missing values",
                data_info={"total_rows": len(df_copy), "missing_columns": len(high_missing_cols)},
                errors=errors
            )
    
    return df_copy, missing_props


def validate_data_range(
    df: pd.DataFrame,
    column_ranges: Dict[str, Tuple[float, float]],
    handle_outliers: str = 'none',
    raise_exceptions: bool = False
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validate that data falls within specified ranges.
    
    Args:
        df: DataFrame to validate
        column_ranges: Dictionary mapping columns to (min, max) ranges
        handle_outliers: How to handle outliers ('none', 'clip', 'remove', 'nan')
        raise_exceptions: Whether to raise exceptions on validation errors
        
    Returns:
        Tuple of (validated DataFrame, dictionary of outlier counts)
        
    Raises:
        DataValidationError: If validation fails and raise_exceptions is True
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Count outliers for each column
    outlier_counts = {}
    errors = []
    
    for col, (min_val, max_val) in column_ranges.items():
        if col not in df_copy.columns:
            errors.append(f"Column '{col}' not found in DataFrame")
            continue
        
        # Count outliers
        below_min = (df_copy[col] < min_val).sum()
        above_max = (df_copy[col] > max_val).sum()
        outlier_counts[col] = below_min + above_max
        
        if outlier_counts[col] > 0:
            errors.append(f"Column '{col}' has {outlier_counts[col]} values outside range [{min_val}, {max_val}]")
            
            # Handle outliers
            if handle_outliers == 'clip':
                df_copy[col] = df_copy[col].clip(min_val, max_val)
            elif handle_outliers == 'remove':
                outlier_mask = (df_copy[col] < min_val) | (df_copy[col] > max_val)
                df_copy = df_copy[~outlier_mask]
            elif handle_outliers == 'nan':
                outlier_mask = (df_copy[col] < min_val) | (df_copy[col] > max_val)
                df_copy.loc[outlier_mask, col] = np.nan
    
    # Raise exception if requested and errors exist
    if raise_exceptions and errors:
        raise DataValidationError(
            "Data range validation failed",
            data_info={"total_rows": len(df), "outlier_columns": len(outlier_counts)},
            errors=errors
        )
    
    return df_copy, outlier_counts


#------------------------------------------------------------------------
# Signal Validation Functions
#------------------------------------------------------------------------

def validate_signals(
    signals: Union[pd.DataFrame, List[Dict]],
    expected_columns: Optional[List[str]] = None,
    min_signals: Optional[int] = None,
    max_signals: Optional[int] = None,
    check_timestamp_order: bool = True,
    check_signal_values: bool = False,
    allowed_signal_values: Optional[List[Any]] = None,
    as_dataframe: bool = True,
    raise_exceptions: bool = False
) -> Union[pd.DataFrame, List[Dict]]:
    """
    Validate signals for consistency and expected format.
    
    Args:
        signals: Signals as DataFrame or list of dictionaries
        expected_columns: List of expected columns/keys
        min_signals: Minimum number of signals
        max_signals: Maximum number of signals
        check_timestamp_order: Whether to check if timestamps are in order
        check_signal_values: Whether to check signal values against allowed values
        allowed_signal_values: List of allowed signal values
        as_dataframe: Whether to return signals as DataFrame
        raise_exceptions: Whether to raise exceptions on validation errors
        
    Returns:
        Validated signals (as DataFrame or list)
        
    Raises:
        ValidationError: If validation fails and raise_exceptions is True
    """
    errors = []
    
    # Convert to DataFrame if necessary
    if isinstance(signals, list):
        if not signals:
            signals_df = pd.DataFrame()
        else:
            signals_df = pd.DataFrame(signals)
    elif isinstance(signals, pd.DataFrame):
        signals_df = signals.copy()
    else:
        errors.append(f"Signals must be a DataFrame or list of dictionaries, got {type(signals).__name__}")
        if raise_exceptions:
            raise ValidationError("\n".join(errors), source="signal_validation")
        # Return empty
        return pd.DataFrame() if as_dataframe else []
    
    # Check if DataFrame is empty
    if signals_df.empty:
        if min_signals is not None and min_signals > 0:
            errors.append(f"No signals found, expected at least {min_signals}")
        
        # Return empty
        return signals_df if as_dataframe else []
    
    # Check expected columns
    if expected_columns:
        missing_columns = [col for col in expected_columns if col not in signals_df.columns]
        if missing_columns:
            errors.append(f"Missing expected columns: {', '.join(missing_columns)}")
    
    # Check number of signals
    if min_signals is not None and len(signals_df) < min_signals:
        errors.append(f"Too few signals: expected at least {min_signals}, got {len(signals_df)}")
    
    if max_signals is not None and len(signals_df) > max_signals:
        errors.append(f"Too many signals: expected at most {max_signals}, got {len(signals_df)}")
    
    # Check timestamp order
    if check_timestamp_order and 'timestamp' in signals_df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(signals_df['timestamp']):
            try:
                signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            except Exception as e:
                errors.append(f"Could not convert timestamp column to datetime: {e}")
        
        # Check if timestamps are in order
        if not signals_df['timestamp'].is_monotonic_increasing:
            errors.append("Timestamps are not in ascending order")
    
    # Check signal values
    if check_signal_values and 'signal' in signals_df.columns:
        if allowed_signal_values:
            invalid_values = signals_df[~signals_df['signal'].isin(allowed_signal_values)]['signal'].unique()
            if len(invalid_values) > 0:
                errors.append(f"Invalid signal values: {invalid_values}. Allowed values: {allowed_signal_values}")
    
    # Raise exception if requested and errors exist
    if raise_exceptions and errors:
        raise ValidationError(
            "Signal validation failed: " + "\n".join(errors),
            source="signal_validation"
        )
    
    if as_dataframe:
        return signals_df
    else:
        return signals_df.to_dict('records')


class SignalSchema(BaseModel):
    """
    Schema for validating signal objects.
    
    This class defines the expected structure and constraints for
    signal objects used throughout the analytics engine.
    """
    timestamp: datetime
    signal_type: str
    value: float
    source: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        """Configuration for schema validation."""
        extra = "forbid"  # Forbid extra fields
    
    @validator('confidence')
    def check_confidence(cls, v):
        """Validate confidence score."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Confidence score must be between 0 and 1")
        return v


def validate_signal_objects(
    signals: List[Any],
    strict: bool = True,
    coerce: bool = False,
    raise_exceptions: bool = False
) -> Tuple[List[Any], List[str]]:
    """
    Validate a list of signal objects against the signal schema.
    
    Args:
        signals: List of signal objects to validate
        strict: Whether to use strict validation
        coerce: Whether to try to coerce values to the expected types
        raise_exceptions: Whether to raise exceptions on validation errors
        
    Returns:
        Tuple of (validated signals, list of validation errors)
        
    Raises:
        SchemaValidationError: If validation fails and raise_exceptions is True
    """
    errors = []
    validated_signals = []
    
    # Validate each signal
    for i, signal in enumerate(signals):
        try:
            # Convert to dict if it's not already
            if hasattr(signal, "__dict__"):
                signal_dict = signal.__dict__
            elif hasattr(signal, "to_dict"):
                signal_dict = signal.to_dict()
            elif isinstance(signal, dict):
                signal_dict = signal
            else:
                errors.append(f"Signal {i} has invalid type: {type(signal).__name__}")
                continue
            
            # Validate against schema
            validated = SignalSchema(**signal_dict)
            validated_signals.append(validated)
        
        except PydanticValidationError as e:
            # Extract error details
            error_dict = e.errors()
            error_messages = [f"{'.'.join(err['loc'])}: {err['msg']}" for err in error_dict]
            
            # Add to errors list
            errors.append(f"Signal {i} failed validation: {'; '.join(error_messages)}")
            
            # Try to fix errors if coercion is enabled
            if coerce:
                try:
                    coerced_dict = signal_dict.copy()
                    
                    # Try to coerce common fields
                    if 'timestamp' in coerced_dict and not isinstance(coerced_dict['timestamp'], datetime):
                        coerced_dict['timestamp'] = pd.to_datetime(coerced_dict['timestamp'])
                    
                    if 'value' in coerced_dict and not isinstance(coerced_dict['value'], (int, float)):
                        coerced_dict['value'] = float(coerced_dict['value'])
                    
                    if 'confidence' in coerced_dict and coerced_dict['confidence'] is not None:
                        if not isinstance(coerced_dict['confidence'], (int, float)):
                            coerced_dict['confidence'] = float(coerced_dict['confidence'])
                        
                        # Clip confidence to [0, 1]
                        coerced_dict['confidence'] = max(0, min(1, coerced_dict['confidence']))
                    
                    # Try validation again with coerced values
                    validated = SignalSchema(**coerced_dict)
                    validated_signals.append(validated)
                    
                    # Replace the error message with a warning
                    errors[-1] = f"Signal {i} needed coercion: {'; '.join(error_messages)}"
                
                except Exception as coerce_error:
                    # Coercion failed
                    errors[-1] += f" (coercion failed: {coerce_error})"
    
    # Raise exception if requested and errors exist
    if raise_exceptions and (errors and strict):
        raise SchemaValidationError(
            "Signal validation failed",
            schema_errors={i: error for i, error in enumerate(errors)}
        )
    
    return validated_signals, errors


#------------------------------------------------------------------------
# Validation Context Managers and Decorators
#------------------------------------------------------------------------

@contextmanager
def validation_context(
    context_name: str,
    raise_exceptions: bool = True,
    log_errors: bool = True
):
    """
    Context manager for validation operations.
    
    Args:
        context_name: Name of the validation context
        raise_exceptions: Whether to raise exceptions on validation errors
        log_errors: Whether to log validation errors
        
    Yields:
        List to collect validation errors
        
    Raises:
        ValidationError: If validation fails and raise_exceptions is True
    """
    errors = []
    
    try:
        yield errors
    except Exception as e:
        # Add the exception to the errors list
        errors.append(f"Exception: {str(e)}")
        
        # Re-raise exception if requested
        if raise_exceptions:
            raise
    finally:
        # Log errors if requested
        if log_errors and errors:
            for error in errors:
                logger.warning(f"Validation error in {context_name}: {error}")
            
        # Raise ValidationError if requested and errors exist
        if raise_exceptions and errors:
            raise ValidationError(
                f"Validation failed in {context_name}", 
                source=context_name
            )


def validate_inputs(
    **validators: Dict[str, Callable]
):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Dictionary mapping parameter names to validator functions
        
    Returns:
        Decorated function
        
    Example:
        @validate_inputs(
            window=lambda x: validate_numeric_param(x, "window", min_value=1),
            method=lambda x: validate_string_param(x, "method", allowed_values=["mean", "median"])
        )
        def calculate_statistic(data, window, method="mean"):
            # Function body
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            
            # Bind arguments to parameters
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate parameters
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    # Get the argument value
                    value = bound_args.arguments[param_name]
                    
                    # Apply the validator
                    try:
                        validated_value = validator(value)
                        
                        # Update the argument with the validated value
                        bound_args.arguments[param_name] = validated_value
                    
                    except ValidationError as e:
                        # Add function name to error
                        e.source = f"{func.__name__}"
                        raise e
            
            # Call the function with validated arguments
            return func(*bound_args.args, **bound_args.kwargs)
        
        return wrapper
    
    return decorator


def validate_return(
    validator: Callable[[Any], Any],
    description: str = "return value"
):
    """
    Decorator to validate function return value.
    
    Args:
        validator: Function to validate the return value
        description: Description of the return value for error messages
        
    Returns:
        Decorated function
        
    Example:
        @validate_return(lambda x: isinstance(x, pd.DataFrame), "DataFrame")
        def process_data(data):
            # Function body
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function
            result = func(*args, **kwargs)
            
            # Validate the return value
            try:
                if not validator(result):
                    raise ValidationError(
                        f"Invalid {description} from {func.__name__}",
                        source=f"{func.__name__}"
                    )
            except Exception as e:
                if not isinstance(e, ValidationError):
                    e = ValidationError(
                        f"Error validating {description} from {func.__name__}: {e}",
                        source=f"{func.__name__}"
                    )
                raise e
            
            return result
        
        return wrapper
    
    return decorator


def schema_validator(
    schema_model: Type[BaseModel],
    context_name: str = "schema validation"
):
    """
    Decorator to validate data against a pydantic schema.
    
    Args:
        schema_model: Pydantic model class to validate against
        context_name: Name of the validation context
        
    Returns:
        Decorated function
        
    Example:
        @schema_validator(UserSchema)
        def process_user(user_data: Dict):
            # Function body
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            
            # Find the parameter to validate (assume first param of matching type)
            param_to_validate = None
            param_name = None
            
            for name, param in sig.parameters.items():
                if param.annotation == Dict or param.annotation == dict:
                    param_to_validate = name
                    param_name = name
                    break
            
            # If no parameter found, just call the function
            if param_to_validate is None:
                return func(*args, **kwargs)
            
            # Get the argument to validate
            arg_dict = None
            if param_to_validate < len(args):
                arg_dict = args[param_to_validate]
            elif param_name in kwargs:
                arg_dict = kwargs[param_name]
            
            # If no argument found, just call the function
            if arg_dict is None:
                return func(*args, **kwargs)
            
            # Validate the argument against the schema
            try:
                validated = schema_model(**arg_dict)
                
                # Update the argument with the validated object
                if param_to_validate < len(args):
                    args_list = list(args)
                    args_list[param_to_validate] = validated.dict()
                    args = tuple(args_list)
                else:
                    kwargs[param_name] = validated.dict()
                
            except PydanticValidationError as e:
                # Extract error details
                error_dict = e.errors()
                schema_errors = {'.'.join(err['loc']): err['msg'] for err in error_dict}
                
                raise SchemaValidationError(
                    f"Schema validation failed in {context_name}",
                    schema_errors=schema_errors
                )
            
            # Call the function with validated arguments
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def safe_validation(
    default_value: Any = None,
    log_errors: bool = True
):
    """
    Decorator to catch and handle validation errors.
    
    Args:
        default_value: Value to return if validation fails
        log_errors: Whether to log validation errors
        
    Returns:
        Decorated function
        
    Example:
        @safe_validation(default_value=[])
        def process_data(data):
            # Function that may raise ValidationError
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValidationError as e:
                if log_errors:
                    logger.warning(f"Validation error in {func.__name__}: {e}")
                return default_value
        
        return wrapper
    
    return decorator