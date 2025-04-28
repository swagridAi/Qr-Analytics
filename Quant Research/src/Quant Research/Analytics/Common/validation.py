"""
Validation Utilities

This module provides reusable validation functions and utilities to simplify
and standardize validation logic throughout the codebase. It reduces code duplication
and provides consistent error handling patterns.

Features:
- Core validation helpers for common patterns
- Parameter validation utilities
- DataFrame validation functions
- Time series specific validation
- Validation error collection and reporting

Usage:
    ```python
    from quant_research.core.validation_utils import (
        validate_param_base,
        validate_type,
        validate_numeric,
        validate_dataframe,
        ValidationErrorCollector
    )
    
    # Basic parameter validation
    def my_function(window_size: int, threshold: float = 0.5):
        # Validate parameters
        window_size = validate_numeric(
            window_size, 'window_size', min_value=1, integer_only=True
        )
        threshold = validate_numeric(
            threshold, 'threshold', min_value=0.0, max_value=1.0, allow_none=True, default=0.5
        )
        
        # Function logic...
    
    # Validation with error collection
    def validate_input_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        collector = ValidationErrorCollector()
        
        # Validate DataFrame has required columns
        has_columns, missing = validate_columns_exist(
            df, ['timestamp', 'close', 'volume'], raise_error=False
        )
        
        if not has_columns:
            collector.add_error(f"Missing required columns: {', '.join(missing)}")
            
        # Validate DataFrame has enough rows
        if len(df) < 100:
            collector.add_error(f"Insufficient data: {len(df)} rows (minimum 100)")
        
        # Check if we have any errors
        if collector.has_errors():
            logger.warning(f"Validation errors: {collector.get_error_message()}")
            return df, False
            
        return df, True
    ```
"""

# Standard library imports
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, TypeVar, Generic, Set

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from quant_research.core.errors import (
    ValidationError, DataValidationError, ParameterError,
    create_param_error, create_missing_columns_error, create_type_error, create_value_error
)

# Configure module logger
logger = logging.getLogger("quant_research.core.validation_utils")

# Type variables for generic functions
T = TypeVar('T')
DFType = TypeVar('DFType', bound=pd.DataFrame)

#------------------------------------------------------------------------
# Base Validation Utilities
#------------------------------------------------------------------------

def validate_param_base(
    value: T,
    param_name: str,
    allow_none: bool = False,
    default: Optional[T] = None,
    source: Optional[str] = None
) -> Optional[T]:
    """
    Base validation function for all parameter types.
    
    This function provides a foundation for parameter validation by handling
    None values consistently according to the specified options.
    
    Args:
        value: The parameter value to validate
        param_name: Name of the parameter
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        source: Source of the validation (for error reporting)
        
    Returns:
        The validated value, default value, or None
        
    Raises:
        ParameterError: If value is None and neither allow_none nor default is provided
    """
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise create_param_error(param_name, "must not be None", source=source)
    return value


def validate_constraint(
    value: T,
    param_name: str,
    constraint_func: Callable[[T], bool],
    failure_message: str,
    source: Optional[str] = None
) -> T:
    """
    Generic constraint validator for parameters.
    
    This function tests a value against a constraint function and raises
    an error with the specified message if the constraint is not satisfied.
    
    Args:
        value: The parameter value to validate
        param_name: Name of the parameter
        constraint_func: Function that returns True if the constraint is satisfied
        failure_message: Error message if constraint fails
        source: Source of the validation (for error reporting)
        
    Returns:
        The validated value
        
    Raises:
        ParameterError: If the constraint is not satisfied
        
    Example:
        ```python
        # Validate a parameter is positive
        value = validate_constraint(value, 'param', lambda x: x > 0, "must be positive")
        
        # Validate a string parameter has a minimum length
        name = validate_constraint(name, 'name', lambda x: len(x) >= 3, "must be at least 3 characters")
        ```
    """
    if not constraint_func(value):
        raise create_param_error(param_name, failure_message, value, source=source)
    return value


#------------------------------------------------------------------------
# Parameter Validation Functions
#------------------------------------------------------------------------

def validate_type(
    value: Any,
    param_name: str,
    expected_type: Union[Type, Tuple[Type, ...]],
    allow_none: bool = False,
    default: Optional[Any] = None,
    source: Optional[str] = None
) -> Any:
    """
    Validate that a parameter is of the expected type.
    
    Args:
        value: The parameter value to validate
        param_name: Name of the parameter
        expected_type: Expected type or tuple of types
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        source: Source of the validation (for error reporting)
        
    Returns:
        The validated value
        
    Raises:
        ParameterError: If the parameter is not of the expected type
        
    Example:
        ```python
        # Validate integer parameter
        window = validate_type(window, 'window', int)
        
        # Validate string parameter with default
        name = validate_type(name, 'name', str, default="default_name")
        
        # Validate parameter that can be int or float
        threshold = validate_type(threshold, 'threshold', (int, float))
        ```
    """
    # Handle None values
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise create_param_error(param_name, "must not be None", source=source)
    
    # Check type
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = [t.__name__ for t in expected_type]
            expected_str = ", ".join(type_names)
            message = f"must be one of: {expected_str}, got {type(value).__name__}"
        else:
            message = f"must be {expected_type.__name__}, got {type(value).__name__}"
        
        raise create_param_error(param_name, message, value, source=source)
    
    return value


def validate_numeric(
    value: Any,
    param_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_none: bool = False,
    default: Optional[float] = None,
    integer_only: bool = False,
    source: Optional[str] = None
) -> Optional[Union[int, float]]:
    """
    Validate a numeric parameter with optional range constraints.
    
    Args:
        value: The parameter value to validate
        param_name: Name of the parameter
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        integer_only: Whether to require integer values
        source: Source of the validation (for error reporting)
        
    Returns:
        The validated numeric value
        
    Raises:
        ParameterError: If validation fails
        
    Example:
        ```python
        # Validate positive integer
        window_size = validate_numeric(window_size, 'window_size', min_value=1, integer_only=True)
        
        # Validate probability between 0 and 1
        threshold = validate_numeric(threshold, 'threshold', min_value=0.0, max_value=1.0)
        
        # Validate optional parameter with default
        alpha = validate_numeric(alpha, 'alpha', min_value=0, allow_none=True, default=0.05)
        ```
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise create_param_error(param_name, "must not be None", source=source)
    
    # Check if value is numeric
    try:
        if integer_only:
            # For integers, convert and check if conversion preserves value
            numeric_value = int(value)
            if float(numeric_value) != float(value):
                raise create_param_error(
                    param_name, "must be an integer", value, source=source
                )
        else:
            numeric_value = float(value)
    except (ValueError, TypeError):
        raise create_param_error(
            param_name, "must be numeric", value, source=source
        )
    
    # Check minimum value
    if min_value is not None and numeric_value < min_value:
        raise create_param_error(
            param_name, f"must be at least {min_value}", numeric_value, source=source
        )
    
    # Check maximum value
    if max_value is not None and numeric_value > max_value:
        raise create_param_error(
            param_name, f"must be at most {max_value}", numeric_value, source=source
        )
    
    return numeric_value


def validate_string(
    value: Any,
    param_name: str,
    allowed_values: Optional[List[str]] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_none: bool = False,
    default: Optional[str] = None,
    case_sensitive: bool = True,
    source: Optional[str] = None
) -> Optional[str]:
    """
    Validate a string parameter with optional constraints.
    
    Args:
        value: The parameter value to validate
        param_name: Name of the parameter
        allowed_values: List of allowed string values
        min_length: Minimum allowed string length
        max_length: Maximum allowed string length
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        case_sensitive: Whether validation is case-sensitive
        source: Source of the validation (for error reporting)
        
    Returns:
        The validated string value
        
    Raises:
        ParameterError: If validation fails
        
    Example:
        ```python
        # Validate string from allowed values
        method = validate_string(method, 'method', allowed_values=['mean', 'median', 'mode'])
        
        # Validate string length
        name = validate_string(name, 'name', min_length=3, max_length=50)
        
        # Case-insensitive validation
        color = validate_string(color, 'color', allowed_values=['red', 'green', 'blue'], case_sensitive=False)
        ```
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise create_param_error(param_name, "must not be None", source=source)
    
    # Convert to string
    try:
        string_value = str(value)
    except Exception:
        raise create_param_error(
            param_name, "must be convertible to string", value, source=source
        )
    
    # Check length constraints
    if min_length is not None and len(string_value) < min_length:
        raise create_param_error(
            param_name, f"must have at least {min_length} characters", string_value, source=source
        )
    
    if max_length is not None and len(string_value) > max_length:
        raise create_param_error(
            param_name, f"must have at most {max_length} characters", string_value, source=source
        )
    
    # Check allowed values if specified
    if allowed_values:
        if case_sensitive:
            if string_value not in allowed_values:
                raise create_param_error(
                    param_name, 
                    f"must be one of: {', '.join(allowed_values)}", 
                    string_value, 
                    source=source
                )
        else:
            if string_value.lower() not in [v.lower() for v in allowed_values]:
                raise create_param_error(
                    param_name, 
                    f"must be one of: {', '.join(allowed_values)} (case-insensitive)", 
                    string_value, 
                    source=source
                )
                
            # Return the properly-cased version
            for allowed in allowed_values:
                if string_value.lower() == allowed.lower():
                    return allowed
    
    return string_value


def validate_datetime(
    value: Any,
    param_name: str,
    min_datetime: Optional[datetime] = None,
    max_datetime: Optional[datetime] = None,
    allow_none: bool = False,
    default: Optional[datetime] = None,
    source: Optional[str] = None
) -> Optional[datetime]:
    """
    Validate a datetime parameter with optional range constraints.
    
    Args:
        value: The parameter value to validate
        param_name: Name of the parameter
        min_datetime: Minimum allowed datetime (inclusive)
        max_datetime: Maximum allowed datetime (inclusive)
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        source: Source of the validation (for error reporting)
        
    Returns:
        The validated datetime value
        
    Raises:
        ParameterError: If validation fails
        
    Example:
        ```python
        # Validate datetime in range
        start_date = validate_datetime(start_date, 'start_date', 
                                      min_datetime=datetime(2020, 1, 1),
                                      max_datetime=datetime.now())
        
        # Validate with default
        as_of_date = validate_datetime(as_of_date, 'as_of_date', 
                                      allow_none=True,
                                      default=datetime.now())
        ```
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise create_param_error(param_name, "must not be None", source=source)
    
    # Convert to datetime
    try:
        if isinstance(value, datetime):
            dt_value = value
        elif isinstance(value, pd.Timestamp):
            dt_value = value.to_pydatetime()
        elif isinstance(value, str):
            dt_value = pd.to_datetime(value).to_pydatetime()
        else:
            dt_value = pd.to_datetime(value).to_pydatetime()
    except Exception:
        raise create_param_error(
            param_name, "must be a valid datetime", value, source=source
        )
    
    # Check minimum datetime
    if min_datetime is not None and dt_value < min_datetime:
        raise create_param_error(
            param_name, f"must be at or after {min_datetime}", dt_value, source=source
        )
    
    # Check maximum datetime
    if max_datetime is not None and dt_value > max_datetime:
        raise create_param_error(
            param_name, f"must be at or before {max_datetime}", dt_value, source=source
        )
    
    return dt_value


def validate_list(
    value: Any,
    param_name: str,
    item_type: Optional[Type] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_none: bool = False,
    default: Optional[List] = None,
    item_validator: Optional[Callable[[Any, str], Any]] = None,
    source: Optional[str] = None
) -> Optional[List]:
    """
    Validate a list parameter with optional constraints.
    
    Args:
        value: The parameter value to validate
        param_name: Name of the parameter
        item_type: Expected type of list items
        min_length: Minimum allowed list length
        max_length: Maximum allowed list length
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        item_validator: Function to validate each list item
        source: Source of the validation (for error reporting)
        
    Returns:
        The validated list
        
    Raises:
        ParameterError: If validation fails
        
    Example:
        ```python
        # Validate list of strings
        columns = validate_list(columns, 'columns', item_type=str)
        
        # Validate list of positive integers
        windows = validate_list(
            windows, 'windows', 
            item_type=int, 
            item_validator=lambda x, n: validate_numeric(x, n, min_value=1)
        )
        
        # Validate with length constraints
        top_n = validate_list(top_n, 'top_n', min_length=1, max_length=10)
        ```
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise create_param_error(param_name, "must not be None", source=source)
    
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
    except Exception:
        raise create_param_error(
            param_name, "must be convertible to a list", value, source=source
        )
    
    # Check list length
    if min_length is not None and len(list_value) < min_length:
        raise create_param_error(
            param_name, f"must have at least {min_length} items", 
            value, source=source
        )
    
    if max_length is not None and len(list_value) > max_length:
        raise create_param_error(
            param_name, f"must have at most {max_length} items", 
            value, source=source
        )
    
    # Validate list items if type or validator provided
    if item_type is not None or item_validator is not None:
        validated_items = []
        for i, item in enumerate(list_value):
            # Check item type
            if item_type is not None and not isinstance(item, item_type):
                raise create_param_error(
                    f"{param_name}[{i}]", 
                    f"must be of type {item_type.__name__}", 
                    item, source=source
                )
            
            # Apply item validator if provided
            if item_validator is not None:
                try:
                    validated_item = item_validator(item, f"{param_name}[{i}]")
                    validated_items.append(validated_item)
                except Exception as e:
                    raise create_param_error(
                        f"{param_name}[{i}]", 
                        f"failed validation: {e}", 
                        item, source=source
                    )
            else:
                validated_items.append(item)
        
        return validated_items
    
    return list_value


def validate_dict(
    value: Any,
    param_name: str,
    key_type: Optional[Type] = None,
    value_type: Optional[Type] = None,
    required_keys: Optional[List[str]] = None,
    allowed_keys: Optional[List[str]] = None,
    allow_none: bool = False,
    default: Optional[Dict] = None,
    value_validator: Optional[Callable[[Any, str], Any]] = None,
    source: Optional[str] = None
) -> Optional[Dict]:
    """
    Validate a dictionary parameter with optional constraints.
    
    Args:
        value: The parameter value to validate
        param_name: Name of the parameter
        key_type: Expected type of dictionary keys
        value_type: Expected type of dictionary values
        required_keys: List of required dictionary keys
        allowed_keys: List of allowed dictionary keys
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        value_validator: Function to validate each dictionary value
        source: Source of the validation (for error reporting)
        
    Returns:
        The validated dictionary
        
    Raises:
        ParameterError: If validation fails
        
    Example:
        ```python
        # Validate dictionary with required keys
        options = validate_dict(options, 'options', required_keys=['mode', 'threshold'])
        
        # Validate dictionary values
        mappings = validate_dict(
            mappings, 'mappings', 
            key_type=str, 
            value_type=int, 
            value_validator=lambda x, n: validate_numeric(x, n, min_value=0)
        )
        
        # Validate with allowed keys
        params = validate_dict(params, 'params', allowed_keys=['alpha', 'beta', 'gamma'])
        ```
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise create_param_error(param_name, "must not be None", source=source)
    
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
    except Exception:
        raise create_param_error(
            param_name, "must be convertible to a dictionary", value, source=source
        )
    
    # Check required keys
    if required_keys:
        missing_keys = [key for key in required_keys if key not in dict_value]
        if missing_keys:
            raise create_param_error(
                param_name, 
                f"missing required keys: {', '.join(missing_keys)}", 
                value, source=source
            )
    
    # Check allowed keys
    if allowed_keys:
        invalid_keys = [key for key in dict_value if key not in allowed_keys]
        if invalid_keys:
            raise create_param_error(
                param_name, 
                f"contains invalid keys: {', '.join(invalid_keys)}", 
                value, source=source
            )
    
    # Check key type
    if key_type is not None:
        invalid_keys = [key for key in dict_value if not isinstance(key, key_type)]
        if invalid_keys:
            keys_str = ", ".join(str(k) for k in invalid_keys[:5])
            if len(invalid_keys) > 5:
                keys_str += f" and {len(invalid_keys) - 5} more"
            raise create_param_error(
                param_name, 
                f"keys must be of type {key_type.__name__}, invalid keys: {keys_str}", 
                value, source=source
            )
    
    # Validate dictionary values
    if value_type is not None or value_validator is not None:
        validated_dict = {}
        for key, val in dict_value.items():
            # Check value type
            if value_type is not None and not isinstance(val, value_type):
                raise create_param_error(
                    f"{param_name}.{key}", 
                    f"must be of type {value_type.__name__}", 
                    val, source=source
                )
            
            # Apply value validator if provided
            if value_validator is not None:
                try:
                    validated_val = value_validator(val, f"{param_name}.{key}")
                    validated_dict[key] = validated_val
                except Exception as e:
                    raise create_param_error(
                        f"{param_name}.{key}", 
                        f"failed validation: {e}", 
                        val, source=source
                    )
            else:
                validated_dict[key] = val
        
        return validated_dict
    
    return dict_value


def validate_callable(
    value: Any,
    param_name: str,
    expected_args: Optional[List[str]] = None,
    expected_return_type: Optional[Type] = None,
    allow_none: bool = False,
    default: Optional[Callable] = None,
    source: Optional[str] = None
) -> Optional[Callable]:
    """
    Validate a callable parameter (function or method).
    
    Args:
        value: The parameter value to validate
        param_name: Name of the parameter
        expected_args: List of expected argument names
        expected_return_type: Expected return type
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        source: Source of the validation (for error reporting)
        
    Returns:
        The validated callable
        
    Raises:
        ParameterError: If validation fails
        
    Example:
        ```python
        # Validate a function with specific arguments
        transformer = validate_callable(transformer, 'transformer', 
                                       expected_args=['data', 'window'])
                                       
        # Validate a function with expected return type
        scorer = validate_callable(scorer, 'scorer', expected_return_type=float)
        ```
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise create_param_error(param_name, "must not be None", source=source)
    
    # Check if value is callable
    if not callable(value):
        raise create_param_error(
            param_name, "must be callable", value, source=source
        )
    
    # Check expected arguments
    if expected_args is not None:
        import inspect
        sig = inspect.signature(value)
        param_names = list(sig.parameters.keys())
        
        # Check if all expected arguments are present
        missing_args = [arg for arg in expected_args if arg not in param_names]
        if missing_args:
            raise create_param_error(
                param_name, 
                f"missing expected arguments: {', '.join(missing_args)}", 
                value, source=source
            )
    
    # Check return type if possible and expected
    if expected_return_type is not None:
        import inspect
        sig = inspect.signature(value)
        if sig.return_annotation != inspect.Signature.empty:
            if sig.return_annotation != expected_return_type:
                raise create_param_error(
                    param_name, 
                    f"should return {expected_return_type.__name__}, but returns {sig.return_annotation.__name__}", 
                    value, source=source
                )
    
    return value


#------------------------------------------------------------------------
# DataFrame Validation Functions
#------------------------------------------------------------------------

def check_dataframe_type(df: Any) -> pd.DataFrame:
    """
    Validate and return a DataFrame object.
    
    Args:
        df: Object to validate as DataFrame
        
    Returns:
        Input object if it is a DataFrame
        
    Raises:
        TypeError: If input is not a pandas DataFrame
        
    Example:
        ```python
        # Basic usage
        df = check_dataframe_type(data)
        ```
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
    return df


def validate_columns_exist(
    df: pd.DataFrame,
    required_columns: List[str],
    raise_error: bool = True
) -> Tuple[bool, List[str]]:
    """
    Check if required columns exist in DataFrame.
    
    Args:
        df: DataFrame to check
        required_columns: List of column names that must exist
        raise_error: Whether to raise an exception if missing columns
        
    Returns:
        Tuple of (success flag, list of missing columns)
        
    Raises:
        ValidationError: If raise_error is True and columns are missing
        
    Example:
        ```python
        # Check with exception
        validate_columns_exist(df, ['timestamp', 'close', 'volume'])
        
        # Check without exception
        columns_present, missing = validate_columns_exist(
            df, ['timestamp', 'close', 'volume'], raise_error=False
        )
        if not columns_present:
            logger.warning(f"Missing columns: {missing}")
        ```
    """
    df = check_dataframe_type(df)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns and raise_error:
        raise create_missing_columns_error(missing_columns)
    
    return len(missing_columns) == 0, missing_columns


def check_column_dtypes(
    df: pd.DataFrame,
    dtype_map: Dict[str, Type],
    coerce: bool = False
) -> pd.DataFrame:
    """
    Validate or coerce column data types.
    
    Args:
        df: DataFrame to validate
        dtype_map: Dictionary mapping column names to expected types
        coerce: Whether to coerce columns to expected types
        
    Returns:
        Original or coerced DataFrame
        
    Raises:
        ValidationError: If types don't match and coerce is False
        
    Example:
        ```python
        # Validate types
        check_column_dtypes(df, {'timestamp': 'datetime64[ns]', 'value': 'float64'})
        
        # Coerce types
        df = check_column_dtypes(
            df, 
            {'timestamp': 'datetime64[ns]', 'value': 'float64'},
            coerce=True
        )
        ```
    """
    df = check_dataframe_type(df)
    result = df.copy() if coerce else df
    errors = []
    
    for col, expected_type in dtype_map.items():
        if col not in result.columns:
            continue
            
        actual_type = result[col].dtype
        type_matches = pd.api.types.is_dtype_equal(actual_type, expected_type)
        
        if not type_matches:
            if coerce:
                try:
                    result[col] = result[col].astype(expected_type)
                except Exception as e:
                    errors.append(f"Column '{col}': {e}")
            else:
                errors.append(f"Column '{col}' has type {actual_type}, expected {expected_type}")
    
    if errors and not coerce:
        raise ValidationError("\n".join(errors))
        
    return result


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
    raise_exception: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Comprehensive DataFrame validation with multiple checks.
    
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
        raise_exception: Whether to raise exception on validation errors
        
    Returns:
        Tuple of (validated DataFrame, list of validation errors)
        
    Raises:
        DataValidationError: If validation fails and raise_exception is True
        
    Example:
        ```python
        # Validate with exception
        df, _ = validate_dataframe(
            df,
            required_columns=['timestamp', 'close', 'volume'],
            dtypes={'timestamp': 'datetime64[ns]', 'close': 'float64'},
            min_rows=100,
            check_index_type=pd.DatetimeIndex,
            check_sorted_index=True
        )
        
        # Validate without exception
        df, errors = validate_dataframe(
            df,
            required_columns=['timestamp', 'close', 'volume'],
            raise_exception=False
        )
        if errors:
            logger.warning(f"Validation issues: {errors}")
        ```
    """
    # Basic DataFrame validation
    try:
        df = check_dataframe_type(df)
    except TypeError as e:
        if raise_exception:
            raise DataValidationError(str(e))
        return df, [str(e)]
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    errors = []
    
    # Check required columns
    if required_columns:
        _, missing = validate_columns_exist(df_copy, required_columns, raise_error=False)
        if missing:
            errors.append(f"Missing required columns: {', '.join(missing)}")
    
    # Check data types
    if dtypes:
        for col, expected_type in dtypes.items():
            if col not in df_copy.columns:
                # Column was already reported as missing
                continue
                
            # Check if column type matches expected type
            try:
                actual_type = df_copy[col].dtype
                if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                    errors.append(f"Column '{col}' has incorrect type: expected {expected_type}, got {actual_type}")
            except Exception as e:
                errors.append(f"Error checking type for column '{col}': {e}")
    
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
    if raise_exception and errors:
        raise DataValidationError(
            "DataFrame validation failed",
            data_info={"shape": df_copy.shape, "columns": list(df_copy.columns)},
            errors=errors
        )
    
    return df_copy, errors


#------------------------------------------------------------------------
# Time Series Validation Functions
#------------------------------------------------------------------------

def ensure_datetime_index(
    df: pd.DataFrame, 
    datetime_col: Optional[str] = None,
    infer_from_columns: bool = True,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Ensure DataFrame has DatetimeIndex, converting if necessary.
    
    Args:
        df: DataFrame to validate
        datetime_col: Column name to use for datetime index
        infer_from_columns: Whether to try finding datetime columns
        inplace: Whether to modify the DataFrame in place
        
    Returns:
        DataFrame with datetime index
        
    Raises:
        ValueError: If cannot create datetime index
        
    Example:
        ```python
        # Using specified column
        df = ensure_datetime_index(df, datetime_col='timestamp')
        
        # Automatic detection
        df = ensure_datetime_index(df)
        ```
    """
    result = df if inplace else df.copy()
    
    # Already has DatetimeIndex
    if isinstance(result.index, pd.DatetimeIndex):
        return result
        
    # Use specified column
    if datetime_col is not None:
        if datetime_col not in result.columns:
            raise ValueError(f"Datetime column '{datetime_col}' not found in DataFrame")
        
        result[datetime_col] = pd.to_datetime(result[datetime_col])
        return result.set_index(datetime_col)
        
    # Try to infer from columns if allowed
    if infer_from_columns:
        # First, try common datetime column names
        common_names = ['timestamp', 'time', 'date', 'datetime']
        for col in common_names:
            if col in result.columns:
                result[col] = pd.to_datetime(result[col])
                return result.set_index(col)
        
        # Next, try columns with datetime-like names
        time_cols = [col for col in result.columns 
                    if any(kw in col.lower() for kw in ['time', 'date', 'dt', 'timestamp'])]
        if time_cols:
            col = time_cols[0]
            result[col] = pd.to_datetime(result[col])
            return result.set_index(col)
    
    # Try to convert existing index
    try:
        result.index = pd.to_datetime(result.index)
        return result
    except:
        raise ValueError(
            "Could not create DatetimeIndex. Please provide a valid datetime column "
            "or ensure the index is convertible to datetime."
        )


def validate_time_index_properties(
    df: pd.DataFrame,
    min_points: Optional[int] = None,
    max_gaps: Optional[int] = None,
    max_gap_size: Optional[timedelta] = None,
    freq: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate properties of a DatetimeIndex.
    
    Args:
        df: DataFrame to validate
        min_points: Minimum number of data points required
        max_gaps: Maximum number of gaps allowed
        max_gap_size: Maximum size of time gaps allowed
        freq: Expected frequency of the time series
        
    Returns:
        Tuple of (success flag, list of validation errors)
        
    Example:
        ```python
        # Basic validation
        is_valid, errors = validate_time_index_properties(
            df, 
            min_points=100,
            max_gaps=5,
            max_gap_size=timedelta(days=1)
        )
        
        if not is_valid:
            logger.warning(f"Time index issues: {errors}")
        ```
    """
    # Check if index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, ["Index is not DatetimeIndex"]
        
    errors = []
    
    # Validate size
    if min_points is not None and len(df) < min_points:
        errors.append(f"Time series has {len(df)} points, minimum required is {min_points}")
        
    # Check for gaps
    if max_gaps is not None or max_gap_size is not None:
        if len(df) > 1:
            # Calculate time differences
            idx = df.index
            diffs = idx[1:] - idx[:-1]
            
            if max_gap_size is not None:
                large_gaps = diffs > max_gap_size
                gap_count = large_gaps.sum()
                gap_positions = np.where(large_gaps)[0]
                
                if gap_count > 0:
                    if max_gaps is not None and gap_count > max_gaps:
                        # List some of the gaps
                        gap_examples = []
                        for i in gap_positions[:min(3, len(gap_positions))]:
                            start_time = idx[i]
                            end_time = idx[i + 1]
                            gap_examples.append(f"{start_time} to {end_time} ({diffs[i]})")
                        
                        errors.append(
                            f"Found {gap_count} gaps larger than {max_gap_size}. "
                            f"Examples: {', '.join(gap_examples)}"
                        )
    
    # Check frequency if specified
    if freq is not None:
        # Try to infer frequency
        inferred_freq = pd.infer_freq(df.index)
        
        if inferred_freq != freq:
            errors.append(f"Index frequency is {inferred_freq}, expected {freq}")
    
    return len(errors) == 0, errors


def detect_time_series_frequency(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the frequency of time series data.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        Detected frequency as pandas frequency string or None if can't be detected
        
    Raises:
        TypeError: If DataFrame doesn't have a datetime index
        
    Example:
        ```python
        # Detect frequency
        freq = detect_time_series_frequency(df)
        print(f"Detected frequency: {freq}")
        ```
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a datetime index to detect frequency")
    
    # Try pandas infer_freq first
    freq = pd.infer_freq(df.index)
    
    if freq is not None:
        return freq
        
    # Calculate time deltas and get the most common
    if len(df) > 1:
        # Get time differences
        time_diffs = df.index[1:] - df.index[:-1]
        
        if len(time_diffs) > 0:
            # Find the most common difference
            most_common_diff = pd.Series(time_diffs).value_counts().index[0]
            
            # Convert to pandas frequency string (approximate)
            seconds = most_common_diff.total_seconds()
            
            if seconds < 60:
                return f"{int(seconds)}S"
            elif seconds < 3600:
                minutes = int(seconds / 60)
                return f"{minutes}T"
            elif seconds < 86400:
                hours = int(seconds / 3600)
                return f"{hours}H"
            elif 86400 <= seconds < 604800:
                days = int(seconds / 86400)
                return f"{days}D"
            elif 604800 <= seconds < 2592000:
                weeks = int(seconds / 604800)
                return f"{weeks}W"
            else:
                months = int(seconds / 2592000)
                return f"{months}M"
    
    # Could not detect frequency
    return None


#------------------------------------------------------------------------
# Error Collection and Handling
#------------------------------------------------------------------------

class ValidationErrorCollector:
    """
    Utility class to collect validation errors.
    
    This class provides a convenient way to collect validation errors
    during complex validation processes, and either return them or
    raise exceptions as needed.
    
    Attributes:
        errors (List[str]): List of collected error messages
        raise_on_error (bool): Whether to raise exceptions immediately
        
    Example:
        ```python
        # Collect errors without raising exceptions
        collector = ValidationErrorCollector()
        
        # Validate multiple aspects of data
        if len(df) < 100:
            collector.add_error("Insufficient data: needs at least 100 rows")
            
        if 'timestamp' not in df.columns:
            collector.add_error("Missing required column: 'timestamp'")
            
        # Check if we have any errors
        if collector.has_errors():
            logger.warning(f"Validation failed: {collector.get_error_message()}")
            return None
            
        # Continue with valid data
        return process_data(df)
        ```
    """
    
    def __init__(self, raise_on_error: bool = False):
        """
        Initialize the error collector.
        
        Args:
            raise_on_error: Whether to raise exceptions immediately when errors are added
        """
        self.errors = []
        self.raise_on_error = raise_on_error
        
    def add_error(self, message: str):
        """
        Add an error message to the collection.
        
        Args:
            message: Error message to add
            
        Raises:
            ValidationError: If raise_on_error is True
        """
        self.errors.append(message)
        
        if self.raise_on_error:
            raise ValidationError(message)
            
    def add_errors(self, errors: List[str]):
        """
        Add multiple error messages.
        
        Args:
            errors: List of error messages to add
            
        Raises:
            ValidationError: If raise_on_error is True and errors list is not empty
        """
        if not errors:
            return
            
        self.errors.extend(errors)
        
        if self.raise_on_error:
            raise ValidationError("\n".join(errors))
            
    def has_errors(self) -> bool:
        """
        Check if any errors were collected.
        
        Returns:
            True if errors were collected, False otherwise
        """
        return len(self.errors) > 0
        
    def raise_if_errors(self, error_class: Type[Exception] = ValidationError):
        """
        Raise an exception if errors were collected.
        
        Args:
            error_class: Exception class to raise
            
        Raises:
            Exception: Of the specified class if errors were collected
        """
        if self.has_errors():
            raise error_class("\n".join(self.errors))
            
    def get_error_message(self, separator: str = "\n") -> str:
        """
        Get combined error message.
        
        Args:
            separator: String to use for joining error messages
            
        Returns:
            Combined error message
        """
        return separator.join(self.errors)
        
    def reset(self):
        """Reset the error collection."""
        self.errors = []
        
    def __bool__(self) -> bool:
        """Boolean conversion returns True if no errors."""
        return not self.has_errors()
        
    def __str__(self) -> str:
        """String representation shows errors or success message."""
        if self.has_errors():
            return f"Validation errors ({len(self.errors)}): {self.get_error_message()}"
        return "Validation successful (no errors)"