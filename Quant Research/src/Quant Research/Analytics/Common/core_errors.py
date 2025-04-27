"""
Core Error Classes and Utilities

This module defines a standardized error hierarchy and utilities for error handling
throughout the quant research package. It provides consistent error formatting,
specialized error types, and helper functions to reduce code duplication.

The module follows a hierarchical structure with QuantError as the base class,
and specialized subclasses for different error categories.

Usage:
    ```python
    from quant_research.core.errors import (
        QuantError, ValidationError, DataError,
        create_param_error, create_missing_columns_error
    )
    
    # Raise simple error
    raise ValidationError("Invalid input data")
    
    # Use helper to create consistent parameter errors
    raise create_param_error("window_size", "must be a positive integer")
    
    # Create data error with details
    raise DataError(
        "Failed to process data",
        code="DATA_PROCESSING_ERROR",
        details={"file": "data.csv", "reason": "corrupt file"}
    )
    ```
"""

# Standard library imports
import logging
import sys
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Type

# Configure module logger
logger = logging.getLogger("quant_research.core.errors")


class ErrorCode(str, Enum):
    """Standard error codes for the quant research package."""
    
    # General errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    
    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    TYPE_ERROR = "TYPE_ERROR"
    VALUE_ERROR = "VALUE_ERROR"
    
    # Parameter errors
    PARAM_ERROR = "PARAM_ERROR"
    MISSING_PARAM = "MISSING_PARAM"
    INVALID_PARAM = "INVALID_PARAM"
    INCOMPATIBLE_PARAMS = "INCOMPATIBLE_PARAMS"
    
    # Data errors
    DATA_ERROR = "DATA_ERROR"
    DATA_FORMAT_ERROR = "DATA_FORMAT_ERROR"
    MISSING_DATA = "MISSING_DATA"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    DATA_QUALITY_ERROR = "DATA_QUALITY_ERROR"
    
    # API and I/O errors
    IO_ERROR = "IO_ERROR"
    API_ERROR = "API_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    
    # Computation errors
    COMPUTATION_ERROR = "COMPUTATION_ERROR"
    CONVERGENCE_ERROR = "CONVERGENCE_ERROR"
    NUMERICAL_ERROR = "NUMERICAL_ERROR"
    
    # Logic errors
    LOGIC_ERROR = "LOGIC_ERROR"
    STATE_ERROR = "STATE_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"


class QuantError(Exception):
    """
    Base exception class for all quant research errors.
    
    This class standardizes error formatting and provides additional
    context for debugging and user feedback. All other error classes
    should inherit from this base class.
    
    Attributes:
        message: Primary error message
        code: Error code for categorization
        param_name: Name of the parameter that caused the error (if applicable)
        source: Source of the error (component, module, or function)
        details: Additional error details for debugging
        timestamp: When the error occurred
    """
    
    def __init__(
        self, 
        message: str, 
        code: Union[str, ErrorCode] = ErrorCode.UNKNOWN_ERROR,
        param_name: Optional[str] = None,
        source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize the error with context.
        
        Args:
            message: Primary error message
            code: Error code for categorization
            param_name: Name of the parameter that caused the error
            source: Source of the error (component, module, or function)
            details: Additional error details for debugging
            timestamp: When the error occurred (defaults to now)
        """
        self.message = message
        self.code = code.value if isinstance(code, ErrorCode) else code
        self.param_name = param_name
        self.source = source
        self.details = details or {}
        self.timestamp = timestamp or datetime.now()
        
        # Format the full error message
        parts = []
        
        if self.code:
            parts.append(f"[{self.code}]")
        
        parts.append(self.message)
        
        if self.param_name:
            parts.append(f"(parameter: {self.param_name})")
        
        if self.source:
            parts.append(f"[source: {self.source}]")
        
        full_message = " ".join(parts)
        super().__init__(full_message)
    
    def log(self, level: int = logging.ERROR) -> None:
        """
        Log this error at the specified level.
        
        Args:
            level: Logging level (default: ERROR)
        """
        logger.log(level, str(self), exc_info=self)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the error
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "param_name": self.param_name,
            "source": self.source,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


#------------------------------------------------------------------------
# Validation Errors
#------------------------------------------------------------------------

class ValidationError(QuantError):
    """
    Exception raised for validation errors.
    
    This error is raised when input validation fails and should be caught
    by the calling function to handle appropriately.
    """
    
    def __init__(
        self, 
        message: str, 
        code: Union[str, ErrorCode] = ErrorCode.VALIDATION_ERROR,
        param_name: Optional[str] = None,
        source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the validation error."""
        super().__init__(message, code, param_name, source, details)


class DataValidationError(ValidationError):
    """
    Exception raised for data validation errors.
    
    This error is raised when input data validation fails and includes
    information about the specific data issues.
    """
    
    def __init__(
        self, 
        message: str, 
        data_info: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None,
        code: Union[str, ErrorCode] = ErrorCode.DATA_FORMAT_ERROR,
        source: Optional[str] = None
    ):
        """
        Initialize the data validation error.
        
        Args:
            message: Primary error message
            data_info: Information about the data being validated
            errors: List of specific validation errors
            code: Error code
            source: Source of the error
        """
        self.data_info = data_info or {}
        self.errors = errors or []
        
        # Format the error message
        full_message = message
        if errors:
            full_message = f"{message}: {'; '.join(errors)}"
            
        details = {
            "data_info": self.data_info,
            "errors": self.errors
        }
        
        super().__init__(full_message, code, source=source or "data_validation", details=details)


class SchemaValidationError(ValidationError):
    """
    Exception raised for schema validation errors.
    
    This error is raised when data does not conform to the expected schema.
    """
    
    def __init__(
        self, 
        message: str, 
        schema_errors: Optional[Dict[str, str]] = None,
        schema_name: Optional[str] = None,
        code: Union[str, ErrorCode] = ErrorCode.SCHEMA_ERROR,
        source: Optional[str] = None
    ):
        """
        Initialize the schema validation error.
        
        Args:
            message: Primary error message
            schema_errors: Dictionary mapping fields to error messages
            schema_name: Name of the schema for reference
            code: Error code
            source: Source of the error
        """
        self.schema_errors = schema_errors or {}
        self.schema_name = schema_name
        
        # Format the error message
        full_message = message
        if schema_errors:
            error_details = "; ".join([f"{field}: {error}" for field, error in schema_errors.items()])
            full_message = f"{message}: {error_details}"
            
        details = {
            "schema_errors": self.schema_errors
        }
        
        if schema_name:
            details["schema_name"] = schema_name
            
        super().__init__(full_message, code, source=source or "schema_validation", details=details)


class ParameterError(ValidationError):
    """
    Exception raised for parameter validation errors.
    
    This error is raised when a function or method parameter fails validation.
    """
    
    def __init__(
        self, 
        message: str, 
        param_name: str,
        value: Any = None,
        expected: Any = None,
        code: Union[str, ErrorCode] = ErrorCode.PARAM_ERROR,
        source: Optional[str] = None
    ):
        """
        Initialize the parameter error.
        
        Args:
            message: Primary error message
            param_name: Name of the invalid parameter
            value: Invalid value (if applicable)
            expected: Expected value or type (if applicable)
            code: Error code
            source: Source of the error
        """
        details = {}
        
        if value is not None:
            details["value"] = str(value)
            
        if expected is not None:
            if isinstance(expected, type):
                details["expected_type"] = expected.__name__
            else:
                details["expected"] = str(expected)
        
        super().__init__(message, code, param_name, source, details)


#------------------------------------------------------------------------
# Data Errors
#------------------------------------------------------------------------

class DataError(QuantError):
    """
    Exception raised for data-related errors.
    
    This error is raised when there are issues with data processing,
    missing data, data quality, etc.
    """
    
    def __init__(
        self, 
        message: str, 
        code: Union[str, ErrorCode] = ErrorCode.DATA_ERROR,
        source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the data error."""
        super().__init__(message, code, source=source, details=details)


class MissingDataError(DataError):
    """
    Exception raised when required data is missing.
    
    This error is raised when required data fields or files are missing.
    """
    
    def __init__(
        self, 
        message: str, 
        missing_items: Union[List[str], Set[str]],
        code: Union[str, ErrorCode] = ErrorCode.MISSING_DATA,
        source: Optional[str] = None
    ):
        """
        Initialize the missing data error.
        
        Args:
            message: Primary error message
            missing_items: List of missing data items
            code: Error code
            source: Source of the error
        """
        details = {"missing_items": list(missing_items)}
        super().__init__(message, code, source, details)


class InsufficientDataError(DataError):
    """
    Exception raised when there is not enough data for processing.
    
    This error is raised when the data is present but insufficient for
    the requested operation (e.g., too few samples for statistical significance).
    """
    
    def __init__(
        self, 
        message: str, 
        actual_size: int,
        required_size: int,
        code: Union[str, ErrorCode] = ErrorCode.INSUFFICIENT_DATA,
        source: Optional[str] = None
    ):
        """
        Initialize the insufficient data error.
        
        Args:
            message: Primary error message
            actual_size: Actual amount of data available
            required_size: Minimum amount of data required
            code: Error code
            source: Source of the error
        """
        details = {
            "actual_size": actual_size,
            "required_size": required_size,
            "missing": required_size - actual_size
        }
        super().__init__(message, code, source, details)


class DataQualityError(DataError):
    """
    Exception raised for data quality issues.
    
    This error is raised when the data has quality issues such as
    missing values, outliers, or other anomalies.
    """
    
    def __init__(
        self, 
        message: str, 
        quality_issues: Dict[str, Any],
        code: Union[str, ErrorCode] = ErrorCode.DATA_QUALITY_ERROR,
        source: Optional[str] = None
    ):
        """
        Initialize the data quality error.
        
        Args:
            message: Primary error message
            quality_issues: Dictionary with details about quality issues
            code: Error code
            source: Source of the error
        """
        details = {"quality_issues": quality_issues}
        super().__init__(message, code, source, details)


#------------------------------------------------------------------------
# Computational Errors
#------------------------------------------------------------------------

class ComputationError(QuantError):
    """
    Exception raised for computation-related errors.
    
    This error is raised when there are issues with numerical calculations,
    algorithm convergence, or other computational problems.
    """
    
    def __init__(
        self, 
        message: str, 
        code: Union[str, ErrorCode] = ErrorCode.COMPUTATION_ERROR,
        algorithm: Optional[str] = None,
        source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the computation error.
        
        Args:
            message: Primary error message
            code: Error code
            algorithm: Name of the algorithm that failed
            source: Source of the error
            details: Additional error details
        """
        error_details = details or {}
        
        if algorithm:
            error_details["algorithm"] = algorithm
            
        super().__init__(message, code, source=source, details=error_details)


class ConvergenceError(ComputationError):
    """
    Exception raised when an algorithm fails to converge.
    
    This error is raised when an iterative algorithm fails to reach
    a stable solution within the allowed iterations or tolerance.
    """
    
    def __init__(
        self, 
        message: str, 
        iterations: int,
        tolerance: Optional[float] = None,
        algorithm: Optional[str] = None,
        code: Union[str, ErrorCode] = ErrorCode.CONVERGENCE_ERROR,
        source: Optional[str] = None
    ):
        """
        Initialize the convergence error.
        
        Args:
            message: Primary error message
            iterations: Number of iterations performed
            tolerance: Convergence tolerance (if applicable)
            algorithm: Name of the algorithm that failed to converge
            code: Error code
            source: Source of the error
        """
        details = {"iterations": iterations}
        
        if tolerance is not None:
            details["tolerance"] = tolerance
            
        super().__init__(message, code, algorithm, source, details)


#------------------------------------------------------------------------
# I/O and API Errors
#------------------------------------------------------------------------

class IOError(QuantError):
    """
    Exception raised for I/O-related errors.
    
    This error is raised when there are issues with file I/O,
    network connections, or external API calls.
    """
    
    def __init__(
        self, 
        message: str, 
        code: Union[str, ErrorCode] = ErrorCode.IO_ERROR,
        path: Optional[str] = None,
        operation: Optional[str] = None,
        source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the I/O error.
        
        Args:
            message: Primary error message
            code: Error code
            path: File or resource path (if applicable)
            operation: I/O operation that failed (read, write, etc.)
            source: Source of the error
            details: Additional error details
        """
        error_details = details or {}
        
        if path:
            error_details["path"] = path
            
        if operation:
            error_details["operation"] = operation
            
        super().__init__(message, code, source=source, details=error_details)


class APIError(IOError):
    """
    Exception raised for API-related errors.
    
    This error is raised when there are issues with external API calls,
    such as rate limiting, authentication failures, or unexpected responses.
    """
    
    def __init__(
        self, 
        message: str, 
        endpoint: str,
        status_code: Optional[int] = None,
        response: Optional[Any] = None,
        code: Union[str, ErrorCode] = ErrorCode.API_ERROR,
        source: Optional[str] = None
    ):
        """
        Initialize the API error.
        
        Args:
            message: Primary error message
            endpoint: API endpoint that was called
            status_code: HTTP status code (if applicable)
            response: API response (if available)
            code: Error code
            source: Source of the error
        """
        details = {"endpoint": endpoint}
        
        if status_code is not None:
            details["status_code"] = status_code
            
        if response is not None:
            try:
                # Try to convert response to string or dict for logging
                if hasattr(response, 'json'):
                    details["response"] = response.json()
                elif hasattr(response, 'text'):
                    details["response"] = response.text
                else:
                    details["response"] = str(response)
            except:
                details["response"] = "Unable to parse response"
        
        super().__init__(message, code, path=endpoint, operation="api_call", source=source, details=details)


#------------------------------------------------------------------------
# Logic Errors
#------------------------------------------------------------------------

class LogicError(QuantError):
    """
    Exception raised for logical errors in the application.
    
    This error is raised when there is a logical inconsistency or
    invalid state in the application.
    """
    
    def __init__(
        self, 
        message: str, 
        code: Union[str, ErrorCode] = ErrorCode.LOGIC_ERROR,
        source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the logic error."""
        super().__init__(message, code, source=source, details=details)


class StateError(LogicError):
    """
    Exception raised for invalid object state.
    
    This error is raised when an operation is attempted on an object
    that is in an invalid state for that operation.
    """
    
    def __init__(
        self, 
        message: str, 
        current_state: Optional[str] = None,
        expected_state: Optional[str] = None,
        code: Union[str, ErrorCode] = ErrorCode.STATE_ERROR,
        source: Optional[str] = None
    ):
        """
        Initialize the state error.
        
        Args:
            message: Primary error message
            current_state: Current state of the object
            expected_state: Expected state for the operation
            code: Error code
            source: Source of the error
        """
        details = {}
        
        if current_state is not None:
            details["current_state"] = current_state
            
        if expected_state is not None:
            details["expected_state"] = expected_state
            
        super().__init__(message, code, source, details)


class DependencyError(LogicError):
    """
    Exception raised for missing dependencies.
    
    This error is raised when a required dependency is missing or
    incompatible with the current environment.
    """
    
    def __init__(
        self, 
        message: str, 
        dependency: str,
        required_version: Optional[str] = None,
        code: Union[str, ErrorCode] = ErrorCode.DEPENDENCY_ERROR,
        source: Optional[str] = None
    ):
        """
        Initialize the dependency error.
        
        Args:
            message: Primary error message
            dependency: Name of the missing or incompatible dependency
            required_version: Required version of the dependency (if applicable)
            code: Error code
            source: Source of the error
        """
        details = {"dependency": dependency}
        
        if required_version is not None:
            details["required_version"] = required_version
            
        super().__init__(message, code, source, details)


#------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------

def create_param_error(
    param_name: str, 
    reason: str, 
    value: Any = None,
    expected: Any = None,
    code: Union[str, ErrorCode] = ErrorCode.PARAM_ERROR,
    source: Optional[str] = None
) -> ParameterError:
    """
    Create a standardized parameter error.
    
    Args:
        param_name: Name of the invalid parameter
        reason: Reason why the parameter is invalid
        value: Invalid value (if applicable)
        expected: Expected value or type (if applicable)
        code: Error code
        source: Source of the error
        
    Returns:
        ParameterError instance
    """
    message = f"Invalid parameter '{param_name}': {reason}"
    return ParameterError(message, param_name, value, expected, code, source)


def create_missing_param_error(
    param_name: str,
    code: Union[str, ErrorCode] = ErrorCode.MISSING_PARAM,
    source: Optional[str] = None
) -> ParameterError:
    """
    Create a standardized missing parameter error.
    
    Args:
        param_name: Name of the missing parameter
        code: Error code
        source: Source of the error
        
    Returns:
        ParameterError instance
    """
    message = f"Missing required parameter '{param_name}'"
    return ParameterError(message, param_name, None, None, code, source)


def create_type_error(
    param_name: str,
    value: Any,
    expected_type: Union[Type, List[Type]],
    code: Union[str, ErrorCode] = ErrorCode.TYPE_ERROR,
    source: Optional[str] = None
) -> ParameterError:
    """
    Create a standardized type error.
    
    Args:
        param_name: Name of the parameter with wrong type
        value: Invalid value
        expected_type: Expected type or list of types
        code: Error code
        source: Source of the error
        
    Returns:
        ParameterError instance
    """
    if isinstance(expected_type, list):
        type_names = [t.__name__ for t in expected_type]
        expected_str = ", ".join(type_names)
        message = f"Parameter '{param_name}' must be one of: {expected_str}, got {type(value).__name__}"
    else:
        message = f"Parameter '{param_name}' must be {expected_type.__name__}, got {type(value).__name__}"
    
    return ParameterError(message, param_name, value, expected_type, code, source)


def create_value_error(
    param_name: str,
    value: Any,
    constraint: str,
    code: Union[str, ErrorCode] = ErrorCode.VALUE_ERROR,
    source: Optional[str] = None
) -> ParameterError:
    """
    Create a standardized value error.
    
    Args:
        param_name: Name of the parameter with invalid value
        value: Invalid value
        constraint: Description of the constraint that was violated
        code: Error code
        source: Source of the error
        
    Returns:
        ParameterError instance
    """
    message = f"Parameter '{param_name}' value {value} is invalid: {constraint}"
    return ParameterError(message, param_name, value, None, code, source)


def create_missing_columns_error(
    missing: Union[List[str], Set[str]],
    code: Union[str, ErrorCode] = ErrorCode.MISSING_DATA,
    source: Optional[str] = None
) -> MissingDataError:
    """
    Create a standardized missing columns error.
    
    Args:
        missing: List of missing column names
        code: Error code
        source: Source of the error
        
    Returns:
        MissingDataError instance
    """
    columns_str = ", ".join(missing)
    message = f"Missing required columns: {columns_str}"
    return MissingDataError(message, missing, code, source)


def create_insufficient_data_error(
    actual_size: int,
    required_size: int,
    message: Optional[str] = None,
    code: Union[str, ErrorCode] = ErrorCode.INSUFFICIENT_DATA,
    source: Optional[str] = None
) -> InsufficientDataError:
    """
    Create a standardized insufficient data error.
    
    Args:
        actual_size: Actual amount of data available
        required_size: Minimum amount of data required
        message: Custom error message (optional)
        code: Error code
        source: Source of the error
        
    Returns:
        InsufficientDataError instance
    """
    if message is None:
        message = f"Insufficient data: got {actual_size} rows, need at least {required_size}"
    
    return InsufficientDataError(message, actual_size, required_size, code, source)


def handle_exception(
    exception: Exception, 
    log_level: int = logging.ERROR,
    reraise: bool = True,
    convert_to: Optional[Type[QuantError]] = None,
    fallback_message: str = "An unexpected error occurred",
    source: Optional[str] = None
) -> Optional[QuantError]:
    """
    Handle an exception with standardized logging and optional conversion.
    
    Args:
        exception: The exception to handle
        log_level: Logging level for this exception
        reraise: Whether to re-raise the exception after handling
        convert_to: QuantError subclass to convert the exception to
        fallback_message: Message to use if converting a non-QuantError
        source: Source to use if converting a non-QuantError
        
    Returns:
        Converted exception if convert_to is specified and reraise is False
        
    Raises:
        The original or converted exception if reraise is True
    """
    # Extract traceback info
    _, _, tb = sys.exc_info()
    tb_str = "".join(traceback.format_tb(tb))
    
    # Get current function name as source if not provided
    if source is None:
        frame = traceback.extract_stack()[-2]
        source = f"{frame.filename}:{frame.lineno} in {frame.name}"
    
    # Handle QuantError subclasses
    if isinstance(exception, QuantError):
        # Log the error with its original source
        logger.log(log_level, str(exception), exc_info=exception)
        
        # Convert to another type if requested
        if convert_to is not None and not isinstance(exception, convert_to):
            converted = convert_to(
                message=exception.message,
                code=exception.code,
                source=exception.source or source,
                details=dict(exception.details, original_error=str(exception), traceback=tb_str)
            )
            
            if reraise:
                raise converted from exception
            return converted
        
        # Re-raise the original exception if requested
        if reraise:
            raise
        return exception
    
    # Handle non-QuantError exceptions
    else:
        # Create error message from exception
        error_message = str(exception) or fallback_message
        
        # Get appropriate error code based on exception type
        if isinstance(exception, ValueError):
            code = ErrorCode.VALUE_ERROR
        elif isinstance(exception, TypeError):
            code = ErrorCode.TYPE_ERROR
        elif isinstance(exception, KeyError) or isinstance(exception, AttributeError):
            code = ErrorCode.MISSING_DATA
        else:
            code = ErrorCode.INTERNAL_ERROR
        
        # Create a QuantError or specified subclass
        error_class = convert_to or QuantError
        converted = error_class(
            message=error_message,
            code=code,
            source=source,
            details={"original_error": str(exception), "traceback": tb_str}
        )
        
        # Log the converted error
        logger.log(log_level, str(converted), exc_info=exception)
        
        # Re-raise the converted exception if requested
        if reraise:
            raise converted from exception
        return converted


def validate_not_none(value: Any, param_name: str, source: Optional[str] = None) -> Any:
    """
    Validate that a value is not None.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        source: Source of the validation
        
    Returns:
        The original value if not None
        
    Raises:
        ParameterError: If the value is None
    """
    if value is None:
        raise create_missing_param_error(param_name, source=source)
    return value


def validate_type(
    value: Any, 
    expected_type: Union[Type, Tuple[Type, ...]], 
    param_name: str,
    source: Optional[str] = None,
    allow_none: bool = False
) -> Any:
    """
    Validate that a value is of the expected type.
    
    Args:
        value: Value to validate
        expected_type: Expected type or tuple of types
        param_name: Name of the parameter
        source: Source of the validation
        allow_none: Whether None is an allowed value
        
    Returns:
        The original value if of the correct type
        
    Raises:
        ParameterError: If the value is of the wrong type
    """
    if value is None:
        if allow_none:
            return None
        raise create_missing_param_error(param_name, source=source)
    
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = [t.__name__ for t in expected_type]
            expected_str = ", ".join(type_names)
            message = f"Parameter '{param_name}' must be one of: {expected_str}, got {type(value).__name__}"
        else:
            message = f"Parameter '{param_name}' must be {expected_type.__name__}, got {type(value).__name__}"
        
        raise ParameterError(
            message=message,
            param_name=param_name,
            value=value,
            expected=expected_type,
            code=ErrorCode.TYPE_ERROR,
            source=source
        )
    
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
    Validate a numeric parameter.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_none: Whether None is an allowed value
        default: Default value to use if value is None and allow_none is False
        integer_only: Whether to require integer values
        source: Source of the validation
        
    Returns:
        Validated numeric value (float, int, or None)
        
    Raises:
        ParameterError: If validation fails
    """
    # Handle None value
    if value is None:
        if allow_none:
            return None
        elif default is not None:
            return default
        else:
            raise create_missing_param_error(param_name, source=source)
    
    # Check if value is numeric
    try:
        if integer_only:
            # For integers, convert and check if conversion preserves value
            numeric_value = int(value)
            if float(numeric_value) != float(value):
                raise create_param_error(
                    param_name=param_name,
                    reason="must be an integer",
                    value=value,
                    expected="integer",
                    source=source
                )
        else:
            numeric_value = float(value)
    except (ValueError, TypeError):
        raise create_param_error(
            param_name=param_name,
            reason="must be numeric",
            value=value,
            expected="number",
            source=source
        )
    
    # Check minimum value
    if min_value is not None and numeric_value < min_value:
        raise create_param_error(
            param_name=param_name,
            reason=f"must be at least {min_value}",
            value=numeric_value,
            expected=f">= {min_value}",
            source=source
        )
    
    # Check maximum value
    if max_value is not None and numeric_value > max_value:
        raise create_param_error(
            param_name=param_name,
            reason=f"must be at most {max_value}",
            value=numeric_value,
            expected=f"<= {max_value}",
            source=source
        )
    
    return numeric_value