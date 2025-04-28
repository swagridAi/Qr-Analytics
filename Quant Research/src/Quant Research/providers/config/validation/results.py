# src/quant_research/core/config/validation/result.py
"""
Models for representing validation results and reports.

This module provides classes for capturing validation outcomes, aggregating
results, and generating comprehensive validation reports.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Iterable, Set


class ValidationSeverity(str, Enum):
    """Severity levels for validation results"""
    ERROR = "error"       # Validation failure that must be fixed
    WARNING = "warning"   # Potential issue to consider fixing
    INFO = "info"         # Informational message about validation


class ValidationResult:
    """
    Result of a single validation check.
    
    Captures the outcome of validating a single aspect of a configuration,
    including whether it passed, any error message, and the severity level.
    """
    
    def __init__(
        self,
        is_valid: bool,
        message: str,
        severity: str = ValidationSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        validator_name: Optional[str] = None,
        field_name: Optional[str] = None
    ):
        """
        Initialize a validation result.
        
        Args:
            is_valid: Whether the validation passed
            message: Descriptive message about the validation
            severity: Severity level (error, warning, info)
            details: Optional dictionary with additional details
            validator_name: Name of the validator that produced this result
            field_name: Name of the field being validated
        """
        self.is_valid = is_valid
        self.message = message
        self.severity = severity if isinstance(severity, ValidationSeverity) else ValidationSeverity(severity)
        self.details = details or {}
        self.validator_name = validator_name
        self.field_name = field_name
    
    def __str__(self) -> str:
        """String representation of the validation result"""
        status = "PASS" if self.is_valid else "FAIL"
        field_info = f"[{self.field_name}] " if self.field_name else ""
        return f"{status} ({self.severity}): {field_info}{self.message}"
    
    @property
    def is_error(self) -> bool:
        """Check if this result is an error"""
        return self.severity == ValidationSeverity.ERROR and not self.is_valid
    
    @property
    def is_warning(self) -> bool:
        """Check if this result is a warning"""
        return self.severity == ValidationSeverity.WARNING and not self.is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "is_valid": self.is_valid,
            "message": self.message,
            "severity": self.severity.value,
            "details": self.details,
            "validator_name": self.validator_name,
            "field_name": self.field_name
        }


class ValidationResults:
    """
    Collection of validation results.
    
    Manages multiple validation results and provides methods for
    querying and summarizing the validation outcomes.
    """
    
    def __init__(self, results: Optional[Iterable[ValidationResult]] = None):
        """
        Initialize a validation results collection.
        
        Args:
            results: Optional initial results to include
        """
        self._results: List[ValidationResult] = []
        self._field_results: Dict[str, List[ValidationResult]] = {}
        self._validator_results: Dict[str, List[ValidationResult]] = {}
        
        if results:
            for result in results:
                self.add_result(result)
    
    def add_result(self, result: ValidationResult) -> None:
        """
        Add a validation result to the collection.
        
        Args:
            result: Validation result to add
        """
        self._results.append(result)
        
        # Index by field name
        if result.field_name:
            if result.field_name not in self._field_results:
                self._field_results[result.field_name] = []
            self._field_results[result.field_name].append(result)
        
        # Index by validator name
        if result.validator_name:
            if result.validator_name not in self._validator_results:
                self._validator_results[result.validator_name] = []
            self._validator_results[result.validator_name].append(result)
    
    def add_results(self, results: Iterable[ValidationResult]) -> None:
        """
        Add multiple validation results to the collection.
        
        Args:
            results: Validation results to add
        """
        for result in results:
            self.add_result(result)
    
    @property
    def is_valid(self) -> bool:
        """Check if all validations passed (no errors)"""
        return not any(result.is_error for result in self._results)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return any(result.is_warning for result in self._results)
    
    @property
    def all_passed(self) -> bool:
        """Check if all validations passed (including warnings)"""
        return all(result.is_valid for result in self._results)
    
    @property
    def count(self) -> int:
        """Total number of validation results"""
        return len(self._results)
    
    @property
    def errors(self) -> List[ValidationResult]:
        """List of all error results"""
        return [r for r in self._results if r.is_error]
    
    @property
    def warnings(self) -> List[ValidationResult]:
        """List of all warning results"""
        return [r for r in self._results if r.is_warning]
    
    @property
    def passed(self) -> List[ValidationResult]:
        """List of all passed results"""
        return [r for r in self._results if r.is_valid]
    
    @property
    def error_count(self) -> int:
        """Number of errors"""
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        """Number of warnings"""
        return len(self.warnings)
    
    @property
    def field_names(self) -> Set[str]:
        """Set of all field names in the results"""
        return {r.field_name for r in self._results if r.field_name is not None}
    
    @property
    def validator_names(self) -> Set[str]:
        """Set of all validator names in the results"""
        return {r.validator_name for r in self._results if r.validator_name is not None}
    
    def get_results_for_field(self, field_name: str) -> List[ValidationResult]:
        """
        Get validation results for a specific field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            List of validation results for the field
        """
        return self._field_results.get(field_name, [])
    
    def get_results_for_validator(self, validator_name: str) -> List[ValidationResult]:
        """
        Get validation results from a specific validator.
        
        Args:
            validator_name: Name of the validator
            
        Returns:
            List of validation results from the validator
        """
        return self._validator_results.get(validator_name, [])
    
    def get_field_status(self) -> Dict[str, bool]:
        """
        Get validation status by field.
        
        Returns:
            Dictionary mapping field names to validation status (True if valid)
        """
        return {
            field: not any(r.is_error for r in results)
            for field, results in self._field_results.items()
        }
    
    def __iter__(self):
        """Iterator for validation results"""
        return iter(self._results)
    
    def __len__(self) -> int:
        """Number of validation results"""
        return len(self._results)
    
    def __bool__(self) -> bool:
        """Boolean value (True if valid)"""
        return self.is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "is_valid": self.is_valid,
            "has_warnings": self.has_warnings,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "total_count": self.count,
            "results": [r.to_dict() for r in self._results],
            "field_status": self.get_field_status()
        }
    
    def format_summary(self) -> str:
        """
        Format a summary of validation results.
        
        Returns:
            Formatted summary string
        """
        lines = []
        
        # Overall status
        if self.is_valid:
            if self.has_warnings:
                lines.append("✓ VALID WITH WARNINGS")
            else:
                lines.append("✓ VALID")
        else:
            lines.append("✗ INVALID")
        
        # Counts
        lines.append(f"Total: {self.count}, Errors: {self.error_count}, Warnings: {self.warning_count}")
        
        # List errors and warnings
        if self.errors:
            lines.append("\nErrors:")
            for result in self.errors:
                field_info = f"[{result.field_name}] " if result.field_name else ""
                lines.append(f"  ✗ {field_info}{result.message}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for result in self.warnings:
                field_info = f"[{result.field_name}] " if result.field_name else ""
                lines.append(f"  ! {field_info}{result.message}")
        
        return "\n".join(lines)


class ValidationReport:
    """
    Comprehensive validation report with categorized results.
    
    Provides structured reporting of validation outcomes, including
    grouping by field, severity, and validator.
    """
    
    def __init__(self, results: ValidationResults, config_name: Optional[str] = None):
        """
        Initialize a validation report.
        
        Args:
            results: Validation results to report
            config_name: Optional name of the configuration being validated
        """
        self.results = results
        self.config_name = config_name
    
    def get_report(self, include_passed: bool = False) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Args:
            include_passed: Whether to include passed validations in the report
            
        Returns:
            Dictionary containing the validation report
        """
        report = {
            "summary": {
                "config_name": self.config_name,
                "is_valid": self.results.is_valid,
                "has_warnings": self.results.has_warnings,
                "error_count": self.results.error_count,
                "warning_count": self.results.warning_count,
                "total_count": self.results.count
            },
            "errors": [r.to_dict() for r in self.results.errors],
            "warnings": [r.to_dict() for r in self.results.warnings],
        }
        
        if include_passed:
            report["passed"] = [r.to_dict() for r in self.results.passed]
        
        # Group by field
        report["fields"] = {}
        for field in self.results.field_names:
            field_results = self.results.get_results_for_field(field)
            report["fields"][field] = {
                "is_valid": not any(r.is_error for r in field_results),
                "error_count": sum(1 for r in field_results if r.is_error),
                "warning_count": sum(1 for r in field_results if r.is_warning),
                "results": [r.to_dict() for r in field_results]
            }
        
        return report
    
    def format_report(self, verbose: bool = False) -> str:
        """
        Format a readable validation report.
        
        Args:
            verbose: Whether to include detailed information
            
        Returns:
            Formatted report string
        """
        lines = []
        
        # Header
        if self.config_name:
            lines.append(f"Validation Report for {self.config_name}")
        else:
            lines.append("Validation Report")
        lines.append("=" * 50)
        
        # Summary
        status = "VALID" if self.results.is_valid else "INVALID"
        if self.results.has_warnings and self.results.is_valid:
            status += " WITH WARNINGS"
        
        lines.append(f"Status: {status}")
        lines.append(f"Results: {self.results.count} total, "
                    f"{self.results.error_count} errors, "
                    f"{self.results.warning_count} warnings")
        
        # Errors
        if self.results.errors:
            lines.append("\nErrors:")
            for i, error in enumerate(self.results.errors, 1):
                field_info = f"[{error.field_name}] " if error.field_name else ""
                validator_info = f" ({error.validator_name})" if error.validator_name else ""
                lines.append(f"{i}. {field_info}{error.message}{validator_info}")
                
                # Add details in verbose mode
                if verbose and error.details:
                    for key, value in error.details.items():
                        lines.append(f"   - {key}: {value}")
        
        # Warnings
        if self.results.warnings:
            lines.append("\nWarnings:")
            for i, warning in enumerate(self.results.warnings, 1):
                field_info = f"[{warning.field_name}] " if warning.field_name else ""
                validator_info = f" ({warning.validator_name})" if warning.validator_name else ""
                lines.append(f"{i}. {field_info}{warning.message}{validator_info}")
                
                # Add details in verbose mode
                if verbose and warning.details:
                    for key, value in warning.details.items():
                        lines.append(f"   - {key}: {value}")
        
        # Field summary in verbose mode
        if verbose and self.results.field_names:
            lines.append("\nField Status:")
            for field in sorted(self.results.field_names):
                field_results = self.results.get_results_for_field(field)
                is_valid = not any(r.is_error for r in field_results)
                status_str = "✓ VALID" if is_valid else "✗ INVALID"
                lines.append(f"{field}: {status_str}")
        
        return "\n".join(lines)