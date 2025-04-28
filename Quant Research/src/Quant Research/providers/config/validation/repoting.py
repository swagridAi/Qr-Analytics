"""
Reporting module for configuration validation results.

This module provides utilities to format, display, and log validation results
in various formats including text, JSON, and log messages.
"""

import json
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, TextIO, Union
import sys
from dataclasses import asdict

from .result import ValidationResult, ValidationResults, Severity

logger = logging.getLogger(__name__)


class OutputFormat(str, Enum):
    """Output formats for validation reporting"""
    TEXT = "text"
    JSON = "json"
    SUMMARY = "summary"


class ValidationReporter:
    """
    Reporter for validation results with various output formats.
    
    This class handles the formatting and presentation of validation results
    in different formats suitable for various output contexts.
    """
    
    @staticmethod
    def format_result(result: ValidationResult, indent: int = 0) -> str:
        """
        Format a single validation result as a human-readable string.
        
        Args:
            result: The validation result to format
            indent: Indentation level (number of spaces)
            
        Returns:
            Formatted string representation of the result
        """
        indent_str = ' ' * indent
        
        # Use appropriate prefix based on severity
        prefix = {
            Severity.ERROR: "❌ ERROR:",
            Severity.WARNING: "⚠️ WARNING:",
            Severity.INFO: "ℹ️ INFO:"
        }.get(result.severity, "•")
        
        # Format basic message
        message = f"{indent_str}{prefix} {result.message}"
        
        # Add details if available
        if result.details:
            details_str = '\n'.join(
                f"{indent_str}  - {key}: {value}" 
                for key, value in result.details.items()
            )
            message = f"{message}\n{details_str}"
            
        return message
    
    @staticmethod
    def format_results(results: ValidationResults, 
                     include_details: bool = True, 
                     include_info: bool = False) -> str:
        """
        Format validation results as a human-readable text report.
        
        Args:
            results: Validation results to format
            include_details: Whether to include detailed information
            include_info: Whether to include info-level results
            
        Returns:
            Formatted text report of validation results
        """
        lines = ["Configuration Validation Results:"]
        
        # Group results by context if available
        contexts = {}
        ungrouped = []
        
        for result in results.all_results:
            # Skip info results if not requested
            if result.severity == Severity.INFO and not include_info:
                continue
                
            context = getattr(result, 'context', None)
            if context:
                if context not in contexts:
                    contexts[context] = []
                contexts[context].append(result)
            else:
                ungrouped.append(result)
        
        # Add grouped results
        for context, context_results in contexts.items():
            lines.append(f"\n=== {context.title()} ===")
            for result in context_results:
                lines.append(ValidationReporter.format_result(result, indent=2))
        
        # Add ungrouped results
        if ungrouped:
            lines.append("\n=== General ===")
            for result in ungrouped:
                lines.append(ValidationReporter.format_result(result, indent=2))
        
        # Add summary
        lines.append("\n=== Summary ===")
        status = "✅ VALID" if results.is_valid else "❌ INVALID"
        lines.append(f"{status}: Found {len(results.errors)} errors and {len(results.warnings)} warnings")
        
        return "\n".join(lines)
    
    @staticmethod
    def as_dict(results: ValidationResults) -> Dict[str, Any]:
        """
        Convert validation results to a serializable dictionary.
        
        Args:
            results: Validation results to convert
            
        Returns:
            Dictionary representation of validation results
        """
        # Convert individual results to dictionaries
        all_results = [asdict(r) for r in results.all_results]
        
        # Create the result dictionary
        return {
            "is_valid": results.is_valid,
            "error_count": len(results.errors),
            "warning_count": len(results.warnings),
            "info_count": len(results.info),
            "results": all_results
        }
    
    @staticmethod
    def to_json(results: ValidationResults, indent: int = 2) -> str:
        """
        Convert validation results to a JSON string.
        
        Args:
            results: Validation results to convert
            indent: JSON indentation level
            
        Returns:
            JSON string representation of validation results
        """
        return json.dumps(ValidationReporter.as_dict(results), indent=indent)
    
    @staticmethod
    def summary(results: ValidationResults) -> str:
        """
        Generate a short summary of validation results.
        
        Args:
            results: Validation results to summarize
            
        Returns:
            Summary string
        """
        status = "Valid" if results.is_valid else "Invalid"
        return (
            f"Validation Status: {status} "
            f"({len(results.errors)} errors, {len(results.warnings)} warnings, "
            f"{len(results.info)} info messages)"
        )
    
    @staticmethod
    def log_results(results: ValidationResults, logger: Optional[logging.Logger] = None) -> None:
        """
        Log validation results at appropriate levels.
        
        Args:
            results: Validation results to log
            logger: Logger to use (defaults to module logger)
        """
        log = logger or logging.getLogger(__name__)
        
        # Log a summary first
        if results.is_valid:
            log.info(f"Configuration valid with {len(results.warnings)} warnings")
        else:
            log.error(f"Configuration invalid with {len(results.errors)} errors")
        
        # Log individual results at their appropriate levels
        for result in results.all_results:
            message = ValidationReporter.format_result(result)
            
            if result.severity == Severity.ERROR:
                log.error(message)
            elif result.severity == Severity.WARNING:
                log.warning(message)
            else:
                log.info(message)
    
    @staticmethod
    def print_results(results: ValidationResults, 
                    output_format: OutputFormat = OutputFormat.TEXT,
                    file: TextIO = sys.stdout,
                    include_details: bool = True,
                    include_info: bool = False) -> None:
        """
        Print validation results to a file or stdout.
        
        Args:
            results: Validation results to print
            output_format: Format to use for output
            file: File to write to (defaults to stdout)
            include_details: Whether to include detailed information
            include_info: Whether to include info-level results
        """
        if output_format == OutputFormat.TEXT:
            print(ValidationReporter.format_results(
                results, include_details, include_info), file=file)
        elif output_format == OutputFormat.JSON:
            print(ValidationReporter.to_json(results), file=file)
        elif output_format == OutputFormat.SUMMARY:
            print(ValidationReporter.summary(results), file=file)
        else:
            raise ValueError(f"Unknown output format: {output_format}")


def format_validation_report(results: ValidationResults, 
                           format_type: str = "text",
                           include_details: bool = True,
                           include_info: bool = False) -> str:
    """
    Format validation results into a report.
    
    Args:
        results: Validation results to format
        format_type: Output format ('text', 'json', or 'summary')
        include_details: Whether to include detailed information
        include_info: Whether to include info-level results
        
    Returns:
        Formatted validation report
    """
    try:
        output_format = OutputFormat(format_type.lower())
    except ValueError:
        logger.warning(f"Unknown format '{format_type}', defaulting to 'text'")
        output_format = OutputFormat.TEXT
    
    if output_format == OutputFormat.TEXT:
        return ValidationReporter.format_results(
            results, include_details, include_info)
    elif output_format == OutputFormat.JSON:
        return ValidationReporter.to_json(results)
    elif output_format == OutputFormat.SUMMARY:
        return ValidationReporter.summary(results)
    
    # Should never reach here due to enum validation
    return ""


def get_color_for_severity(severity: Severity) -> str:
    """
    Get ANSI color code for a severity level.
    
    Args:
        severity: Severity level
        
    Returns:
        ANSI color code
    """
    if severity == Severity.ERROR:
        return "\033[91m"  # Red
    elif severity == Severity.WARNING:
        return "\033[93m"  # Yellow
    elif severity == Severity.INFO:
        return "\033[94m"  # Blue
    return "\033[0m"  # Reset


def colorize_output(message: str, severity: Severity) -> str:
    """
    Add color to a message based on severity.
    
    Args:
        message: Message to colorize
        severity: Severity level
        
    Returns:
        Colorized message
    """
    color = get_color_for_severity(severity)
    reset = "\033[0m"
    return f"{color}{message}{reset}"


def make_table(results: ValidationResults) -> List[List[str]]:
    """
    Format validation results as a table.
    
    Args:
        results: Validation results
        
    Returns:
        List of rows, where each row is a list of column values
    """
    # Header row
    table = [["Field", "Status", "Message"]]
    
    # Data rows
    for result in results.all_results:
        field = getattr(result, 'field', 'N/A')
        status = "✅" if result.is_valid else "❌"
        table.append([field, status, result.message])
    
    return table