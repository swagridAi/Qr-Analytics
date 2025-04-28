# src/quant_research/providers/config_validation.py
import logging
import inspect
from typing import Dict, Any, Type, List, Optional, Union, Set, Tuple
import asyncio
from dataclasses import dataclass

from ..core.config import ProviderConfig, ProviderType
from .base import BaseProvider


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a configuration validation check"""
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "error"  # "error", "warning", or "info"


class ConfigValidator:
    """Utility class for validating provider configurations"""
    
    @staticmethod
    def validate_config(config: ProviderConfig) -> List[ValidationResult]:
        """
        Validate a provider configuration and return validation results
        
        Args:
            config: Provider configuration to validate
            
        Returns:
            List of validation results
        """
        results = []
        
        # 1. Basic validation through Pydantic (this happens when the config is created)
        # If we got here, the basic validation passed
        
        # 2. Check if required fields are present and non-empty
        results.extend(ConfigValidator._validate_required_fields(config))
        
        # 3. Check environment variables
        results.extend(ConfigValidator._validate_environment_variables(config))
        
        # 4. Check for suspicious or problematic values
        results.extend(ConfigValidator._check_for_problems(config))
        
        return results
    
    @staticmethod
    def validate_config_for_provider(
        config: ProviderConfig,
        provider_class: Type[BaseProvider]
    ) -> List[ValidationResult]:
        """
        Validate a configuration for compatibility with a specific provider class
        
        Args:
            config: Provider configuration to validate
            provider_class: Provider class to validate against
            
        Returns:
            List of validation results
        """
        results = []
        
        # 1. Basic validation
        results.extend(ConfigValidator.validate_config(config))
        
        # 2. Check if config type matches provider's expected type
        results.extend(ConfigValidator._validate_config_type_match(config, provider_class))
        
        # 3. Check if provider requires fields that are missing in config
        results.extend(ConfigValidator._validate_provider_required_fields(config, provider_class))
        
        return results
    
    @staticmethod
    async def validate_connection(
        config: ProviderConfig,
        provider: Union[BaseProvider, Type[BaseProvider]]
    ) -> List[ValidationResult]:
        """
        Validate that a connection can be established with the provider
        
        Args:
            config: Provider configuration
            provider: Provider instance or class
            
        Returns:
            List of validation results
        """
        results = []
        
        # If given a class, instantiate it
        if inspect.isclass(provider):
            try:
                provider_instance = provider(config)
            except Exception as e:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Failed to instantiate provider: {e}",
                    details={"error": str(e), "provider_class": provider.__name__},
                    severity="error"
                ))
                return results
        else:
            provider_instance = provider
        
        # Try to connect
        try:
            await provider_instance.connect()
            
            # Check if connection was successful
            is_connected = await provider_instance.is_connected()
            
            if is_connected:
                results.append(ValidationResult(
                    is_valid=True,
                    message="Successfully connected to provider",
                    severity="info"
                ))
            else:
                results.append(ValidationResult(
                    is_valid=False,
                    message="Connection failed - provider reports it is not connected",
                    severity="error"
                ))
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Connection failed: {e}",
                details={"error": str(e)},
                severity="error"
            ))
        finally:
            # Try to disconnect (cleanup)
            try:
                await provider_instance.disconnect()
            except Exception as e:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Failed to disconnect: {e}",
                    details={"error": str(e)},
                    severity="warning"
                ))
        
        return results
    
    @staticmethod
    def _validate_required_fields(config: ProviderConfig) -> List[ValidationResult]:
        """Check if required fields are present and non-empty"""
        results = []
        
        # Check for empty fields that shouldn't be empty
        empty_fields = []
        
        if not config.name:
            empty_fields.append("name")
        
        if empty_fields:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Required fields cannot be empty: {', '.join(empty_fields)}",
                details={"empty_fields": empty_fields},
                severity="error"
            ))
        
        return results
    
    @staticmethod
    def _validate_environment_variables(config: ProviderConfig) -> List[ValidationResult]:
        """Check if required environment variables are set"""
        results = []
        
        if config.require_auth:
            missing_keys = []
            for key in config.api_keys:
                if not config.get_api_key(key):
                    missing_keys.append(f"{config.env_prefix}{key}")
            
            if missing_keys:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Missing required environment variables: {', '.join(missing_keys)}",
                    details={"missing_env_vars": missing_keys},
                    severity="error"
                ))
        
        return results
    
    @staticmethod
    def _check_for_problems(config: ProviderConfig) -> List[ValidationResult]:
        """Check for suspicious or problematic values"""
        results = []
        
        # Check connection timeout value
        if config.connection.timeout > 120:
            results.append(ValidationResult(
                is_valid=True,  # This is a warning, not an error
                message=f"Connection timeout is set to {config.connection.timeout}s, which is unusually high",
                details={"timeout": config.connection.timeout},
                severity="warning"
            ))
        
        # Check pool size
        if config.connection.pool_size > 20:
            results.append(ValidationResult(
                is_valid=True,  # This is a warning, not an error
                message=f"Connection pool size is set to {config.connection.pool_size}, which may consume excessive resources",
                details={"pool_size": config.connection.pool_size},
                severity="warning"
            ))
        
        # Rate limit checks
        if config.rate_limit and config.rate_limit.get("requests_per_second", 0) > 100:
            results.append(ValidationResult(
                is_valid=True,  # This is a warning, not an error
                message=f"Rate limit of {config.rate_limit.get('requests_per_second')} requests per second is unusually high",
                details={"rate_limit": config.rate_limit},
                severity="warning"
            ))
        
        return results
    
    @staticmethod
    def _validate_config_type_match(
        config: ProviderConfig,
        provider_class: Type[BaseProvider]
    ) -> List[ValidationResult]:
        """Check if the config type matches the provider's expected type"""
        results = []
        
        # Try to get the expected config type from the provider's type annotations
        expected_config_type = None
        
        # Check class __orig_bases__ for Generic parameters
        if hasattr(provider_class, '__orig_bases__'):
            for base in provider_class.__orig_bases__:
                if hasattr(base, '__origin__') and base.__origin__ is BaseProvider:
                    if hasattr(base, '__args__') and base.__args__:
                        expected_config_type = base.__args__[0]
                        break
        
        if expected_config_type and not isinstance(config, expected_config_type):
            results.append(ValidationResult(
                is_valid=False,
                message=f"Configuration type mismatch: Provider {provider_class.__name__} expects {expected_config_type.__name__}, got {config.__class__.__name__}",
                details={
                    "expected_type": expected_config_type.__name__,
                    "actual_type": config.__class__.__name__
                },
                severity="error"
            ))
        
        return results
    
    @staticmethod
    def _validate_provider_required_fields(
        config: ProviderConfig,
        provider_class: Type[BaseProvider]
    ) -> List[ValidationResult]:
        """Check if the provider requires fields that are missing in the config"""
        results = []
        
        # This requires introspection of the provider class, which might not be reliable
        # A better approach is for the provider to define its own validation method
        
        # For now, let's check based on provider name and typical requirements
        if "ccxt" in provider_class.__name__.lower():
            required_fields = ["exchange", "symbols"]
            for field in required_fields:
                if not hasattr(config, field) or not getattr(config, field):
                    results.append(ValidationResult(
                        is_valid=False,
                        message=f"Missing required field for {provider_class.__name__}: {field}",
                        details={"missing_field": field},
                        severity="error"
                    ))
        
        return results


class ConfigValidationSuite:
    """Suite of configuration validation tests"""
    
    @staticmethod
    async def run_validation_suite(
        config: ProviderConfig,
        provider_class: Type[BaseProvider],
        test_connection: bool = True
    ) -> Dict[str, List[ValidationResult]]:
        """
        Run a comprehensive validation suite on a provider configuration
        
        Args:
            config: Provider configuration to validate
            provider_class: Provider class to validate against
            test_connection: Whether to test the connection
            
        Returns:
            Dictionary of validation results by category
        """
        results = {}
        
        # Basic configuration validation
        results["basic_validation"] = ConfigValidator.validate_config(config)
        
        # Provider-specific validation
        results["provider_compatibility"] = ConfigValidator.validate_config_for_provider(
            config, provider_class
        )
        
        # Connection test (optional)
        if test_connection:
            results["connection_test"] = await ConfigValidator.validate_connection(
                config, provider_class
            )
        
        return results
    
    @staticmethod
    def format_validation_results(
        results: Dict[str, List[ValidationResult]],
        include_details: bool = False
    ) -> str:
        """
        Format validation results as a readable string
        
        Args:
            results: Validation results by category
            include_details: Whether to include detailed information
            
        Returns:
            Formatted validation results
        """
        lines = ["Configuration Validation Results:"]
        
        # Track overall validity
        is_valid = True
        error_count = 0
        warning_count = 0
        
        for category, category_results in results.items():
            # Skip empty categories
            if not category_results:
                continue
            
            lines.append(f"\n=== {category.replace('_', ' ').title()} ===")
            
            for result in category_results:
                # Add to counts
                if result.severity == "error":
                    error_count += 1
                    is_valid = False
                elif result.severity == "warning":
                    warning_count += 1
                
                # Format the message with appropriate prefix
                prefix = {
                    "error": "❌ ERROR:",
                    "warning": "⚠️ WARNING:",
                    "info": "ℹ️ INFO:"
                }.get(result.severity, "•")
                
                lines.append(f"{prefix} {result.message}")
                
                # Add details if requested
                if include_details and result.details:
                    for key, value in result.details.items():
                        lines.append(f"  - {key}: {value}")
        
        # Add summary
        lines.append("\n=== Summary ===")
        status = "✅ VALID" if is_valid else "❌ INVALID"
        lines.append(f"{status}: Found {error_count} errors and {warning_count} warnings")
        
        return "\n".join(lines)
    
    @staticmethod
    async def validate_and_report(
        config: ProviderConfig,
        provider_class: Type[BaseProvider],
        test_connection: bool = True,
        include_details: bool = False
    ) -> Tuple[bool, str]:
        """
        Validate a configuration and generate a report
        
        Args:
            config: Provider configuration to validate
            provider_class: Provider class to validate against
            test_connection: Whether to test the connection
            include_details: Whether to include detailed information
            
        Returns:
            Tuple of (is_valid, report)
        """
        results = await ConfigValidationSuite.run_validation_suite(
            config, provider_class, test_connection
        )
        
        # Check if valid (no errors)
        is_valid = not any(
            result.severity == "error" and not result.is_valid
            for category_results in results.values()
            for result in category_results
        )
        
        report = ConfigValidationSuite.format_validation_results(
            results, include_details
        )
        
        return is_valid, report