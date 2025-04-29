"""
Configuration Templates and Generators

This module provides utilities for template-based configuration generation,
allowing for quick creation of configurations based on patterns and recipes.

Features:
- Template library for common pipeline configurations
- Variable substitution in templates
- Recipe-based configuration assembly
- Parameterized generators for different pipeline types
- Configuration discovery and reuse
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Callable, TypeVar, Generic, Type
import json
import yaml
import copy
import importlib.resources
from enum import Enum

from pydantic import BaseModel

# Import the layered config system
from .config import (
    BaseConfig, PipelineConfig, ProviderConfig, AnalyticsConfig, 
    BacktestConfig, CoreConfig, OutputConfig, ConfigurationManager,
    ConfigProfile, AnalyticsModuleConfig
)

# Type definitions
T = TypeVar('T', bound=BaseConfig)

#-----------------------------------------------------------------------
# Template System
#-----------------------------------------------------------------------

class TemplateVariable:
    """Represents a variable in a configuration template."""
    
    def __init__(self, 
                name: str, 
                description: str, 
                default: Any = None,
                required: bool = False,
                options: Optional[List[Any]] = None,
                validator: Optional[Callable[[Any], bool]] = None):
        """
        Initialize template variable.
        
        Args:
            name: Variable name
            description: Description of the variable
            default: Default value if not provided
            required: Whether the variable is required
            options: List of valid options for this variable
            validator: Optional function to validate the value
        """
        self.name = name
        self.description = description
        self.default = default
        self.required = required
        self.options = options
        self.validator = validator
    
    def validate(self, value: Any) -> bool:
        """
        Validate a value for this variable.
        
        Args:
            value: Value to validate
            
        Returns:
            Whether the value is valid
        """
        # If required, check if value is provided
        if self.required and value is None:
            return False
            
        # If options are specified, check if value is in options
        if self.options is not None and value is not None:
            if value not in self.options:
                return False
                
        # If validator is specified, use it
        if self.validator is not None and value is not None:
            if not self.validator(value):
                return False
                
        return True
    
    def __repr__(self) -> str:
        """String representation of the variable."""
        return f"TemplateVariable({self.name}, default={self.default}, required={self.required})"

class ConfigTemplate:
    """
    Configuration template with variable substitution.
    
    Templates allow for parameterized configuration generation with
    variable substitution, validation, and sensible defaults.
    """
    
    def __init__(self, 
                name: str, 
                template: Dict[str, Any],
                variables: Optional[Dict[str, TemplateVariable]] = None,
                description: str = ""):
        """
        Initialize configuration template.
        
        Args:
            name: Template name
            template: Template dictionary with variables
            variables: Dictionary of template variables
            description: Template description
        """
        self.name = name
        self.template = template
        self.variables = variables or {}
        self.description = description
    
    def get_required_variables(self) -> List[str]:
        """Get names of required variables."""
        return [name for name, var in self.variables.items() if var.required]
    
    def get_variable_defaults(self) -> Dict[str, Any]:
        """Get default values for all variables."""
        return {name: var.default for name, var in self.variables.items()}
    
    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """
        Validate variable values.
        
        Args:
            variables: Dictionary of variable values
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check required variables
        for name, var in self.variables.items():
            if var.required and name not in variables:
                errors.append(f"Required variable '{name}' is missing")
        
        # Validate provided variables
        for name, value in variables.items():
            if name in self.variables:
                var = self.variables[name]
                if not var.validate(value):
                    if var.options:
                        errors.append(f"Invalid value for '{name}'. Must be one of: {var.options}")
                    else:
                        errors.append(f"Invalid value for '{name}'")
            else:
                errors.append(f"Unknown variable '{name}'")
        
        return errors
    
    def substitute_variables(self, 
                           template_dict: Dict[str, Any], 
                           variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute variables in a template dictionary.
        
        Args:
            template_dict: Template dictionary
            variables: Variable values
            
        Returns:
            Dictionary with variables substituted
        """
        # Create a deep copy to avoid modifying the original
        result = copy.deepcopy(template_dict)
        
        def substitute_in_value(value, variables):
            """Recursive substitution in nested structures."""
            if isinstance(value, str):
                # Substitute string variables ${var_name}
                for var_name, var_value in variables.items():
                    placeholder = f"${{{var_name}}}"
                    if placeholder in value:
                        # If the entire string is just the placeholder, replace with the actual value
                        if value == placeholder:
                            return var_value
                        # Otherwise, replace the placeholder in the string
                        value = value.replace(placeholder, str(var_value))
                return value
            elif isinstance(value, dict):
                # Recursively substitute in dictionary
                return {k: substitute_in_value(v, variables) for k, v in value.items()}
            elif isinstance(value, list):
                # Recursively substitute in list
                return [substitute_in_value(item, variables) for item in value]
            else:
                # Return other values unchanged
                return value
        
        # Perform substitution
        result = substitute_in_value(result, variables)
        
        return result
    
    def instantiate(self, 
                   variables: Dict[str, Any], 
                   config_class: Type[T] = None) -> Union[Dict[str, Any], T]:
        """
        Create a configuration instance from the template.
        
        Args:
            variables: Variable values
            config_class: Optional config class to instantiate
            
        Returns:
            Configuration instance or dictionary
            
        Raises:
            ValueError: If variable validation fails
        """
        # Validate variables
        errors = self.validate_variables(variables)
        if errors:
            raise ValueError(f"Template variable validation failed: {'; '.join(errors)}")
        
        # Fill in defaults for missing variables
        all_variables = self.get_variable_defaults()
        all_variables.update(variables)
        
        # Substitute variables in template
        config_dict = self.substitute_variables(self.template, all_variables)
        
        # Create instance if requested
        if config_class:
            return config_class(**config_dict)
        
        return config_dict

class TemplateLibrary:
    """
    Library of configuration templates.
    
    Provides storage, lookup, and management for configuration templates.
    """
    
    def __init__(self):
        """Initialize template library."""
        self._templates: Dict[str, ConfigTemplate] = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """Load built-in templates from package resources."""
        try:
            # Try to load templates from package resources
            import importlib.resources as pkg_resources
            from . import templates
            
            for template_file in pkg_resources.contents(templates):
                if template_file.endswith('.yaml') or template_file.endswith('.yml'):
                    template_text = pkg_resources.read_text(templates, template_file)
                    template_data = yaml.safe_load(template_text)
                    
                    # Register template
                    name = template_file.rsplit('.', 1)[0]
                    self.register_template_dict(name, template_data)
        except (ImportError, ModuleNotFoundError):
            # Fall back to hardcoded templates if package resources not available
            self._register_hardcoded_templates()
    
    def _register_hardcoded_templates(self):
        """Register hardcoded templates."""
        # Example templates
        crypto_template = {
            "name": "${pipeline_name}",
            "provider": {
                "id": "crypto_ccxt",
                "settings": {
                    "exchange": "${exchange}",
                    "timeframe": "${timeframe}"
                },
                "symbols": "${symbols}"
            },
            "analytics": {
                "modules": [
                    {
                        "name": "volatility",
                        "params": {
                            "estimator": "yang_zhang",
                            "window": 20
                        }
                    },
                    {
                        "name": "regime_detector",
                        "params": {
                            "method": "hmm",
                            "n_states": 3
                        }
                    }
                ]
            },
            "backtest": {
                "strategy": "${strategy}",
                "risk": {
                    "stop_loss_pct": 5.0,
                    "max_position_size": 0.1
                }
            }
        }
        
        # Register template with variables
        self.register_template(
            "crypto_backtest",
            crypto_template,
            {
                "pipeline_name": TemplateVariable(
                    "pipeline_name", 
                    "Name of the pipeline", 
                    default="crypto_strategy",
                    required=True
                ),
                "exchange": TemplateVariable(
                    "exchange", 
                    "Cryptocurrency exchange",
                    default="binance",
                    options=["binance", "coinbase", "kraken", "ftx"]
                ),
                "timeframe": TemplateVariable(
                    "timeframe",
                    "Candle timeframe",
                    default="1h",
                    options=["1m", "5m", "15m", "1h", "4h", "1d"]
                ),
                "symbols": TemplateVariable(
                    "symbols",
                    "Trading pairs to analyze",
                    default=["BTC/USDT", "ETH/USDT"],
                    required=True
                ),
                "strategy": TemplateVariable(
                    "strategy",
                    "Trading strategy",
                    default="momentum",
                    options=["momentum", "mean_reversion", "regime_adaptive"]
                )
            },
            "Cryptocurrency backtest pipeline template"
        )
        
        # Add more hardcoded templates here
    
    def register_template(self, 
                         name: str, 
                         template: Dict[str, Any],
                         variables: Dict[str, TemplateVariable],
                         description: str = "") -> None:
        """
        Register a template with variables.
        
        Args:
            name: Template name
            template: Template dictionary
            variables: Dictionary of template variables
            description: Template description
        """
        self._templates[name] = ConfigTemplate(
            name=name,
            template=template,
            variables=variables,
            description=description
        )
    
    def register_template_dict(self, 
                             name: str, 
                             template_dict: Dict[str, Any]) -> None:
        """
        Register a template from a dictionary containing template and variables.
        
        Args:
            name: Template name
            template_dict: Dictionary with 'template', 'variables', and 'description'
        """
        # Extract template and variables
        template = template_dict.get('template', {})
        variables_dict = template_dict.get('variables', {})
        description = template_dict.get('description', '')
        
        # Convert variable dictionaries to TemplateVariable objects
        variables = {}
        for var_name, var_info in variables_dict.items():
            variables[var_name] = TemplateVariable(
                name=var_name,
                description=var_info.get('description', ''),
                default=var_info.get('default'),
                required=var_info.get('required', False),
                options=var_info.get('options')
            )
        
        # Register template
        self.register_template(name, template, variables, description)
    
    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """Get a template by name."""
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available templates."""
        return list(self._templates.keys())
    
    def get_template_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about templates.
        
        Args:
            name: Optional template name for specific info
            
        Returns:
            Dictionary with template information
        """
        if name:
            template = self.get_template(name)
            if not template:
                return {}
                
            return {
                "name": template.name,
                "description": template.description,
                "variables": {
                    var_name: {
                        "description": var.description,
                        "default": var.default,
                        "required": var.required,
                        "options": var.options
                    }
                    for var_name, var in template.variables.items()
                }
            }
        else:
            # Return info for all templates
            return {
                name: {
                    "description": template.description,
                    "required_variables": template.get_required_variables()
                }
                for name, template in self._templates.items()
            }
    
    def instantiate_template(self, 
                            name: str, 
                            variables: Dict[str, Any],
                            config_class: Optional[Type[T]] = None) -> Union[Dict[str, Any], T]:
        """
        Create a configuration from a template.
        
        Args:
            name: Template name
            variables: Variable values
            config_class: Optional config class to instantiate
            
        Returns:
            Configuration instance or dictionary
            
        Raises:
            ValueError: If template not found or validation fails
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template not found: {name}")
            
        return template.instantiate(variables, config_class)

#-----------------------------------------------------------------------
# Recipe System - Layered Configuration Assembly
#-----------------------------------------------------------------------

class ConfigRecipe:
    """
    Recipe for assembling a configuration from component templates.
    
    Recipes allow combining multiple component templates into a
    complete pipeline configuration with component-level substitutions.
    """
    
    def __init__(self, 
                name: str,
                components: Dict[str, str],
                variables: Dict[str, Any] = None,
                description: str = ""):
        """
        Initialize configuration recipe.
        
        Args:
            name: Recipe name
            components: Dictionary mapping component names to template names
            variables: Dictionary of variable values for templates
            description: Recipe description
        """
        self.name = name
        self.components = components
        self.variables = variables or {}
        self.description = description
    
    def instantiate(self, 
                   template_library: TemplateLibrary,
                   additional_variables: Optional[Dict[str, Any]] = None) -> PipelineConfig:
        """
        Create a pipeline configuration from the recipe.
        
        Args:
            template_library: Template library for component templates
            additional_variables: Additional variable values
            
        Returns:
            Complete pipeline configuration
            
        Raises:
            ValueError: If a component template is not found
        """
        # Combine variables
        variables = self.variables.copy()
        if additional_variables:
            variables.update(additional_variables)
            
        # Create configuration manager
        manager = ConfigurationManager()
        
        # Create component configurations
        components = {}
        for component_name, template_name in self.components.items():
            try:
                # Get component variables
                component_vars = {
                    k.split('.', 1)[1]: v 
                    for k, v in variables.items() 
                    if k.startswith(f"{component_name}.")
                }
                
                # Add global variables
                component_vars.update({
                    k: v for k, v in variables.items() 
                    if '.' not in k
                })
                
                # Instantiate component template
                component = template_library.instantiate_template(
                    template_name, component_vars
                )
                
                components[component_name] = component
            except ValueError as e:
                raise ValueError(f"Error creating component '{component_name}': {e}")
                
        # Create pipeline configuration
        pipeline_name = variables.get('name', self.name)
        
        # Check for required components
        if 'provider' not in components:
            raise ValueError("Provider component is required")
            
        # Create pipeline configuration
        pipeline_args = {
            "name": pipeline_name,
            "provider": components['provider']
        }
        
        # Add optional components
        for component_name in ['core', 'analytics', 'backtest', 'output']:
            if component_name in components:
                pipeline_args[component_name] = components[component_name]
                
        # Create pipeline config
        pipeline_config = PipelineConfig(**pipeline_args)
        
        # Set up directories
        pipeline_config.setup()
        
        return pipeline_config

class RecipeLibrary:
    """
    Library of configuration recipes.
    
    Provides storage, lookup, and management for configuration recipes.
    """
    
    def __init__(self, template_library: Optional[TemplateLibrary] = None):
        """
        Initialize recipe library.
        
        Args:
            template_library: Optional template library to use
        """
        self._recipes: Dict[str, ConfigRecipe] = {}
        self._template_library = template_library or TemplateLibrary()
        self._load_builtin_recipes()
    
    def _load_builtin_recipes(self):
        """Load built-in recipes."""
        # Register hardcoded recipes
        self.register_recipe(
            name="crypto_momentum",
            components={
                "provider": "crypto_provider",
                "analytics": "momentum_analytics",
                "backtest": "basic_backtest",
                "output": "standard_output"
            },
            variables={
                "name": "crypto_momentum_strategy",
                "provider.exchange": "binance",
                "provider.symbols": ["BTC/USDT", "ETH/USDT"],
                "backtest.strategy": "momentum"
            },
            description="Cryptocurrency momentum strategy"
        )
        
        # Add more built-in recipes here
    
    def register_recipe(self, 
                       name: str,
                       components: Dict[str, str],
                       variables: Dict[str, Any] = None,
                       description: str = "") -> None:
        """
        Register a configuration recipe.
        
        Args:
            name: Recipe name
            components: Dictionary mapping component names to template names
            variables: Dictionary of variable values for templates
            description: Recipe description
        """
        self._recipes[name] = ConfigRecipe(
            name=name,
            components=components,
            variables=variables or {},
            description=description
        )
    
    def get_recipe(self, name: str) -> Optional[ConfigRecipe]:
        """Get a recipe by name."""
        return self._recipes.get(name)
    
    def list_recipes(self) -> List[str]:
        """List all available recipes."""
        return list(self._recipes.keys())
    
    def get_recipe_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about recipes.
        
        Args:
            name: Optional recipe name for specific info
            
        Returns:
            Dictionary with recipe information
        """
        if name:
            recipe = self.get_recipe(name)
            if not recipe:
                return {}
                
            return {
                "name": recipe.name,
                "description": recipe.description,
                "components": recipe.components,
                "variables": recipe.variables
            }
        else:
            # Return info for all recipes
            return {
                name: {
                    "description": recipe.description,
                    "components": recipe.components
                }
                for name, recipe in self._recipes.items()
            }
    
    def instantiate_recipe(self, 
                          name: str, 
                          variables: Optional[Dict[str, Any]] = None) -> PipelineConfig:
        """
        Create a pipeline configuration from a recipe.
        
        Args:
            name: Recipe name
            variables: Additional variable values
            
        Returns:
            Pipeline configuration
            
        Raises:
            ValueError: If recipe not found or instantiation fails
        """
        recipe = self.get_recipe(name)
        if not recipe:
            raise ValueError(f"Recipe not found: {name}")
            
        return recipe.instantiate(self._template_library, variables)

#-----------------------------------------------------------------------
# Configuration Generators - Parameterized Configuration Creation
#-----------------------------------------------------------------------

class ConfigGenerator:
    """
    Generator for creating configurations from parameters.
    
    Generators are higher-level abstractions that create complete
    configurations based on business-focused parameters rather than
    technical configuration details.
    """
    
    def __init__(self, 
                name: str,
                template_library: Optional[TemplateLibrary] = None,
                recipe_library: Optional[RecipeLibrary] = None):
        """
        Initialize configuration generator.
        
        Args:
            name: Generator name
            template_library: Optional template library to use
            recipe_library: Optional recipe library to use
        """
        self.name = name
        self._template_library = template_library or TemplateLibrary()
        self._recipe_library = recipe_library or RecipeLibrary(self._template_library)
    
    def generate_crypto_config(self, 
                             strategy_type: str,
                             exchange: str,
                             symbols: List[str],
                             timeframe: str = "1h",
                             risk_profile: str = "medium") -> PipelineConfig:
        """
        Generate a cryptocurrency trading pipeline configuration.
        
        Args:
            strategy_type: Type of strategy (momentum, mean_reversion, etc.)
            exchange: Exchange to trade on
            symbols: Trading pairs to analyze
            timeframe: Candle timeframe
            risk_profile: Risk profile (low, medium, high)
            
        Returns:
            Complete pipeline configuration
        """
        # Map risk profile to risk parameters
        risk_profiles = {
            "low": {
                "stop_loss_pct": 2.0,
                "max_position_size": 0.05,
                "target_volatility": 0.10
            },
            "medium": {
                "stop_loss_pct": 5.0,
                "max_position_size": 0.10,
                "target_volatility": 0.15
            },
            "high": {
                "stop_loss_pct": 10.0,
                "max_position_size": 0.20,
                "target_volatility": 0.25
            }
        }
        
        risk_params = risk_profiles.get(risk_profile.lower(), risk_profiles["medium"])
        
        # Map strategy type to specific modules
        strategy_modules = {
            "momentum": [
                {"name": "momentum", "params": {"lookback": 20, "smoothing": 2}}
            ],
            "mean_reversion": [
                {"name": "mean_reversion", "params": {"lookback": 20, "zscore_entry": 2.0, "zscore_exit": 0.5}}
            ],
            "regime_adaptive": [
                {"name": "regime_detector", "params": {"method": "hmm", "n_states": 3}},
                {"name": "volatility", "params": {"estimator": "garch", "p": 1, "q": 1}}
            ]
        }
        
        modules = strategy_modules.get(strategy_type.lower(), [])
        
        # Generate configuration
        try:
            # Try to use a recipe if available
            recipe_name = f"crypto_{strategy_type.lower()}"
            if recipe_name in self._recipe_library.list_recipes():
                return self._recipe_library.instantiate_recipe(
                    recipe_name,
                    {
                        "name": f"{exchange}_{strategy_type}_{timeframe}",
                        "provider.exchange": exchange,
                        "provider.symbols": symbols,
                        "provider.timeframe": timeframe,
                        "backtest.risk.stop_loss_pct": risk_params["stop_loss_pct"],
                        "backtest.risk.max_position_size": risk_params["max_position_size"],
                        "backtest.risk.target_volatility": risk_params["target_volatility"]
                    }
                )
        except ValueError:
            # Fall back to manual configuration creation
            pass
            
        # Create configuration using template if recipe not available
        try:
            template_name = "crypto_backtest"
            if template_name in self._template_library.list_templates():
                return self._template_library.instantiate_template(
                    template_name,
                    {
                        "pipeline_name": f"{exchange}_{strategy_type}_{timeframe}",
                        "exchange": exchange,
                        "timeframe": timeframe,
                        "symbols": symbols,
                        "strategy": strategy_type
                    },
                    PipelineConfig
                )
        except ValueError:
            # Fall back to manual configuration creation
            pass
        
        # Manual configuration creation as final fallback
        provider = ProviderConfig(
            id=f"crypto_ccxt",
            symbols=symbols,
            settings={
                "exchange": exchange,
                "timeframe": timeframe
            }
        )
        
        analytics = AnalyticsConfig()
        for module in modules:
            analytics.add_module(module["name"], **module["params"])
        
        # Add volatility module if not present
        if not any(m.name == "volatility" for m in analytics.modules):
            analytics.add_module("volatility", estimator="yang_zhang", window=20)
        
        backtest = BacktestConfig(
            strategy=strategy_type,
            risk=risk_params
        )
        
        # Create configuration manager
        manager = ConfigurationManager(app_name="crypto")
        
        # Create pipeline config
        return manager.create_pipeline_config(
            name=f"{exchange}_{strategy_type}_{timeframe}",
            provider_config=provider,
            analytics_config=analytics,
            backtest_config=backtest
        )
    
    def generate_equities_config(self,
                               strategy_type: str,
                               symbols: List[str],
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               benchmark: Optional[str] = "SPY") -> PipelineConfig:
        """
        Generate an equities trading pipeline configuration.
        
        Args:
            strategy_type: Type of strategy (momentum, mean_reversion, etc.)
            symbols: Stock symbols to analyze
            start_date: Optional start date for backtest
            end_date: Optional end date for backtest
            benchmark: Benchmark symbol
            
        Returns:
            Complete pipeline configuration
        """
        # Generate configuration
        provider = ProviderConfig(
            id=f"equities_yahoo",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            settings={
                "interval": "1d",
                "adjust_prices": True
            }
        )
        
        analytics = AnalyticsConfig()
        
        # Add strategy-specific modules
        if strategy_type.lower() == "momentum":
            analytics.add_module("momentum", lookback=20, smoothing=2)
        elif strategy_type.lower() == "mean_reversion":
            analytics.add_module("mean_reversion", lookback=20, zscore_entry=2.0, zscore_exit=0.5)
        elif strategy_type.lower() == "regime_adaptive":
            analytics.add_module("regime_detector", method="hmm", n_states=3)
            analytics.add_module("volatility", estimator="garch", p=1, q=1)
        
        # Add default volatility module if not already present
        if not any(m.name == "volatility" for m in analytics.modules):
            analytics.add_module("volatility", estimator="yang_zhang", window=20)
        
        backtest = BacktestConfig(
            strategy=strategy_type,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            risk={"stop_loss_pct": 3.0, "max_position_size": 0.05}
        )
        
        # Create configuration manager
        manager = ConfigurationManager(app_name="equities")
        
        # Create pipeline config
        return manager.create_pipeline_config(
            name=f"equities_{strategy_type}",
            provider_config=provider,
            analytics_config=analytics,
            backtest_config=backtest
        )

#-----------------------------------------------------------------------
# Configuration Directory Management
#-----------------------------------------------------------------------

class ConfigurationDirectory:
    """
    Manage a directory of configuration files.
    
    Provides functionality for discovery, versioning, and organization
    of configuration files in a directory structure.
    """
    
    def __init__(self, directory: Union[str, Path]):
        """
        Initialize configuration directory.
        
        Args:
            directory: Path to configuration directory
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
    
    def list_configurations(self, pattern: Optional[str] = None) -> List[str]:
        """
        List configuration files in the directory.
        
        Args:
            pattern: Optional glob pattern to filter files
            
        Returns:
            List of configuration file paths
        """
        if pattern:
            return [str(p) for p in self.directory.glob(pattern)]
        else:
            return [str(p) for p in self.directory.glob("*.yaml") if p.is_file()]
    
    def get_latest_version(self, base_name: str) -> Optional[str]:
        """
        Get the latest version of a configuration.
        
        Args:
            base_name: Base name of the configuration
            
        Returns:
            Path to the latest version or None if not found
        """
        # Find all versions of the configuration
        versions = list(self.directory.glob(f"{base_name}*.yaml"))
        
        if not versions:
            return None
            
        # Sort by modification time
        versions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return str(versions[0])
    
    def load_configuration(self, 
                          filename: str, 
                          config_class: Type[T] = PipelineConfig) -> T:
        """
        Load a configuration from a file.
        
        Args:
            filename: Configuration filename
            config_class: Configuration class to instantiate
            
        Returns:
            Configuration instance
            
        Raises:
            ValueError: If file not found or invalid
        """
        path = self.directory / filename
        
        if not path.exists():
            path = self.directory / f"{filename}.yaml"
            
        if not path.exists():
            raise ValueError(f"Configuration file not found: {filename}")
            
        manager = ConfigurationManager()
        return manager.load_from_path(config_class, path)
    
    def save_configuration(self, 
                          config: BaseConfig, 
                          filename: str,
                          overwrite: bool = False,
                          versioned: bool = True) -> str:
        """
        Save a configuration to a file.
        
        Args:
            config: Configuration to save
            filename: Filename to save as
            overwrite: Whether to overwrite existing file
            versioned: Whether to create a versioned copy
            
        Returns:
            Path to the saved file
        """
        # Normalize filename
        if not filename.endswith('.yaml'):
            filename = f"{filename}.yaml"
            
        path = self.directory / filename
        
        # Check if file exists
        if path.exists() and not overwrite:
            if versioned:
                # Create versioned filename
                base, ext = path.stem, path.suffix
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{base}_v{timestamp}{ext}"
                path = self.directory / filename
            else:
                raise ValueError(f"Configuration file already exists: {filename}")
        
        # Save configuration
        manager = ConfigurationManager()
        manager.save_config(config, path)
        
        return str(path)
    
    def create_from_template(self, 
                           template_name: str,
                           variables: Dict[str, Any],
                           output_filename: Optional[str] = None) -> str:
        """
        Create a configuration from a template.
        
        Args:
            template_name: Template name
            variables: Template variables
            output_filename: Optional output filename
            
        Returns:
            Path to the saved configuration
            
        Raises:
            ValueError: If template not found or creation fails
        """
        # Create template library
        template_library = TemplateLibrary()
        
        # Create configuration from template
        config = template_library.instantiate_template(
            template_name, variables, PipelineConfig
        )
        
        # Generate filename if not provided
        if not output_filename:
            output_filename = f"{variables.get('pipeline_name', template_name)}.yaml"
            
        # Save configuration
        return self.save_configuration(config, output_filename)
    
    def create_from_recipe(self, 
                         recipe_name: str,
                         variables: Dict[str, Any],
                         output_filename: Optional[str] = None) -> str:
        """
        Create a configuration from a recipe.
        
        Args:
            recipe_name: Recipe name
            variables: Recipe variables
            output_filename: Optional output filename
            
        Returns:
            Path to the saved configuration
            
        Raises:
            ValueError: If recipe not found or creation fails
        """
        # Create recipe library
        recipe_library = RecipeLibrary()
        
        # Create configuration from recipe
        config = recipe_library.instantiate_recipe(recipe_name, variables)
        
        # Generate filename if not provided
        if not output_filename:
            output_filename = f"{variables.get('name', recipe_name)}.yaml"
            
        # Save configuration
        return self.save_configuration(config, output_filename)

#-----------------------------------------------------------------------
# Configuration Wizard - Interactive Configuration Creation
#-----------------------------------------------------------------------

class ConfigWizard:
    """
    Wizard for interactive configuration creation.
    
    Guides users through the process of creating a configuration by
    asking questions and generating the configuration based on answers.
    """
    
    def __init__(self, 
                template_library: Optional[TemplateLibrary] = None,
                recipe_library: Optional[RecipeLibrary] = None):
        """
        Initialize configuration wizard.
        
        Args:
            template_library: Optional template library to use
            recipe_library: Optional recipe library to use
        """
        self._template_library = template_library or TemplateLibrary()
        self._recipe_library = recipe_library or RecipeLibrary(self._template_library)
        self._generator = ConfigGenerator(
            "wizard", self._template_library, self._recipe_library
        )
    
    def create_configuration(self, wizard_type: str, **kwargs) -> PipelineConfig:
        """
        Create a configuration using the wizard.
        
        Args:
            wizard_type: Type of wizard to use
            **kwargs: Wizard parameters
            
        Returns:
            Generated configuration
            
        Raises:
            ValueError: If wizard type not supported
        """
        if wizard_type.lower() == "crypto":
            return self._generator.generate_crypto_config(
                strategy_type=kwargs.get('strategy_type', 'momentum'),
                exchange=kwargs.get('exchange', 'binance'),
                symbols=kwargs.get('symbols', ['BTC/USDT']),
                timeframe=kwargs.get('timeframe', '1h'),
                risk_profile=kwargs.get('risk_profile', 'medium')
            )
        elif wizard_type.lower() == "equities":
            return self._generator.generate_equities_config(
                strategy_type=kwargs.get('strategy_type', 'momentum'),
                symbols=kwargs.get('symbols', ['AAPL', 'MSFT', 'AMZN']),
                start_date=kwargs.get('start_date'),
                end_date=kwargs.get('end_date'),
                benchmark=kwargs.get('benchmark', 'SPY')
            )
        else:
            raise ValueError(f"Unsupported wizard type: {wizard_type}")
    
    def run_interactive_wizard(self, wizard_type: str) -> PipelineConfig:
        """
        Run an interactive wizard to create a configuration.
        
        Note: This is a simple command-line wizard. For a real application,
        you would want to implement a more sophisticated UI.
        
        Args:
            wizard_type: Type of wizard to run
            
        Returns:
            Generated configuration
            
        Raises:
            ValueError: If wizard type not supported
        """
        print(f"Configuration Wizard - {wizard_type}")
        print("=" * 50)
        
        answers = {}
        
        if wizard_type.lower() == "crypto":
            # Ask crypto-specific questions
            answers['strategy_type'] = input("Strategy type (momentum, mean_reversion, regime_adaptive): ") or "momentum"
            answers['exchange'] = input("Exchange (binance, coinbase, kraken): ") or "binance"
            
            symbols_input = input("Trading pairs (comma-separated, e.g. BTC/USDT,ETH/USDT): ") or "BTC/USDT"
            answers['symbols'] = [s.strip() for s in symbols_input.split(",")]
            
            answers['timeframe'] = input("Timeframe (1m, 5m, 15m, 1h, 4h, 1d): ") or "1h"
            answers['risk_profile'] = input("Risk profile (low, medium, high): ") or "medium"
            
            return self.create_configuration("crypto", **answers)
            
        elif wizard_type.lower() == "equities":
            # Ask equities-specific questions
            answers['strategy_type'] = input("Strategy type (momentum, mean_reversion, regime_adaptive): ") or "momentum"
            
            symbols_input = input("Stock symbols (comma-separated, e.g. AAPL,MSFT,AMZN): ") or "AAPL,MSFT,AMZN"
            answers['symbols'] = [s.strip() for s in symbols_input.split(",")]
            
            answers['start_date'] = input("Start date (YYYY-MM-DD, optional): ")
            answers['end_date'] = input("End date (YYYY-MM-DD, optional): ")
            answers['benchmark'] = input("Benchmark symbol (default: SPY): ") or "SPY"
            
            return self.create_configuration("equities", **answers)
            
        else:
            raise ValueError(f"Unsupported wizard type: {wizard_type}")
    
    def print_template_info(self, template_name: Optional[str] = None) -> None:
        """
        Print information about available templates.
        
        Args:
            template_name: Optional template name for specific info
        """
        if template_name:
            # Print info for specific template
            template_info = self._template_library.get_template_info(template_name)
            if not template_info:
                print(f"Template not found: {template_name}")
                return
                
            print(f"Template: {template_name}")
            print(f"Description: {template_info['description']}")
            print("\nVariables:")
            for var_name, var_info in template_info['variables'].items():
                required = " (required)" if var_info['required'] else ""
                options = f" Options: {var_info['options']}" if var_info['options'] else ""
                default = f" Default: {var_info['default']}" if var_info['default'] is not None else ""
                print(f"  {var_name}{required}: {var_info['description']}{default}{options}")
        else:
            # Print list of available templates
            templates = self._template_library.list_templates()
            print(f"Available Templates ({len(templates)}):")
            for name in templates:
                info = self._template_library.get_template_info(name)
                print(f"  {name}: {info['description']}")
    
    def print_recipe_info(self, recipe_name: Optional[str] = None) -> None:
        """
        Print information about available recipes.
        
        Args:
            recipe_name: Optional recipe name for specific info
        """
        if recipe_name:
            # Print info for specific recipe
            recipe_info = self._recipe_library.get_recipe_info(recipe_name)
            if not recipe_info:
                print(f"Recipe not found: {recipe_name}")
                return
                
            print(f"Recipe: {recipe_name}")
            print(f"Description: {recipe_info['description']}")
            print("\nComponents:")
            for component_name, template_name in recipe_info['components'].items():
                print(f"  {component_name}: {template_name}")
                
            if recipe_info['variables']:
                print("\nDefault Variables:")
                for var_name, var_value in recipe_info['variables'].items():
                    print(f"  {var_name}: {var_value}")
        else:
            # Print list of available recipes
            recipes = self._recipe_library.list_recipes()
            print(f"Available Recipes ({len(recipes)}):")
            for name in recipes:
                info = self._recipe_library.get_recipe_info(name)
                print(f"  {name}: {info['description']}")

#-----------------------------------------------------------------------
# Utility Functions - Convenience Methods for Common Tasks
#-----------------------------------------------------------------------

def create_config_from_template(
    template_name: str,
    variables: Dict[str, Any],
    output_path: Optional[str] = None,
    config_class: Type[T] = PipelineConfig
) -> T:
    """
    Create a configuration from a template and optionally save it.
    
    Args:
        template_name: Template name
        variables: Template variables
        output_path: Optional path to save configuration
        config_class: Configuration class to instantiate
        
    Returns:
        Configuration instance
        
    Raises:
        ValueError: If template not found or creation fails
    """
    # Create template library
    template_library = TemplateLibrary()
    
    # Create configuration from template
    config = template_library.instantiate_template(
        template_name, variables, config_class
    )
    
    # Save configuration if requested
    if output_path:
        manager = ConfigurationManager()
        manager.save_config(config, output_path)
    
    return config

def create_config_from_recipe(
    recipe_name: str,
    variables: Dict[str, Any],
    output_path: Optional[str] = None
) -> PipelineConfig:
    """
    Create a configuration from a recipe and optionally save it.
    
    Args:
        recipe_name: Recipe name
        variables: Recipe variables
        output_path: Optional path to save configuration
        
    Returns:
        Pipeline configuration
        
    Raises:
        ValueError: If recipe not found or creation fails
    """
    # Create recipe library
    recipe_library = RecipeLibrary()
    
    # Create configuration from recipe
    config = recipe_library.instantiate_recipe(recipe_name, variables)
    
    # Save configuration if requested
    if output_path:
        manager = ConfigurationManager()
        manager.save_config(config, output_path)
    
    return config

def create_config_with_wizard(
    wizard_type: str,
    interactive: bool = False,
    output_path: Optional[str] = None,
    **kwargs
) -> PipelineConfig:
    """
    Create a configuration using the wizard.
    
    Args:
        wizard_type: Type of wizard to use
        interactive: Whether to run interactive wizard
        output_path: Optional path to save configuration
        **kwargs: Wizard parameters for non-interactive mode
        
    Returns:
        Generated configuration
        
    Raises:
        ValueError: If wizard type not supported
    """
    # Create wizard
    wizard = ConfigWizard()
    
    # Run wizard
    if interactive:
        config = wizard.run_interactive_wizard(wizard_type)
    else:
        config = wizard.create_configuration(wizard_type, **kwargs)
    
    # Save configuration if requested
    if output_path:
        manager = ConfigurationManager()
        manager.save_config(config, output_path)
    
    return config

def list_available_templates() -> List[str]:
    """
    List all available configuration templates.
    
    Returns:
        List of template names
    """
    library = TemplateLibrary()
    return library.list_templates()

def list_available_recipes() -> List[str]:
    """
    List all available configuration recipes.
    
    Returns:
        List of recipe names
    """
    library = RecipeLibrary()
    return library.list_recipes()

def create_default_crypto_config(
    name: str,
    exchange: str,
    symbols: List[str],
    output_path: Optional[str] = None
) -> PipelineConfig:
    """
    Create a default cryptocurrency configuration.
    
    Args:
        name: Pipeline name
        exchange: Exchange name
        symbols: Trading pairs
        output_path: Optional path to save configuration
        
    Returns:
        Pipeline configuration
    """
    generator = ConfigGenerator("default")
    config = generator.generate_crypto_config(
        strategy_type="momentum",
        exchange=exchange,
        symbols=symbols
    )
    
    # Override name
    config.name = name
    
    # Save configuration if requested
    if output_path:
        manager = ConfigurationManager()
        manager.save_config(config, output_path)
    
    return config

def create_default_equities_config(
    name: str,
    symbols: List[str],
    output_path: Optional[str] = None
) -> PipelineConfig:
    """
    Create a default equities configuration.
    
    Args:
        name: Pipeline name
        symbols: Stock symbols
        output_path: Optional path to save configuration
        
    Returns:
        Pipeline configuration
    """
    generator = ConfigGenerator("default")
    config = generator.generate_equities_config(
        strategy_type="momentum",
        symbols=symbols
    )
    
    # Override name
    config.name = name
    
    # Save configuration if requested
    if output_path:
        manager = ConfigurationManager()
        manager.save_config(config, output_path)
    
    return config