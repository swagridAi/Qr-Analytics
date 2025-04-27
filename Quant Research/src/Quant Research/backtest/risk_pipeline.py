"""
Risk Management Pipeline.

This module provides a standardized pipeline for applying risk controls
to strategy positions, ensuring consistent application across strategies.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import pandas as pd
import numpy as np

from quant_research.backtest import risk
from quant_research.backtest.utils import validate_dataframe

logger = logging.getLogger(__name__)


# ===============================================================================
# Parameter Validation Helpers
# ===============================================================================

def _validate_numeric_param(
    param_name: str,
    param_value: Any,
    valid_params: Dict[str, Any],
    warnings: List[str],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    default: Optional[Any] = None,
    required_type: Optional[type] = None
) -> None:
    """
    Validate a numeric parameter.
    
    Args:
        param_name: Name of the parameter
        param_value: Value to validate
        valid_params: Dictionary to store valid parameters
        warnings: List to store validation warnings
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        default: Default value if validation fails
        required_type: Specific type requirement (int, float, etc.)
    """
    if param_value is not None:
        # Check type
        type_valid = isinstance(param_value, (int, float))
        if required_type:
            type_valid = isinstance(param_value, required_type)
        
        if not type_valid:
            warnings.append(f"{param_name} must be {required_type.__name__ if required_type else 'numeric'}, "
                           f"got {type(param_value).__name__}")
            valid_params[param_name] = default
            return
        
        # Check bounds
        if min_val is not None and param_value < min_val:
            warnings.append(f"{param_name} must be at least {min_val}, got {param_value}")
            valid_params[param_name] = default
            return
        
        if max_val is not None and param_value > max_val:
            warnings.append(f"{param_name} must be at most {max_val}, got {param_value}")
            valid_params[param_name] = default
            return
        
        # Valid parameter
        valid_params[param_name] = param_value
    else:
        valid_params[param_name] = None


def _validate_boolean_param(
    param_name: str,
    param_value: Any,
    valid_params: Dict[str, Any],
    warnings: List[str],
    default: bool = False
) -> None:
    """
    Validate a boolean parameter.
    
    Args:
        param_name: Name of the parameter
        param_value: Value to validate
        valid_params: Dictionary to store valid parameters
        warnings: List to store validation warnings
        default: Default value if validation fails
    """
    if not isinstance(param_value, bool):
        warnings.append(f"{param_name} must be boolean, got {type(param_value).__name__}")
        valid_params[param_name] = default
    else:
        valid_params[param_name] = param_value


def _validate_dict_param(
    param_name: str,
    param_value: Any,
    valid_params: Dict[str, Any],
    warnings: List[str],
    validator_func: Optional[Callable] = None
) -> None:
    """
    Validate a dictionary parameter.
    
    Args:
        param_name: Name of the parameter
        param_value: Value to validate
        valid_params: Dictionary to store valid parameters
        warnings: List to store validation warnings
        validator_func: Optional function to validate dictionary contents
    """
    if not isinstance(param_value, dict):
        warnings.append(f"{param_name} must be a dictionary, got {type(param_value).__name__}")
        return
    
    # Apply additional validation if provided
    if validator_func:
        param_value = validator_func(param_value, warnings)
    
    valid_params[param_name] = param_value


def validate_risk_parameters(risk_params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate risk management parameters.
    
    Args:
        risk_params: Dictionary of risk parameters
        
    Returns:
        Tuple of (valid_params, warnings)
    """
    valid_params = {}
    warnings = []
    
    # Parameter groups for validation
    param_groups = {
        # Stop loss parameters
        'stop_loss': [
            # param_name, min_val, max_val, default, required_type
            ('stop_loss_pct', 0, None, None, None),
            ('trailing_stop', None, None, False, bool),
            ('atr_multiple', 0, None, None, None),
            ('atr_periods', 1, None, 14, int),
            ('time_stop_periods', 1, None, None, int)
        ],
        
        # Position sizing parameters
        'position_sizing': [
            ('use_kelly_sizing', None, None, False, bool),
            ('kelly_fraction', 0, 1, 0.5, float),
            ('kelly_lookback', 1, None, 252, int),
            ('target_volatility', 0, None, None, None),
            ('vol_lookback', 1, None, 20, int)
        ],
        
        # Drawdown and trend parameters
        'drawdown_trend': [
            ('max_drawdown_pct', 0, None, None, None),
            ('use_trend_filter', None, None, False, bool),
            ('ma_periods', 1, None, 200, int),
            ('long_only_above_ma', None, None, True, bool),
            ('short_only_below_ma', None, None, True, bool)
        ],
        
        # Position limit parameters
        'position_limits': [
            ('max_portfolio_pct', 0, 1, None, float),
            ('portfolio_value', 0, None, None, None)
        ],
        
        # Correlation parameters
        'correlation': [
            ('use_correlation_scaling', None, None, False, bool),
            ('max_correlation', 0, 1, 0.7, float),
            ('correlation_lookback', 1, None, 60, int)
        ],
        
        # VaR parameters
        'var': [
            ('var_confidence_level', 0, 1, 0.95, float),
            ('var_lookback', 1, None, 252, int),
            ('var_method', None, None, 'historical', str)
        ]
    }
    
    # Validate parameters by group
    for group_name, params in param_groups.items():
        for param_info in params:
            param_name = param_info[0]
            
            if param_name not in risk_params:
                continue
                
            # Extract validation criteria
            min_val = param_info[1] if len(param_info) > 1 else None
            max_val = param_info[2] if len(param_info) > 2 else None
            default = param_info[3] if len(param_info) > 3 else None
            required_type = param_info[4] if len(param_info) > 4 else None
            
            # Handle different parameter types
            if required_type == bool:
                _validate_boolean_param(
                    param_name, 
                    risk_params[param_name], 
                    valid_params, 
                    warnings,
                    default=default
                )
            else:
                _validate_numeric_param(
                    param_name, 
                    risk_params[param_name], 
                    valid_params, 
                    warnings,
                    min_val=min_val,
                    max_val=max_val,
                    default=default,
                    required_type=required_type
                )
    
    # Special handling for dictionary parameters
    
    # 1. max_position_size (can be a number or a dict)
    if 'max_position_size' in risk_params:
        max_pos = risk_params['max_position_size']
        if max_pos is not None:
            if isinstance(max_pos, dict):
                # Validate the dictionary values
                is_valid = True
                for asset, limit in max_pos.items():
                    if not isinstance(limit, (int, float)) or limit <= 0:
                        warnings.append(f"Position limit for {asset} must be positive numeric, got {limit}")
                        is_valid = False
                
                if is_valid:
                    valid_params['max_position_size'] = max_pos
            elif isinstance(max_pos, (int, float)):
                if max_pos <= 0:
                    warnings.append(f"max_position_size must be positive, got {max_pos}")
                else:
                    valid_params['max_position_size'] = max_pos
            else:
                warnings.append(f"max_position_size must be numeric or dict, got {type(max_pos).__name__}")
        else:
            valid_params['max_position_size'] = None
    
    # 2. sector_mappings and sector_limits
    if 'sector_limits' in risk_params:
        sector_limits = risk_params['sector_limits']
        if sector_limits is not None:
            if not isinstance(sector_limits, dict):
                warnings.append(f"sector_limits must be a dictionary, got {type(sector_limits).__name__}")
            else:
                # Check for sector mappings
                sector_mappings = risk_params.get('sector_mappings')
                if sector_mappings is None:
                    warnings.append("sector_mappings is required when using sector_limits")
                elif not isinstance(sector_mappings, dict):
                    warnings.append(f"sector_mappings must be a dictionary, got {type(sector_mappings).__name__}")
                else:
                    valid_params['sector_mappings'] = sector_mappings
                    valid_params['sector_limits'] = sector_limits
        else:
            valid_params['sector_limits'] = None
    
    return valid_params, warnings


# ===============================================================================
# Risk Control Application Helpers
# ===============================================================================

def _apply_risk_control(
    result: pd.DataFrame,
    prices_df: pd.DataFrame,
    control_name: str,
    control_func: Callable,
    params: Dict[str, Any],
    applied_controls: List[str]
) -> pd.DataFrame:
    """
    Apply a risk control with standardized error handling and logging.
    
    Args:
        result: DataFrame with positions
        prices_df: DataFrame with price data
        control_name: Name of the risk control
        control_func: Function to apply the risk control
        params: Parameters for the risk control function
        applied_controls: List to track applied controls
        
    Returns:
        DataFrame with risk control applied
    """
    try:
        logger.info(f"Applying {control_name} with parameters: {params}")
        result = control_func(result, prices_df, **params)
        applied_controls.append(control_name)
        return result
    except Exception as e:
        logger.error(f"Error applying {control_name}: {str(e)}")
        return result


def apply_risk_pipeline(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    risk_params: Dict[str, Any],
    strategy_name: str = "unknown"
) -> pd.DataFrame:
    """
    Apply a standardized risk management pipeline to positions.
    
    This function applies risk controls in a consistent order:
    1. Position sizing (Kelly, volatility targeting)
    2. Stop loss mechanisms (standard, ATR-based)
    3. Time-based stops
    4. Drawdown protection
    5. Trend filtering
    6. Position limits
    7. Correlation management
    8. Sector exposure limits
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price data
        risk_params: Dictionary of risk parameters
        strategy_name: Name of strategy (for logging)
        
    Returns:
        DataFrame with risk-adjusted positions
    """
    logger.info(f"Applying risk management pipeline for {strategy_name} strategy")
    
    # Validate input data
    if positions_df is None or positions_df.empty:
        logger.warning("No positions to apply risk controls to")
        return pd.DataFrame()
    
    if prices_df is None or prices_df.empty:
        logger.warning("No price data available for risk controls")
        return pd.DataFrame()
    
    # Validate risk parameters
    valid_params, warnings = validate_risk_parameters(risk_params)
    
    # Log any validation warnings
    for warning in warnings:
        logger.warning(f"Risk parameter warning: {warning}")
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Track applied controls
    applied_controls = []
    
    # Define risk control definitions
    # Each entry contains:
    # 1. Check function (returns True if control should be applied)
    # 2. Risk control function to apply
    # 3. Parameter mapping function (extracts relevant parameters)
    # 4. Control name for logging
    risk_controls = [
        # 1. Position Sizing Methods
        (
            lambda p: p.get('use_kelly_sizing', False),
            risk.apply_kelly_sizing,
            lambda p: {
                'lookback': p.get('kelly_lookback', 252),
                'fraction': p.get('kelly_fraction', 0.5)
            },
            "Kelly Criterion Sizing"
        ),
        (
            lambda p: p.get('target_volatility') is not None,
            risk.apply_vol_targeting,
            lambda p: {
                'target_volatility': p.get('target_volatility'),
                'lookback': p.get('vol_lookback', 20)
            },
            "Volatility Targeting"
        ),
        
        # 2. Stop Loss Methods
        (
            lambda p: p.get('stop_loss_pct') is not None,
            risk.apply_stop_loss,
            lambda p: {
                'stop_loss_pct': p.get('stop_loss_pct'),
                'trailing': p.get('trailing_stop', False)
            },
            "Stop Loss"
        ),
        (
            lambda p: p.get('atr_multiple') is not None,
            risk.apply_volatility_stop,
            lambda p: {
                'atr_multiple': p.get('atr_multiple'),
                'atr_periods': p.get('atr_periods', 14)
            },
            "ATR-based Stop"
        ),
        
        # 3. Time-based Stop
        (
            lambda p: p.get('time_stop_periods') is not None,
            lambda df, _prices_df, **kwargs: risk.apply_time_stop(df, **kwargs),
            lambda p: {
                'max_holding_periods': p.get('time_stop_periods')
            },
            "Time-based Stop"
        ),
        
        # 4. Drawdown Protection
        (
            lambda p: p.get('max_drawdown_pct') is not None,
            risk.apply_drawdown_guard,
            lambda p: {
                'max_drawdown_pct': p.get('max_drawdown_pct')
            },
            "Drawdown Guard"
        ),
        
        # 5. Trend Filtering
        (
            lambda p: p.get('use_trend_filter', False),
            risk.apply_trend_filter,
            lambda p: {
                'ma_periods': p.get('ma_periods', 200),
                'long_only_above_ma': p.get('long_only_above_ma', True),
                'short_only_below_ma': p.get('short_only_below_ma', True)
            },
            "Trend Filter"
        ),
        
        # 6. Position Limits
        (
            lambda p: p.get('max_position_size') is not None,
            risk.apply_position_limits,
            lambda p: {
                'max_position_size': p.get('max_position_size'),
                'max_portfolio_pct': p.get('max_portfolio_pct'),
                'portfolio_value': p.get('portfolio_value')
            },
            "Position Limits"
        ),
        
        # 7. Correlation Management
        (
            lambda p: p.get('use_correlation_scaling', False),
            risk.apply_correlation_scaling,
            lambda p: {
                'lookback': p.get('correlation_lookback', 60),
                'max_correlation': p.get('max_correlation', 0.7)
            },
            "Correlation Scaling"
        ),
        
        # 8. Sector Exposure Limits
        (
            lambda p: p.get('sector_limits') is not None and p.get('sector_mappings') is not None,
            risk.apply_sector_limits,
            lambda p: {
                'sector_mappings': p.get('sector_mappings'),
                'max_sector_exposure': p.get('sector_limits')
            },
            "Sector Exposure Limits"
        )
    ]
    
    # Apply each risk control in sequence
    for check_func, control_func, param_extractor, control_name in risk_controls:
        if check_func(valid_params):
            control_params = param_extractor(valid_params)
            result = _apply_risk_control(
                result, 
                prices_df, 
                control_name, 
                control_func, 
                control_params, 
                applied_controls
            )
    
    # Log the applied risk controls
    if applied_controls:
        logger.info(f"Applied risk controls: {', '.join(applied_controls)}")
    else:
        logger.warning("No risk controls were applied")
    
    return result


def extract_risk_params(strategy_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract risk parameters from strategy parameters.
    
    This function standardizes the way risk parameters are extracted from
    different strategy parameter formats.
    
    Args:
        strategy_params: Dictionary of strategy parameters
        
    Returns:
        Dictionary of risk parameters
    """
    risk_params = {}
    
    # Define all recognized risk parameter names
    risk_param_names = [
        # Position sizing parameters
        'use_kelly_sizing', 'kelly_fraction', 'kelly_lookback',
        'target_volatility', 'vol_lookback',
        
        # Stop loss parameters
        'stop_loss_pct', 'trailing_stop',
        'atr_multiple', 'atr_periods',
        'time_stop_periods',
        
        # Drawdown and trend parameters
        'max_drawdown_pct',
        'use_trend_filter', 'ma_periods',
        'long_only_above_ma', 'short_only_below_ma',
        
        # Position limit parameters
        'max_position_size', 'max_portfolio_pct', 'portfolio_value',
        
        # Correlation parameters
        'use_correlation_scaling', 'max_correlation', 'correlation_lookback',
        
        # Sector parameters
        'sector_limits', 'sector_mappings',
        
        # VaR parameters
        'var_confidence_level', 'var_lookback', 'var_method'
    ]
    
    # Check if there's a dedicated 'risk_params' dictionary
    if 'risk_params' in strategy_params and isinstance(strategy_params['risk_params'], dict):
        # Use the dedicated risk parameters
        risk_params.update(strategy_params['risk_params'])
    
    # Also check for risk parameters at the top level
    for param_name in risk_param_names:
        if param_name in strategy_params and param_name not in risk_params:
            risk_params[param_name] = strategy_params[param_name]
    
    return risk_params


def calculate_risk_metrics(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    risk_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculate risk metrics for positions.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price data
        risk_params: Optional risk parameters for calculation
        
    Returns:
        Dictionary with risk metrics
    """
    if positions_df is None or positions_df.empty:
        logger.warning("No positions for risk metric calculation")
        return {}
    
    if prices_df is None or prices_df.empty:
        logger.warning("No price data for risk metric calculation")
        return {}
    
    # Initialize metrics
    metrics = {}
    
    # Extract parameters
    if risk_params is None:
        risk_params = {}
    
    confidence_level = risk_params.get('var_confidence_level', 0.95)
    var_lookback = risk_params.get('var_lookback', 252)
    var_method = risk_params.get('var_method', 'historical')
    
    try:
        # Calculate Value at Risk (VaR)
        var_result = risk.calculate_portfolio_var(
            positions_df,
            prices_df,
            confidence_level=confidence_level,
            lookback=var_lookback,
            method=var_method
        )
        
        # Extract VaR metrics
        for metric_name in ['portfolio_var', 'portfolio_cvar', 'var_pct_of_portfolio', 'portfolio_vol']:
            if metric_name in var_result.columns:
                # Convert to a more user-friendly name
                friendly_name = {
                    'portfolio_var': 'value_at_risk',
                    'portfolio_cvar': 'conditional_var',
                    'var_pct_of_portfolio': 'var_pct',
                    'portfolio_vol': 'portfolio_volatility'
                }.get(metric_name, metric_name)
                
                metrics[friendly_name] = var_result[metric_name].mean()
    
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {str(e)}")
    
    return metrics