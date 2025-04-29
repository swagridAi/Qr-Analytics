"""
Risk management package for backtesting.

This package provides implementations of various risk management techniques including:
- Position sizing (Kelly, volatility targeting, fixed fraction)
- Stop-loss mechanisms
- Drawdown protection
- Exposure controls
"""

# Import all functions from submodules
from quant_research.backtest.risk.position_sizing import (
    apply_kelly_sizing,
    apply_vol_targeting,
    apply_fixed_fraction_sizing
)

from quant_research.backtest.risk.stop_loss import (
    apply_stop_loss,
    apply_volatility_stop,
    apply_time_stop
)

from quant_research.backtest.risk.drawdown import (
    apply_drawdown_guard,
    apply_trend_filter
)

from quant_research.backtest.risk.exposure import (
    apply_position_limits,
    apply_sector_limits,
    apply_correlation_scaling
)

from quant_research.backtest.risk.analytics import (
    calculate_portfolio_var
)

# For backward compatibility, provide all functions at the package level
__all__ = [
    # Position sizing
    'apply_kelly_sizing',
    'apply_vol_targeting',
    'apply_fixed_fraction_sizing',
    
    # Stop loss
    'apply_stop_loss',
    'apply_volatility_stop',
    'apply_time_stop',
    
    # Drawdown protection
    'apply_drawdown_guard',
    'apply_trend_filter',
    
    # Exposure controls
    'apply_position_limits',
    'apply_sector_limits',
    'apply_correlation_scaling',
    
    # Analytics
    'calculate_portfolio_var'
]