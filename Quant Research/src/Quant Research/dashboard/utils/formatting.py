"""
Formatting utilities for the Quant Research dashboard.

This module provides functions for formatting data values, dates, and other
elements for display in the dashboard visualizations and tables.
"""

import locale
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd


# ========== Number Formatting Functions ==========

def format_currency(
    value: Union[float, int, str],
    currency_symbol: str = "$",
    decimals: int = 2,
    thousands_separator: bool = True,
    abbreviate: bool = False,
    prefix: bool = True,
    color_coded: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Format a numeric value as currency.
    
    Args:
        value: Value to format
        currency_symbol: Currency symbol to use
        decimals: Number of decimal places
        thousands_separator: Whether to include thousands separators
        abbreviate: Whether to abbreviate large numbers (K, M, B)
        prefix: Whether to show currency symbol as prefix (True) or suffix (False)
        color_coded: Whether to return a dict with color information
        
    Returns:
        Formatted currency string or dict with formatted value and color
    """
    if value is None:
        result = ""
    elif pd.isna(value):
        result = ""
    else:
        try:
            # Convert to float
            value_float = float(value)
            
            # Handle abbreviation
            if abbreviate and abs(value_float) >= 1000:
                value_str, suffix = _abbreviate_number(value_float, decimals)
            else:
                # Format with specified decimal places
                format_str = f"{{:,.{decimals}f}}" if thousands_separator else f"{{:.{decimals}f}}"
                value_str = format_str.format(value_float)
                suffix = ""
            
            # Add currency symbol
            if prefix:
                result = f"{currency_symbol}{value_str}{suffix}"
            else:
                result = f"{value_str}{suffix} {currency_symbol}"
                
        except (ValueError, TypeError):
            # Return original value if conversion fails
            result = str(value)
    
    if color_coded:
        try:
            value_float = float(value)
            color = "green" if value_float > 0 else "red" if value_float < 0 else "black"
            return {"value": result, "color": color}
        except (ValueError, TypeError):
            return {"value": result, "color": "black"}
    
    return result


def format_percentage(
    value: Union[float, int, str],
    decimals: int = 2,
    include_sign: bool = False,
    as_multiplier: bool = False,
    color_coded: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Format a numeric value as a percentage.
    
    Args:
        value: Value to format (0.01 = 1%)
        decimals: Number of decimal places
        include_sign: Whether to include + sign for positive values
        as_multiplier: Whether value is already a percentage (False) or a multiplier (True)
        color_coded: Whether to return a dict with color information
        
    Returns:
        Formatted percentage string or dict with formatted value and color
    """
    if value is None:
        result = ""
    elif pd.isna(value):
        result = ""
    else:
        try:
            # Convert to float
            value_float = float(value)
            
            # Convert from multiplier to percentage if needed
            if as_multiplier:
                value_float = value_float * 100
            
            # Format with specified decimal places
            format_str = f"{{:.{decimals}f}}%"
            result = format_str.format(value_float)
            
            # Add + sign if requested
            if include_sign and value_float > 0:
                result = f"+{result}"
                
        except (ValueError, TypeError):
            # Return original value if conversion fails
            result = str(value)
    
    if color_coded:
        try:
            value_float = float(value)
            if as_multiplier:
                value_float = value_float * 100
            color = "green" if value_float > 0 else "red" if value_float < 0 else "black"
            return {"value": result, "color": color}
        except (ValueError, TypeError):
            return {"value": result, "color": "black"}
    
    return result


def format_number(
    value: Union[float, int, str],
    decimals: int = 2,
    thousands_separator: bool = True,
    abbreviate: bool = False,
    rounding_method: str = 'round',
    color_coded: bool = False,
    threshold: float = 0
) -> Union[str, Dict[str, Any]]:
    """
    Format a numeric value.
    
    Args:
        value: Value to format
        decimals: Number of decimal places
        thousands_separator: Whether to include thousands separators
        abbreviate: Whether to abbreviate large numbers (K, M, B)
        rounding_method: Method for rounding ('round', 'floor', 'ceil')
        color_coded: Whether to return a dict with color information
        threshold: Threshold for color coding (for comparing to specific value)
        
    Returns:
        Formatted number string or dict with formatted value and color
    """
    if value is None:
        result = ""
    elif pd.isna(value):
        result = ""
    else:
        try:
            # Convert to float
            value_float = float(value)
            
            # Apply rounding method
            if rounding_method == 'floor':
                value_float = np.floor(value_float * 10**decimals) / 10**decimals
            elif rounding_method == 'ceil':
                value_float = np.ceil(value_float * 10**decimals) / 10**decimals
            # Default is 'round', which is handled by format string
            
            # Handle abbreviation
            if abbreviate and abs(value_float) >= 1000:
                value_str, suffix = _abbreviate_number(value_float, decimals)
            else:
                # Format with specified decimal places
                format_str = f"{{:,.{decimals}f}}" if thousands_separator else f"{{:.{decimals}f}}"
                value_str = format_str.format(value_float)
                suffix = ""
            
            result = f"{value_str}{suffix}"
                
        except (ValueError, TypeError):
            # Return original value if conversion fails
            result = str(value)
    
    if color_coded:
        try:
            value_float = float(value)
            if threshold != 0:
                color = "green" if value_float > threshold else "red" if value_float < threshold else "black"
            else:
                color = "green" if value_float > 0 else "red" if value_float < 0 else "black"
            return {"value": result, "color": color}
        except (ValueError, TypeError):
            return {"value": result, "color": "black"}
    
    return result


def format_scientific(
    value: Union[float, int, str],
    precision: int = 6,
    color_coded: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Format a numeric value in scientific notation.
    
    Args:
        value: Value to format
        precision: Number of significant digits
        color_coded: Whether to return a dict with color information
        
    Returns:
        Formatted scientific notation string or dict with formatted value and color
    """
    if value is None:
        result = ""
    elif pd.isna(value):
        result = ""
    else:
        try:
            # Convert to float
            value_float = float(value)
            
            # Format in scientific notation
            result = f"{value_float:.{precision}e}"
                
        except (ValueError, TypeError):
            # Return original value if conversion fails
            result = str(value)
    
    if color_coded:
        try:
            value_float = float(value)
            color = "green" if value_float > 0 else "red" if value_float < 0 else "black"
            return {"value": result, "color": color}
        except (ValueError, TypeError):
            return {"value": result, "color": "black"}
    
    return result


def _abbreviate_number(value: float, decimals: int = 2) -> Tuple[str, str]:
    """
    Abbreviate a large number with suffix (K, M, B, T).
    
    Args:
        value: Number to abbreviate
        decimals: Number of decimal places
        
    Returns:
        Tuple of (abbreviated_value, suffix)
    """
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_value < 1000:
        return f"{sign}{abs_value:.{decimals}f}", ""
    elif abs_value < 1000000:
        return f"{sign}{abs_value/1000:.{decimals}f}", "K"
    elif abs_value < 1000000000:
        return f"{sign}{abs_value/1000000:.{decimals}f}", "M"
    elif abs_value < 1000000000000:
        return f"{sign}{abs_value/1000000000:.{decimals}f}", "B"
    else:
        return f"{sign}{abs_value/1000000000000:.{decimals}f}", "T"


# ========== Date/Time Formatting Functions ==========

def format_date(
    date_value: Union[datetime, str, pd.Timestamp],
    format_str: str = "%Y-%m-%d",
    locale_str: Optional[str] = None
) -> str:
    """
    Format a date value.
    
    Args:
        date_value: Date to format
        format_str: Format string (strftime format)
        locale_str: Optional locale for localized date formatting
        
    Returns:
        Formatted date string
    """
    if date_value is None or pd.isna(date_value):
        return ""
    
    try:
        # Convert to datetime if string
        if isinstance(date_value, str):
            date_value = pd.to_datetime(date_value)
        
        # Set locale if specified
        if locale_str:
            old_locale = locale.getlocale(locale.LC_TIME)
            locale.setlocale(locale.LC_TIME, locale_str)
        
        # Format the date
        result = date_value.strftime(format_str)
        
        # Reset locale if needed
        if locale_str:
            locale.setlocale(locale.LC_TIME, old_locale)
        
        return result
        
    except (ValueError, TypeError, AttributeError):
        # Return original value if conversion fails
        return str(date_value)


def format_time(
    time_value: Union[datetime, str, pd.Timestamp],
    format_str: str = "%H:%M:%S",
    include_milliseconds: bool = False,
    timezone: Optional[str] = None
) -> str:
    """
    Format a time value.
    
    Args:
        time_value: Time to format
        format_str: Format string (strftime format)
        include_milliseconds: Whether to include milliseconds
        timezone: Optional timezone for conversion
        
    Returns:
        Formatted time string
    """
    if time_value is None or pd.isna(time_value):
        return ""
    
    try:
        # Convert to datetime if string
        if isinstance(time_value, str):
            time_value = pd.to_datetime(time_value)
        
        # Convert timezone if specified
        if timezone:
            time_value = time_value.tz_convert(timezone)
        
        # Format the time
        result = time_value.strftime(format_str)
        
        # Add milliseconds if requested
        if include_milliseconds:
            ms = time_value.microsecond // 1000
            result = f"{result}.{ms:03d}"
        
        return result
        
    except (ValueError, TypeError, AttributeError):
        # Return original value if conversion fails
        return str(time_value)


def format_timespan(
    seconds: Union[float, int, timedelta],
    include_milliseconds: bool = False,
    abbreviated: bool = False,
    max_units: int = 2
) -> str:
    """
    Format a timespan (duration) in human-readable form.
    
    Args:
        seconds: Number of seconds or timedelta object
        include_milliseconds: Whether to include milliseconds
        abbreviated: Whether to use abbreviated units (s, m, h, d)
        max_units: Maximum number of units to include
        
    Returns:
        Formatted timespan string (e.g., "2h 30m 45s" or "2 hours, 30 minutes")
    """
    if seconds is None or pd.isna(seconds):
        return ""
    
    try:
        # Convert timedelta to seconds if needed
        if isinstance(seconds, timedelta):
            seconds = seconds.total_seconds()
        
        # Convert to float
        seconds_float = float(seconds)
        
        # Define time units
        if abbreviated:
            units = [('d', 86400), ('h', 3600), ('m', 60), ('s', 1)]
            if include_milliseconds:
                units.append(('ms', 0.001))
        else:
            units = [
                ('day', 'days', 86400),
                ('hour', 'hours', 3600),
                ('minute', 'minutes', 60),
                ('second', 'seconds', 1)
            ]
            if include_milliseconds:
                units.append(('millisecond', 'milliseconds', 0.001))
        
        # Extract time units
        result_parts = []
        remaining = abs(seconds_float)
        
        for unit_info in units:
            if abbreviated:
                unit_abbr, unit_seconds = unit_info
                if remaining >= unit_seconds or (unit_seconds == 1 and not result_parts):
                    unit_value = int(remaining / unit_seconds)
                    remaining %= unit_seconds
                    result_parts.append(f"{unit_value}{unit_abbr}")
                    if len(result_parts) >= max_units:
                        break
            else:
                unit_singular, unit_plural, unit_seconds = unit_info
                if remaining >= unit_seconds or (unit_seconds == 1 and not result_parts):
                    unit_value = int(remaining / unit_seconds)
                    remaining %= unit_seconds
                    unit_name = unit_singular if unit_value == 1 else unit_plural
                    result_parts.append(f"{unit_value} {unit_name}")
                    if len(result_parts) >= max_units:
                        break
        
        # Join parts
        if abbreviated:
            result = " ".join(result_parts)
        else:
            result = ", ".join(result_parts)
        
        # Add negative sign if original value was negative
        if seconds_float < 0:
            result = f"-{result}"
        
        return result
        
    except (ValueError, TypeError):
        # Return original value if conversion fails
        return str(seconds)


def format_relative_time(
    timestamp: Union[datetime, str, pd.Timestamp],
    reference_time: Optional[Union[datetime, str, pd.Timestamp]] = None,
    max_units: int = 1,
    include_seconds: bool = True
) -> str:
    """
    Format a timestamp as a relative time (e.g., "5 minutes ago").
    
    Args:
        timestamp: Timestamp to format
        reference_time: Reference time (defaults to current time)
        max_units: Maximum number of time units to include
        include_seconds: Whether to include seconds for small differences
        
    Returns:
        Formatted relative time string
    """
    if timestamp is None or pd.isna(timestamp):
        return ""
    
    try:
        # Convert to datetime if string
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Use current time if reference not provided
        if reference_time is None:
            reference_time = datetime.now()
        elif isinstance(reference_time, str):
            reference_time = pd.to_datetime(reference_time)
        
        # Calculate time difference
        is_past = timestamp < reference_time
        if is_past:
            diff = reference_time - timestamp
            suffix = "ago"
        else:
            diff = timestamp - reference_time
            suffix = "from now"
        
        # Get total seconds
        seconds = diff.total_seconds()
        
        # Define time units
        units = [
            ('year', 'years', 365.25 * 24 * 3600),
            ('month', 'months', 30.44 * 24 * 3600),
            ('day', 'days', 24 * 3600),
            ('hour', 'hours', 3600),
            ('minute', 'minutes', 60)
        ]
        
        if include_seconds:
            units.append(('second', 'seconds', 1))
        
        # Handle special case for very small differences
        if seconds < 1 and include_seconds:
            return "just now"
        
        # Extract time units
        result_parts = []
        remaining = seconds
        
        for unit_singular, unit_plural, unit_seconds in units:
            if remaining >= unit_seconds:
                unit_value = int(remaining / unit_seconds)
                remaining %= unit_seconds
                unit_name = unit_singular if unit_value == 1 else unit_plural
                result_parts.append(f"{unit_value} {unit_name}")
                if len(result_parts) >= max_units:
                    break
        
        # Handle case where no units were extracted (less than smallest unit)
        if not result_parts:
            if include_seconds:
                result_parts = ["0 seconds"]
            else:
                result_parts = ["less than a minute"]
        
        # Join parts and add suffix
        if max_units == 1:
            # Simple format: "5 minutes ago"
            return f"{result_parts[0]} {suffix}"
        else:
            # Complex format: "5 minutes, 30 seconds ago"
            return f"{', '.join(result_parts)} {suffix}"
        
    except (ValueError, TypeError, AttributeError):
        # Return original value if conversion fails
        return str(timestamp)


# ========== String Formatting Functions ==========

def format_ticker_symbol(
    ticker: str,
    include_exchange: bool = True,
    uppercase: bool = True
) -> str:
    """
    Format a financial ticker symbol.
    
    Args:
        ticker: Ticker symbol
        include_exchange: Whether to include exchange prefix/suffix
        uppercase: Whether to convert to uppercase
        
    Returns:
        Formatted ticker symbol
    """
    if ticker is None or pd.isna(ticker) or not isinstance(ticker, str):
        return ""
    
    result = ticker.strip()
    
    # Handle exchange removal if requested
    if not include_exchange:
        # Remove common exchange suffixes like .L, .PA, etc.
        result = re.sub(r'\.[A-Z]{1,4}$', '', result)
        
        # Remove exchange prefixes like NYSE:, NASDAQ:, etc.
        result = re.sub(r'^[A-Z]+:', '', result)
    
    # Convert to uppercase if requested
    if uppercase:
        result = result.upper()
    
    return result


def format_truncated_text(
    text: str,
    max_length: int = 50,
    truncation_indicator: str = "...",
    truncate_middle: bool = False
) -> str:
    """
    Truncate text to maximum length with indicator.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (including indicator)
        truncation_indicator: Indicator string for truncation
        truncate_middle: Whether to truncate in the middle
        
    Returns:
        Truncated text
    """
    if text is None or pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Check if truncation is needed
    if len(text) <= max_length:
        return text
    
    indicator_len = len(truncation_indicator)
    
    if truncate_middle:
        # Truncate in the middle
        chars_to_keep = max_length - indicator_len
        if chars_to_keep <= 0:
            return truncation_indicator
        
        # Split characters evenly for start and end
        start_chars = chars_to_keep // 2
        end_chars = chars_to_keep - start_chars
        
        return text[:start_chars] + truncation_indicator + text[-end_chars:]
    else:
        # Truncate at the end
        return text[:max_length - indicator_len] + truncation_indicator


def format_capitalization(
    text: str,
    style: str = 'title'
) -> str:
    """
    Format text with specified capitalization style.
    
    Args:
        text: Text to format
        style: Capitalization style ('title', 'sentence', 'upper', 'lower')
        
    Returns:
        Formatted text
    """
    if text is None or pd.isna(text) or not isinstance(text, str):
        return ""
    
    if style == 'title':
        # Title case (capitalize first letter of each word)
        return text.title()
    elif style == 'sentence':
        # Sentence case (capitalize first letter of sentence)
        if not text:
            return text
        return text[0].upper() + text[1:].lower()
    elif style == 'upper':
        # All uppercase
        return text.upper()
    elif style == 'lower':
        # All lowercase
        return text.lower()
    else:
        # Unknown style, return original
        return text


def format_highlight_text(
    text: str,
    keywords: List[str],
    case_sensitive: bool = False,
    highlight_format: str = "<b>{}</b>"
) -> str:
    """
    Highlight keywords in text.
    
    Args:
        text: Text to highlight in
        keywords: List of keywords to highlight
        case_sensitive: Whether matching should be case-sensitive
        highlight_format: Format string for highlighting
        
    Returns:
        Text with highlighted keywords
    """
    if text is None or pd.isna(text) or not isinstance(text, str):
        return ""
    
    if not keywords:
        return text
    
    result = text
    
    # Perform case-insensitive replacements
    if not case_sensitive:
        # Create a pattern for the keywords
        pattern = '|'.join(re.escape(kw) for kw in keywords if kw)
        
        if not pattern:
            return text
        
        # Use regex to find and replace
        def replace_match(match):
            return highlight_format.format(match.group(0))
        
        result = re.sub(f"({pattern})", replace_match, text, flags=re.IGNORECASE)
    else:
        # Case-sensitive replacements
        for keyword in keywords:
            if keyword:
                pattern = re.escape(keyword)
                result = re.sub(f"({pattern})", lambda m: highlight_format.format(m.group(0)), result)
    
    return result


# ========== Color Formatting Functions ==========

def get_color_gradient(
    value: float,
    min_value: float,
    max_value: float,
    min_color: Tuple[int, int, int] = (255, 0, 0),  # Red
    max_color: Tuple[int, int, int] = (0, 255, 0),  # Green
    neutral_value: Optional[float] = None,
    neutral_color: Optional[Tuple[int, int, int]] = None,
    opacity: float = 1.0
) -> str:
    """
    Get a color from a gradient based on value.
    
    Args:
        value: Value to get color for
        min_value: Minimum value in range
        max_value: Maximum value in range
        min_color: RGB color tuple for minimum value
        max_color: RGB color tuple for maximum value
        neutral_value: Optional neutral midpoint value
        neutral_color: RGB color tuple for neutral value
        opacity: Opacity value (0.0 to 1.0)
        
    Returns:
        Color in rgba format: 'rgba(r, g, b, a)'
    """
    if value is None or pd.isna(value):
        # Default to gray for None/NaN values
        return f"rgba(128, 128, 128, {opacity})"
    
    try:
        value_float = float(value)
        
        # Handle equal min/max
        if min_value == max_value:
            # Use max color for equal values
            r, g, b = max_color
            return f"rgba({r}, {g}, {b}, {opacity})"
        
        # Handle neutral midpoint
        if neutral_value is not None and neutral_color is not None:
            if value_float <= neutral_value:
                # Use gradient from min to neutral
                ratio = (value_float - min_value) / (neutral_value - min_value)
                ratio = max(0, min(1, ratio))  # Clamp between 0 and 1
                
                r = int(min_color[0] + ratio * (neutral_color[0] - min_color[0]))
                g = int(min_color[1] + ratio * (neutral_color[1] - min_color[1]))
                b = int(min_color[2] + ratio * (neutral_color[2] - min_color[2]))
            else:
                # Use gradient from neutral to max
                ratio = (value_float - neutral_value) / (max_value - neutral_value)
                ratio = max(0, min(1, ratio))  # Clamp between 0 and 1
                
                r = int(neutral_color[0] + ratio * (max_color[0] - neutral_color[0]))
                g = int(neutral_color[1] + ratio * (max_color[1] - neutral_color[1]))
                b = int(neutral_color[2] + ratio * (max_color[2] - neutral_color[2]))
        else:
            # Simple linear gradient
            ratio = (value_float - min_value) / (max_value - min_value)
            ratio = max(0, min(1, ratio))  # Clamp between 0 and 1
            
            r = int(min_color[0] + ratio * (max_color[0] - min_color[0]))
            g = int(min_color[1] + ratio * (max_color[1] - min_color[1]))
            b = int(min_color[2] + ratio * (max_color[2] - min_color[2]))
        
        # Ensure RGB values are in valid range
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        return f"rgba({r}, {g}, {b}, {opacity})"
        
    except (ValueError, TypeError):
        # Default to gray for invalid values
        return f"rgba(128, 128, 128, {opacity})"


def get_performance_color(
    value: float,
    neutral_threshold: float = 0.0,
    positive_color: str = "rgba(0, 128, 0, 0.7)",  # Green
    negative_color: str = "rgba(255, 0, 0, 0.7)",  # Red
    neutral_color: str = "rgba(128, 128, 128, 0.7)"  # Gray
) -> str:
    """
    Get color for performance values (returns, etc.).
    
    Args:
        value: Value to get color for
        neutral_threshold: Threshold for neutral color
        positive_color: Color for positive values
        negative_color: Color for negative values
        neutral_color: Color for neutral values
        
    Returns:
        Color string (rgba or hex)
    """
    if value is None or pd.isna(value):
        return neutral_color
    
    try:
        value_float = float(value)
        
        if value_float > neutral_threshold:
            return positive_color
        elif value_float < neutral_threshold:
            return negative_color
        else:
            return neutral_color
            
    except (ValueError, TypeError):
        return neutral_color


def get_heatmap_color(
    value: float,
    min_value: float = -1.0,
    max_value: float = 1.0,
    colorscale: str = 'RdBu_r'
) -> str:
    """
    Get color for heatmap visualization (correlation, etc.).
    
    Args:
        value: Value to get color for
        min_value: Minimum value in range
        max_value: Maximum value in range
        colorscale: Name of colorscale ('RdBu_r', 'YlOrRd', etc.)
        
    Returns:
        Color string in RGB format 'rgb(r, g, b)'
    """
    if value is None or pd.isna(value):
        return "rgb(255, 255, 255)"  # White for missing values
    
    try:
        value_float = float(value)
        
        # Handle equal min/max
        if min_value == max_value:
            return "rgb(200, 200, 200)"  # Gray for equal values
        
        # Normalize value to [0, 1]
        norm_value = (value_float - min_value) / (max_value - min_value)
        norm_value = max(0, min(1, norm_value))  # Clamp between 0 and 1
        
        # Common colorscales
        if colorscale == 'RdBu_r':
            # Red to White to Blue (reversed)
            if norm_value <= 0.5:
                # Red to White (0 to 0.5 normalized)
                ratio = norm_value * 2  # Scale to [0, 1]
                r = 255
                g = int(255 * ratio)
                b = int(255 * ratio)
            else:
                # White to Blue (0.5 to 1 normalized)
                ratio = (norm_value - 0.5) * 2  # Scale to [0, 1]
                r = int(255 * (1 - ratio))
                g = int(255 * (1 - ratio))
                b = 255
        elif colorscale == 'YlOrRd':
            # Yellow to Orange to Red
            if norm_value <= 0.5:
                # Yellow to Orange (0 to 0.5 normalized)
                ratio = norm_value * 2  # Scale to [0, 1]
                r = 255
                g = int(255 * (1 - ratio * 0.5))  # Yellow (255,255,0) to Orange (255,128,0)
                b = 0
            else:
                # Orange to Red (0.5 to 1 normalized)
                ratio = (norm_value - 0.5) * 2  # Scale to [0, 1]
                r = 255
                g = int(128 * (1 - ratio))  # Orange (255,128,0) to Red (255,0,0)
                b = 0
        else:
            # Default: Grayscale
            level = int(255 * (1 - norm_value))
            r = g = b = level
        
        # Ensure RGB values are in valid range
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        return f"rgb({r}, {g}, {b})"
        
    except (ValueError, TypeError):
        return "rgb(200, 200, 200)"  # Gray for invalid values


# ========== Table Formatting Functions ==========

def format_table_cell(
    value: Any,
    cell_type: str = 'auto',
    **format_args
) -> Dict[str, Any]:
    """
    Format a value for display in a table cell.
    
    Args:
        value: Cell value
        cell_type: Type of cell ('number', 'currency', 'percentage', 'date', 'time', 'text', 'auto')
        **format_args: Additional arguments for formatting functions
        
    Returns:
        Dict with formatted value and display properties
    """
    result = {"value": "", "style": {}}
    
    if value is None or pd.isna(value):
        return result
    
    # Auto-detect type if not specified
    if cell_type == 'auto':
        if isinstance(value, (int, float)):
            if -10 < value < 10 and value != 0:
                cell_type = 'percentage'
            else:
                cell_type = 'number'
        elif isinstance(value, (datetime, pd.Timestamp)):
            cell_type = 'date'
        else:
            cell_type = 'text'
    
    # Format based on type
    if cell_type == 'number':
        formatted = format_number(value, color_coded=True, **format_args)
        result["value"] = formatted["value"]
        result["style"]["color"] = formatted["color"]
        result["style"]["textAlign"] = "right"
        
    elif cell_type == 'currency':
        formatted = format_currency(value, color_coded=True, **format_args)
        result["value"] = formatted["value"]
        result["style"]["color"] = formatted["color"]
        result["style"]["textAlign"] = "right"
        
    elif cell_type == 'percentage':
        formatted = format_percentage(value, color_coded=True, **format_args)
        result["value"] = formatted["value"]
        result["style"]["color"] = formatted["color"]
        result["style"]["textAlign"] = "right"
        
    elif cell_type == 'date':
        result["value"] = format_date(value, **format_args)
        result["style"]["textAlign"] = "left"
        
    elif cell_type == 'time':
        result["value"] = format_time(value, **format_args)
        result["style"]["textAlign"] = "left"
        
    else:  # text
        result["value"] = str(value)
        result["style"]["textAlign"] = "left"
    
    return result


def format_dataframe_styles(
    df: pd.DataFrame,
    column_formats: Dict[str, Dict[str, Any]] = None,
    index_format: Optional[Dict[str, Any]] = None,
    header_format: Optional[Dict[str, Any]] = None,
    conditional_formats: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Format a DataFrame for display with styles.
    
    Args:
        df: DataFrame to format
        column_formats: Dict of column name -> format settings
        index_format: Format settings for index
        header_format: Format settings for header
        conditional_formats: List of conditional format settings
        
    Returns:
        Dict with formatted data and style information
    """
    if df is None or df.empty:
        return {"data": [], "styles": {}}
    
    # Initialize result structure
    result = {
        "data": [],
        "styles": {
            "cells": {},
            "columns": {},
            "index": {},
            "headers": {}
        }
    }
    
    # Convert DataFrame to list of records for data
    result["data"] = df.reset_index().to_dict('records')
    
    # Apply column formats
    if column_formats:
        for col, format_spec in column_formats.items():
            if col in df.columns:
                col_idx = list(df.columns).index(col)
                cell_type = format_spec.get('type', 'auto')
                format_args = {k: v for k, v in format_spec.items() if k != 'type'}
                
                # Format each cell in the column
                for row_idx in range(len(df)):
                    cell_key = f"{row_idx},{col_idx}"
                    cell_value = df.iloc[row_idx, col_idx]
                    formatted = format_table_cell(cell_value, cell_type, **format_args)
                    result["styles"]["cells"][cell_key] = formatted["style"]
                
                # Store column style
                result["styles"]["columns"][col] = {
                    "width": format_spec.get('width'),
                    "align": format_spec.get('align')
                }
    
    # Apply index format
    if index_format:
        cell_type = index_format.get('type', 'text')
        format_args = {k: v for k, v in index_format.items() if k != 'type'}
        
        for row_idx in range(len(df)):
            index_value = df.index[row_idx]
            formatted = format_table_cell(index_value, cell_type, **format_args)
            result["styles"]["index"][row_idx] = formatted["style"]
    
    # Apply header format
    if header_format:
        for col_idx, col_name in enumerate(df.columns):
            result["styles"]["headers"][col_idx] = {
                "fontWeight": header_format.get('fontWeight', 'bold'),
                "backgroundColor": header_format.get('backgroundColor'),
                "color": header_format.get('color'),
                "textAlign": header_format.get('align')
            }
    
    # Apply conditional formats
    if conditional_formats:
        for cf in conditional_formats:
            column = cf.get('column')
            condition = cf.get('condition')
            value = cf.get('value')
            style = cf.get('style', {})
            
            if column and condition and column in df.columns:
                col_idx = list(df.columns).index(column)
                
                # Apply condition to each cell
                for row_idx in range(len(df)):
                    cell_value = df.iloc[row_idx, col_idx]
                    cell_key = f"{row_idx},{col_idx}"
                    
                    if _check_condition(cell_value, condition, value):
                        # Update cell style
                        if cell_key not in result["styles"]["cells"]:
                            result["styles"]["cells"][cell_key] = {}
                        
                        result["styles"]["cells"][cell_key].update(style)
    
    return result


def _check_condition(value: Any, condition: str, target: Any) -> bool:
    """
    Check if a value satisfies a condition.
    
    Args:
        value: Value to check
        condition: Condition operator ('eq', 'ne', 'gt', 'lt', 'ge', 'le', 'contains', etc.)
        target: Target value for comparison
        
    Returns:
        True if condition is satisfied, False otherwise
    """
    if value is None or pd.isna(value):
        return False
    
    try:
        if condition == 'eq':
            return value == target
        elif condition == 'ne':
            return value != target
        elif condition == 'gt':
            return value > target
        elif condition == 'lt':
            return value < target
        elif condition == 'ge':
            return value >= target
        elif condition == 'le':
            return value <= target
        elif condition == 'contains':
            return target in str(value)
        elif condition == 'startswith':
            return str(value).startswith(target)
        elif condition == 'endswith':
            return str(value).endswith(target)
        elif condition == 'between':
            if isinstance(target, (list, tuple)) and len(target) >= 2:
                return target[0] <= value <= target[1]
            return False
        elif condition == 'in':
            if isinstance(target, (list, tuple)):
                return value in target
            return False
        else:
            return False
    except Exception:
        return False