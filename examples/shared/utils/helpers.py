"""Helper utility functions."""
import json
from typing import Any, Dict, Optional
from datetime import datetime


def parse_json_safely(json_string: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON string.
    
    Args:
        json_string: JSON string to parse
        
    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime as ISO string.
    
    Args:
        dt: Datetime object (uses current time if None)
        
    Returns:
        ISO formatted timestamp
    """
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to max length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_dict_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary value.
    
    Args:
        d: Dictionary to search
        *keys: Keys to traverse
        default: Default value if key not found
        
    Returns:
        Value or default
    """
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def format_tool_output(output: Any, max_length: int = 500) -> str:
    """Format tool output for display.
    
    Args:
        output: Tool output to format
        max_length: Maximum length of output
        
    Returns:
        Formatted string
    """
    if isinstance(output, dict):
        output_str = json.dumps(output, indent=2)
    elif isinstance(output, (list, tuple)):
        output_str = json.dumps(output, indent=2)
    else:
        output_str = str(output)
    
    return truncate_string(output_str, max_length)


def count_tokens_rough(text: str) -> int:
    """Rough token count estimation.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Estimated token count
    """
    # Rough estimation: ~4 characters per token
    return len(text) // 4
