"""Tools for the simple agent demo - with Mudipu tracing."""
import random
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mudipu import trace_tool
from shared.schemas.tool_schema import Tool, ToolParameter


@trace_tool("get_current_time")
def get_current_time() -> str:
    """Get the current time.
    
    Returns:
        Current time as formatted string
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@trace_tool("calculate")
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    try:
        # Only allow basic math operations for safety
        allowed_chars = set("0123456789+-*/()%. ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"
        
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


@trace_tool("get_random_number")
def get_random_number(min_val: int = 1, max_val: int = 100) -> int:
    """Get a random number within a range.
    
    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        
    Returns:
        Random integer
    """
    return random.randint(min_val, max_val)


@trace_tool("get_weather")
def get_weather(city: str) -> str:
    """Get weather information for a city (mock data).
    
    Args:
        city: Name of the city
        
    Returns:
        Weather information
    """
    # This is a mock implementation
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Windy"]
    temperature = random.randint(10, 30)
    condition = random.choice(conditions)
    
    return f"Weather in {city}: {condition}, {temperature}°C"


@trace_tool("search_web")
def search_web(query: str) -> str:
    """Search the web for information (mock implementation).
    
    Args:
        query: Search query
        
    Returns:
        Search results
    """
    # This is a mock implementation
    return f"Mock search results for '{query}':\n1. Article about {query}\n2. Wikipedia page on {query}\n3. Latest news about {query}"


def create_simple_tools() -> list:
    """Create the list of tools for the simple agent.
    
    Returns:
        List of Tool objects
    """
    tools = [
        Tool(
            name="get_current_time",
            description="Get the current date and time",
            function=get_current_time,
            parameters=[]
        ),
        Tool(
            name="calculate",
            description="Evaluate a mathematical expression. Supports +, -, *, /, (), and basic math operations.",
            function=calculate,
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate (e.g., '2 + 2', '(10 * 5) / 2')",
                    required=True
                )
            ]
        ),
        Tool(
            name="get_random_number",
            description="Generate a random number within a specified range",
            function=get_random_number,
            parameters=[
                ToolParameter(
                    name="min_val",
                    type="integer",
                    description="Minimum value (inclusive)",
                    required=False
                ),
                ToolParameter(
                    name="max_val",
                    type="integer",
                    description="Maximum value (inclusive)",
                    required=False
                )
            ]
        ),
        Tool(
            name="get_weather",
            description="Get current weather information for a city (mock data)",
            function=get_weather,
            parameters=[
                ToolParameter(
                    name="city",
                    type="string",
                    description="Name of the city",
                    required=True
                )
            ]
        ),
        Tool(
            name="search_web",
            description="Search the web for information (mock implementation)",
            function=search_web,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True
                )
            ]
        )
    ]
    
    return tools
