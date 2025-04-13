# Tools

Tools are a fundamental component of the Scoras framework, allowing agents to perform specific actions and interact with external systems. This page explains how to create, use, and manage tools effectively.

## What are Scoras Tools?

In Scoras, a tool is a function that an agent can call to perform a specific task. Tools:

- Extend an agent's capabilities beyond text generation
- Have well-defined inputs and outputs
- Include metadata like name, description, and complexity
- Can be shared between agents and exposed via protocols

## Creating Basic Tools

The simplest way to create a tool is with the `@sc.tool` decorator:

```python
import scoras as sc

@sc.tool(name="calculator", description="Perform calculations", complexity="simple")
async def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform basic arithmetic operations.
    
    Args:
        operation: Operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
        
    Returns:
        Result of the operation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")
```

## Tool Parameters

Tools use type hints to define their parameters:

```python
@sc.tool(name="weather", description="Get weather information")
async def get_weather(
    location: str,  # Required parameter
    units: str = "metric",  # Optional parameter with default
    forecast_days: int = 1  # Optional parameter with default
) -> dict:
    """Get weather information for a location."""
    # Implementation...
    return {
        "location": location,
        "temperature": 22,
        "units": units,
        "conditions": "sunny",
        "forecast": [{"day": 1, "temp": 22}, {"day": 2, "temp": 24}][:forecast_days]
    }
```

## Tool Complexity

Tools can have different complexity levels that affect the overall complexity score:

```python
# Simple tool (1.4 points)
@sc.tool(name="echo", description="Echo the input", complexity="simple")
async def echo(text: str) -> str:
    return text

# Standard tool (2 points)
@sc.tool(name="weather", description="Get weather information", complexity="standard")
async def get_weather(location: str) -> dict:
    # Implementation...
    return {"temperature": 72, "conditions": "sunny"}

# Complex tool (3 points)
@sc.tool(name="search", description="Search for information", complexity="complex")
async def search(query: str, max_results: int = 5) -> list:
    # Implementation...
    return [{"title": "Result 1", "snippet": "Information..."}]
```

## Adding Tools to Agents

Tools can be added to agents at creation time or later:

```python
# Add tools at creation time
agent = sc.Agent(
    model="openai:gpt-4o",
    tools=[calculator, get_weather, search]
)

# Add a tool after creation
@sc.tool(name="translate", description="Translate text")
async def translate(text: str, source_lang: str, target_lang: str) -> str:
    # Implementation...
    return f"Translated from {source_lang} to {target_lang}: {text}"

agent.add_tool(translate)
```

## Tool Chains

Combine multiple tools into a chain for more complex operations:

```python
from scoras.tools import ToolChain

# Create a tool chain
weather_chain = ToolChain(
    name="weather_analysis",
    description="Analyze weather patterns",
    tools=[get_weather, analyze_temperature, predict_weather],
    complexity="complex"
)

# Add the chain to an agent
agent.add_tool(weather_chain)
```

## Tool Routers

Route requests to the appropriate tool based on content:

```python
from scoras.tools import ToolRouter

# Create a tool router
router = ToolRouter(
    name="knowledge_router",
    description="Route queries to appropriate knowledge tools",
    tools=[search, database_lookup, api_call],
    routing_strategy="content_based"
)

# Add the router to an agent
agent.add_tool(router)
```

## Tool Builder

Create tools dynamically:

```python
from scoras.tools import ToolBuilder

# Create a tool builder
builder = ToolBuilder()

# Build a simple tool
calculator_tool = builder.build(
    name="calculator",
    description="Perform calculations",
    function=lambda op, a, b: eval(f"{a} {op} {b}"),
    parameters={
        "op": {"type": "string", "description": "Operation to perform"},
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"}
    },
    return_type="number",
    complexity="simple"
)

# Add the built tool to an agent
agent.add_tool(calculator_tool)
```

## HTTP Tools

Create tools that interact with web APIs:

```python
from scoras.tools import HTTPTool

# Create an HTTP tool
weather_api = HTTPTool(
    name="weather_api",
    description="Get weather information from an API",
    base_url="https://api.weather.example.com",
    endpoints={
        "current": {
            "path": "/current",
            "method": "GET",
            "parameters": {
                "location": {"type": "string", "required": True},
                "units": {"type": "string", "default": "metric"}
            }
        },
        "forecast": {
            "path": "/forecast",
            "method": "GET",
            "parameters": {
                "location": {"type": "string", "required": True},
                "days": {"type": "integer", "default": 5}
            }
        }
    },
    auth_type="api_key",
    auth_params={"header_name": "X-API-Key", "key": "your-api-key"},
    complexity="standard"
)

# Add the HTTP tool to an agent
agent.add_tool(weather_api)

# Use the tool
result = await agent.run_sync("What's the weather in New York?")
```

## Protocol Integration

Tools can be exposed via MCP and used by A2A agents:

```python
from scoras.mcp import create_mcp_server

# Create an MCP server with tools
server = create_mcp_server(
    name="ToolServer",
    description="Server with various tools",
    tools=[calculator, get_weather, search]
)

# Run the server
await run_mcp_server(server, host="0.0.0.0", port=8000)
```

## Next Steps

- Learn about [Agents](agents.md) that use tools
- Explore [Workflows](workflows.md) for complex tool orchestration
- Understand [RAG Systems](rag.md) for knowledge-enhanced tools
- Dive into [Protocol Support](../protocols/mcp.md) for sharing tools
