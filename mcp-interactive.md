# Interactive MCP Example

This page demonstrates the Model Context Protocol (MCP) in action with an interactive example. You can experiment with different tools, parameters, and see how the complexity score changes as you interact with the system.

## Try It Yourself

Below is an interactive MCP demo that simulates a server with various tools and a client that can call these tools. You can select different tools, provide parameters, and see the results in real-time.

<div id="mcp-example-demo" class="mcp-interactive-demo"></div>

## How It Works

The interactive demo above demonstrates the key components of the MCP protocol:

1. **MCP Server**: The left panel represents an MCP server that exposes tools with different complexity levels.
2. **MCP Client**: The right panel represents an MCP client that can call tools on the server.
3. **Tool Execution**: When you execute a tool, the client sends a request to the server, which processes it and returns a result.
4. **Complexity Scoring**: The bottom panel shows the complexity score, which increases as you use more complex tools.

## Code Example

Here's how you would implement a similar MCP server and client in Scoras:

```python
import scoras as sc
from scoras.mcp import create_mcp_server, run_mcp_server, MCPClient
import asyncio

# Define tools for the server
@sc.tool(name="calculator", description="Perform calculations", complexity="simple")
async def calculator(operation: str, a: float, b: float) -> float:
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

@sc.tool(name="weather", description="Get weather information", complexity="standard")
async def get_weather(location: str) -> dict:
    # In a real implementation, this would call a weather API
    return {
        "location": location,
        "temperature": 72,
        "conditions": "Sunny",
        "humidity": 45,
        "wind_speed": 5
    }

@sc.tool(name="search", description="Search for information", complexity="complex")
async def search(query: str, max_results: int = 3) -> list:
    # In a real implementation, this would perform a search
    results = []
    for i in range(max_results):
        results.append({
            "title": f"Result {i+1} for {query}",
            "snippet": f"This is a search result about {query}."
        })
    return results

async def main():
    # Create an MCP server
    server = create_mcp_server(
        name="ScorasServer",
        description="Scoras MCP server with various tools",
        tools=[calculator, get_weather, search],
        capabilities=["tools", "streaming"],
        enable_scoring=True
    )
    
    # Start the server in the background
    server_task = asyncio.create_task(
        run_mcp_server(server, host="0.0.0.0", port=8000)
    )
    
    # Create an MCP client
    client = MCPClient(
        server_url="http://localhost:8000",
        enable_scoring=True
    )
    
    # Execute the calculator tool
    calc_result = await client.execute_tool(
        tool_name="calculator",
        parameters={
            "operation": "multiply",
            "a": 5,
            "b": 7
        }
    )
    print(f"Calculator result: {calc_result}")
    
    # Get the complexity score
    score = client.get_complexity_score()
    print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")
    
    # Cancel the server task when done
    server_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Features

The MCP implementation in Scoras provides several key features:

- **Tool Definition**: Define tools using the `@sc.tool` decorator with complexity ratings
- **Server Creation**: Create MCP servers that expose tools via the protocol
- **Client Connection**: Connect to MCP servers to use their tools
- **Complexity Tracking**: Monitor the complexity of tool usage
- **Streaming Support**: Stream responses for long-running operations
- **Context Management**: Maintain conversation context across requests

## Next Steps

- Check out the [A2A Interactive Example](a2a-interactive.md) to see how agents communicate
- Learn more about the [MCP Protocol](../protocols/mcp.md)
- Explore the [MCP API Reference](../api/mcp.md)
