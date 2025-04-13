"""
Example script demonstrating MCP (Model Context Protocol) support in Scoras.

This example shows how to:
1. Create an MCP server with Scoras tools
2. Connect to the server with an MCP client
3. Execute tools remotely
4. Track complexity scores throughout the process

Author: Anderson L. Amaral
"""

import asyncio
import json
import os
from typing import Dict, Any, List

import scoras as sc
from scoras.mcp import (
    create_mcp_server,
    MCPClient,
    MCPContext,
    MCPAgentAdapter,
    run_mcp_server
)

# Define some tools for our MCP server
@sc.tool(name="calculator", description="Perform basic arithmetic operations", complexity="simple")
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

@sc.tool(name="weather", description="Get weather information for a location", complexity="standard")
async def weather(location: str) -> Dict[str, Any]:
    """
    Get weather information for a location.
    
    Args:
        location: Location to get weather for
        
    Returns:
        Weather information
    """
    # In a real implementation, this would call a weather API
    # For this example, we'll return mock data
    return {
        "location": location,
        "temperature": 72,
        "conditions": "Sunny",
        "humidity": 45,
        "wind_speed": 5
    }

@sc.tool(name="search", description="Search for information", complexity="complex")
async def search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for information.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        Search results
    """
    # In a real implementation, this would call a search API
    # For this example, we'll return mock data
    return [
        {
            "title": f"Result {i} for {query}",
            "snippet": f"This is a snippet for result {i} about {query}.",
            "url": f"https://example.com/result/{i}"
        }
        for i in range(1, min(max_results + 1, 6))
    ]

async def run_mcp_server_example():
    """Run the MCP server example."""
    print("=== MCP Server Example ===")
    
    # Create an agent with our tools
    agent = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant with access to tools.",
        tools=[calculator, weather, search],
        enable_scoring=True
    )
    
    # Create an MCP server
    server = create_mcp_server(
        name="ScorasServer",
        description="Scoras MCP server with various tools",
        tools=[calculator, weather, search],
        capabilities=["tools", "streaming"],
        enable_scoring=True
    )
    
    # Print the server info
    print("Server Info:", server.get_server_info())
    
    # Print available tools
    print("Available Tools:", json.dumps(server.get_available_tools(), indent=2))
    
    # Execute a tool directly on the server
    print("\nExecuting calculator tool directly on server...")
    result = await server.execute_tool(
        tool_name="calculator",
        parameters={
            "operation": "multiply",
            "a": 5,
            "b": 7
        }
    )
    print("Result:", result)
    
    # Get the complexity score
    score = server.get_complexity_score()
    print("\nServer Complexity Score:", json.dumps(score, indent=2))
    
    # In a real application, you would run the server with:
    # await run_mcp_server(server, host="0.0.0.0", port=8000)
    # For this example, we'll just print a message
    print("\nIn a real application, the server would be running at http://0.0.0.0:8000")

async def run_mcp_client_example():
    """Run the MCP client example."""
    print("\n=== MCP Client Example ===")
    
    # Create an MCP client
    # In a real application, this would connect to a running server
    # For this example, we'll simulate the connection
    client = MCPClient(
        server_url="http://localhost:8000",
        enable_scoring=True
    )
    
    # Simulate getting server info
    print("Connecting to MCP server...")
    
    # Simulate executing a tool
    print("\nExecuting calculator tool via client...")
    try:
        # In a real application, this would call the server
        # For this example, we'll simulate the response
        result = {
            "result": 35
        }
        print("Result:", result)
    except Exception as e:
        print(f"Error: {e}")
    
    # Simulate executing another tool
    print("\nExecuting weather tool via client...")
    try:
        # In a real application, this would call the server
        # For this example, we'll simulate the response
        result = {
            "result": {
                "location": "New York",
                "temperature": 72,
                "conditions": "Sunny",
                "humidity": 45,
                "wind_speed": 5
            }
        }
        print("Result:", result)
    except Exception as e:
        print(f"Error: {e}")
    
    # Get the complexity score
    score = client.get_complexity_score()
    print("\nClient Complexity Score:", json.dumps(score, indent=2))

async def run_mcp_agent_adapter_example():
    """Run the MCP agent adapter example."""
    print("\n=== MCP Agent Adapter Example ===")
    
    # Create an agent
    agent = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant.",
        enable_scoring=True
    )
    
    # Create an MCP agent adapter
    adapter = MCPAgentAdapter(
        agent=agent,
        enable_scoring=True
    )
    
    # Simulate connecting to an MCP server
    print("Connecting to MCP server...")
    try:
        # In a real application, this would connect to a running server
        # For this example, we'll simulate the connection
        server_id = "example_server"
        print(f"Connected to server with ID: {server_id}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Simulate adding MCP tools to the agent
    print("\nAdding MCP tools to agent...")
    try:
        # In a real application, this would get tools from the server
        # For this example, we'll simulate adding tools
        print("Added 3 tools from the MCP server to the agent")
    except Exception as e:
        print(f"Error: {e}")
    
    # Simulate executing a tool
    print("\nExecuting calculator tool via adapter...")
    try:
        # In a real application, this would call the server
        # For this example, we'll simulate the response
        result = {
            "result": 35
        }
        print("Result:", result)
    except Exception as e:
        print(f"Error: {e}")
    
    # Get the complexity score
    score = adapter.get_complexity_score()
    print("\nAdapter Complexity Score:", json.dumps(score, indent=2))

async def main():
    """Run all examples."""
    await run_mcp_server_example()
    await run_mcp_client_example()
    await run_mcp_agent_adapter_example()

if __name__ == "__main__":
    asyncio.run(main())
