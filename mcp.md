# MCP Protocol

The Model Context Protocol (MCP) is a standardized protocol for interaction between language models and tools. Scoras provides comprehensive support for MCP, allowing your agents to both consume MCP services and act as MCP servers themselves.

## What is MCP?

MCP (Model Context Protocol) is an open protocol that standardizes how language models interact with tools and external services. It enables:

- **Standardized Tool Execution**: Common interface for tool definitions and execution
- **Interoperability**: Tools can be shared across different model providers
- **Streaming Support**: Real-time interaction between models and tools
- **Context Management**: Efficient handling of context between interactions

## Scoras MCP Integration

Scoras provides a complete implementation of MCP with these key components:

- **MCP Servers**: Create servers that expose Scoras tools via MCP
- **MCP Clients**: Connect to MCP servers to use their tools
- **MCP Agent Adapters**: Adapt existing Scoras agents to use MCP
- **Complexity Scoring**: Track complexity across MCP interactions

## Creating an MCP Server

You can create an MCP server to expose your Scoras tools:

```python
import scoras as sc
from scoras.mcp import create_mcp_server, run_mcp_server

# Define tools
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

# Create an MCP server
server = create_mcp_server(
    name="ScorasServer",
    description="Scoras MCP server with tools",
    tools=[calculator],
    capabilities=["tools", "streaming"],
    enable_scoring=True
)

# Run the server
await run_mcp_server(server, host="0.0.0.0", port=8000)
```

## Using an MCP Client

Connect to MCP servers to use their tools:

```python
from scoras.mcp import MCPClient

# Create an MCP client
client = MCPClient(
    server_url="http://localhost:8000",
    enable_scoring=True
)

# Execute a tool on the server
result = await client.execute_tool(
    tool_name="calculator",
    parameters={
        "operation": "multiply",
        "a": 5,
        "b": 7
    }
)
print(result)  # {"result": 35}

# Get the complexity score
score = client.get_complexity_score()
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")
```

## MCP Agent Adapter

Adapt existing Scoras agents to use MCP tools:

```python
import scoras as sc
from scoras.mcp import MCPAgentAdapter

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

# Connect to an MCP server
adapter.connect_to_server("http://localhost:8000")

# Now the agent can use tools from the MCP server
response = await adapter.run("Calculate 5 multiplied by 7")
print(response)

# Get the complexity score
score = adapter.get_complexity_score()
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")
```

## MCP Context Management

MCP provides efficient context management:

```python
from scoras.mcp import MCPContext

# Create a context
context = MCPContext()

# Add messages to the context
context.add_user_message("What's the weather in New York?")
context.add_assistant_message("I'll check the weather for you.")

# Use the context with a tool
result = await client.execute_tool_with_context(
    context=context,
    tool_name="weather",
    parameters={"location": "New York"}
)

# Add the tool result to the context
context.add_tool_result(result)
```

## Complexity Scoring with MCP

Scoras tracks complexity across MCP interactions:

```python
# Create an MCP server with scoring enabled
server = create_mcp_server(
    name="ScorasServer",
    tools=[calculator],
    enable_scoring=True
)

# Execute tools
await server.execute_tool("calculator", {"operation": "add", "a": 1, "b": 2})
await server.execute_tool("calculator", {"operation": "multiply", "a": 3, "b": 4})

# Get detailed complexity report
score_report = server.get_complexity_score()
print(json.dumps(score_report, indent=2))
```

## MCP Specification Compliance

Scoras implements the full MCP specification, including:

- Tool definitions and execution
- Streaming responses
- Context management
- Error handling
- Authentication

For more details on the MCP specification, visit the [official MCP documentation](https://modelcontextprotocol.io/).

## Next Steps

- Check out the [MCP Examples](../examples/mcp.md) for more detailed usage
- Learn about [A2A Protocol](a2a.md) for agent-to-agent communication
- Explore the [MCP API Reference](../api/mcp.md) for detailed documentation
