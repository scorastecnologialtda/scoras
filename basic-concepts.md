# Basic Concepts

This page explains the fundamental concepts and components of the Scoras library.

## Core Components

### Agents

In Scoras, an `Agent` is the primary interface for interacting with language models. Agents can:

- Process natural language inputs
- Generate responses
- Use tools to perform actions
- Track complexity of operations

Agents are configured with a model provider, system prompt, and optional tools:

```python
import scoras as sc

agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
    tools=[my_tool1, my_tool2]
)
```

### Tools

Tools extend an agent's capabilities by allowing it to perform specific actions. A tool is a function with:

- A name and description
- Input parameters with type hints
- A return type
- A complexity rating

Tools are defined using the `@sc.tool` decorator:

```python
@sc.tool(name="weather", description="Get weather information", complexity="standard")
async def get_weather(location: str) -> dict:
    """Get weather information for a location."""
    # Implementation...
    return {"temperature": 72, "conditions": "sunny"}
```

### Workflows

Workflows in Scoras are represented as graphs, with:

- Nodes: Processing steps
- Edges: Connections between steps
- Conditions: Decision points that determine flow

Workflows are created using the `WorkflowGraph` class:

```python
from pydantic import BaseModel

class WorkflowState(BaseModel):
    query: str
    result: str = ""

graph = sc.WorkflowGraph(state_type=WorkflowState)
graph.add_node("process", process_function, "standard")
graph.add_edge("start", "process")
```

### RAG Systems

Retrieval-Augmented Generation (RAG) systems enhance agents with document knowledge:

- Documents: Text content with metadata
- Retrievers: Components that find relevant documents
- RAG Agents: Agents that use retrieved documents to generate responses

```python
from scoras.rag import Document, SimpleRAG

documents = [Document(content="Example content")]
rag = SimpleRAG(agent=my_agent, documents=documents)
```

## Complexity Scoring

Scoras tracks the complexity of agent workflows through a scoring system:

### Score Components

- **Nodes**: 1-1.5 points each
- **Edges**: 1.5-4 points each
- **Tools**: 1.4-3 points each
- **Conditions**: 2.5 points each

### Complexity Ratings

- **Simple**: Score < 10
- **Moderate**: Score 10-25
- **Complex**: Score 25-50
- **Very Complex**: Score 50-100
- **Extremely Complex**: Score > 100

### Tracking Complexity

Complexity tracking is enabled by default but can be explicitly controlled:

```python
agent = sc.Agent(
    model="openai:gpt-4o",
    enable_scoring=True
)

# Later, get the complexity score
score = agent.get_complexity_score()
```

## Protocol Support

Scoras supports two key protocols for agent interoperability:

### MCP (Model Context Protocol)

The Model Context Protocol allows agents to interact with MCP servers and act as MCP servers themselves:

```python
from scoras.mcp import create_mcp_server, MCPClient

server = create_mcp_server(name="MyServer", tools=[my_tool])
client = MCPClient(server_url="http://localhost:8000")
```

### A2A (Agent-to-Agent) Protocol

The Agent-to-Agent protocol enables communication between agents across different frameworks:

```python
from scoras.a2a import create_agent_skill, create_a2a_server

skill = create_agent_skill(id="math", name="Mathematics")
server = create_a2a_server(name="MyAgent", agent=my_agent, skills=[skill])
```

## Next Steps

Now that you understand the basic concepts, you can:

- Explore [Agents](../core-features/agents.md) in more detail
- Learn about [Tools](../core-features/tools.md)
- Understand [Workflows](../core-features/workflows.md)
- Dive into [RAG Systems](../core-features/rag.md)
- Explore [Protocol Support](../protocols/mcp.md)
