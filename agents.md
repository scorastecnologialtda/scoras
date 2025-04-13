# Agents

Agents are the core building blocks of the Scoras framework. This page explains how to create, configure, and use agents effectively.

## What is a Scoras Agent?

A Scoras Agent is an intelligent entity that can:

- Process natural language inputs
- Generate coherent responses
- Use tools to perform actions
- Track the complexity of operations
- Integrate with protocols like MCP and A2A

## Creating a Basic Agent

Creating an agent is straightforward:

```python
import scoras as sc

# Create a basic agent
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant."
)

# Run the agent synchronously
response = agent.run_sync("What is the capital of France?")
print(response)

# Or run it asynchronously
async def main():
    response = await agent.run("Tell me about quantum computing.")
    print(response)
```

## Supported Models

Scoras supports multiple model providers:

```python
# OpenAI
openai_agent = sc.Agent(model="openai:gpt-4o")

# Anthropic
anthropic_agent = sc.Agent(model="anthropic:claude-3-opus")

# Google Gemini
gemini_agent = sc.Agent(model="gemini:gemini-pro")

# Custom model provider
custom_agent = sc.Agent(
    model="custom",
    model_provider=MyCustomProvider()
)
```

## Agent Configuration

Agents can be configured with various options:

```python
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant specialized in science.",
    temperature=0.7,
    max_tokens=1000,
    tools=[my_tool1, my_tool2],
    enable_scoring=True,
    metadata={
        "created_by": "example_user",
        "purpose": "science_assistant"
    }
)
```

## Agent Methods

Agents provide several methods for interaction:

```python
# Basic interaction
response = agent.run_sync("What is the capital of France?")

# With conversation history
agent.add_message("user", "What is the capital of France?")
agent.add_message("assistant", "The capital of France is Paris.")
response = agent.run_sync("What is the population of this city?")

# Stream responses
async for chunk in agent.stream("Tell me a story about a robot."):
    print(chunk, end="", flush=True)

# Reset conversation
agent.reset_conversation()
```

## Adding Tools to Agents

Tools extend an agent's capabilities:

```python
import scoras as sc

# Define tools
@sc.tool(name="calculator", description="Perform calculations", complexity="simple")
async def calculator(operation: str, a: float, b: float) -> float:
    if operation == "add":
        return a + b
    # ... other operations

@sc.tool(name="weather", description="Get weather information", complexity="standard")
async def get_weather(location: str) -> dict:
    # Implementation...
    return {"temperature": 72, "conditions": "sunny"}

# Create an agent with tools
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant with various capabilities.",
    tools=[calculator, get_weather]
)

# Add a tool after creation
@sc.tool(name="search", description="Search for information", complexity="complex")
async def search(query: str, max_results: int = 5) -> list:
    # Implementation...
    return [{"title": "Result 1", "snippet": "Information..."}]

agent.add_tool(search)
```

## Complexity Scoring

Track and understand the complexity of your agent operations:

```python
# Create an agent with scoring enabled
agent = sc.Agent(
    model="openai:gpt-4o",
    enable_scoring=True
)

# Run the agent
response = agent.run_sync("What is the capital of France?")

# Get the complexity score
score = agent.get_complexity_score()
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")

# Get detailed breakdown
print(json.dumps(score, indent=2))
```

## Specialized Agents

Scoras provides specialized agent types for specific use cases:

```python
from scoras.agents import ExpertAgent, CreativeAgent, RAGAgent

# Expert agent for specialized domains
expert = ExpertAgent(
    model="anthropic:claude-3-opus",
    domain="medicine",
    expertise_level="advanced"
)

# Creative agent for content generation
creative = CreativeAgent(
    model="openai:gpt-4o",
    creativity_level="high"
)

# RAG agent with built-in retrieval
from scoras.rag import Document
documents = [Document(content="Example content")]
rag_agent = RAGAgent(
    model="gemini:gemini-pro",
    documents=documents,
    retrieval_type="semantic"
)
```

## Multi-Agent Systems

Combine multiple agents into collaborative systems:

```python
from scoras.agents import MultiAgentSystem

# Create individual agents
researcher = sc.Agent(model="openai:gpt-4o", system_prompt="You are a research specialist.")
writer = sc.Agent(model="anthropic:claude-3-opus", system_prompt="You are a writing expert.")
fact_checker = sc.Agent(model="gemini:gemini-pro", system_prompt="You verify facts.")

# Create a multi-agent system
system = MultiAgentSystem(
    agents={
        "researcher": researcher,
        "writer": writer,
        "fact_checker": fact_checker
    },
    coordinator_prompt="Coordinate the research, writing, and fact-checking process."
)

# Run the system
result = system.run_sync("Create a well-researched article about quantum computing.")
```

## Protocol Integration

Agents can be integrated with MCP and A2A protocols:

```python
from scoras.mcp import MCPAgentAdapter
from scoras.a2a import A2AAgentAdapter

# Create an agent
agent = sc.Agent(model="openai:gpt-4o")

# Adapt for MCP
mcp_adapter = MCPAgentAdapter(agent=agent)
mcp_adapter.connect_to_server("http://localhost:8000")

# Adapt for A2A
a2a_adapter = A2AAgentAdapter(agent=agent)
a2a_adapter.connect_to_agent("http://localhost:8001")
```

## Next Steps

- Learn about [Tools](tools.md) to extend agent capabilities
- Explore [Workflows](workflows.md) for complex agent processes
- Understand [RAG Systems](rag.md) for knowledge-enhanced agents
- Dive into [Complexity Scoring](complexity-scoring.md) for performance insights
