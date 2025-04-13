# Quick Start

This guide will help you get up and running with Scoras quickly. We'll cover the basics of creating agents, using tools, and understanding complexity scores.

## Creating Your First Agent

Let's start by creating a simple agent:

```python
import scoras as sc

# Create a basic agent
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant."
)

# Run the agent
response = agent.run_sync("What is the capital of France?")
print(response)
```

This creates an agent using OpenAI's GPT-4o model and runs it with a simple query.

## Adding Tools to Your Agent

Agents become more powerful when they have tools to work with:

```python
import scoras as sc

# Define a tool using the decorator
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

# Create an agent with the tool
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant with calculation abilities.",
    tools=[calculator]
)

# Run the agent with a query that will use the tool
response = agent.run_sync("What is 25 multiplied by 16?")
print(response)
```

## Building a Simple RAG System

Retrieval-Augmented Generation (RAG) enhances your agent with document knowledge:

```python
import scoras as sc
from scoras.rag import Document, SimpleRAG

# Create documents
documents = [
    Document(content="The capital of France is Paris, known as the City of Light."),
    Document(content="Paris is famous for the Eiffel Tower, built in 1889."),
    Document(content="France has a population of about 67 million people.")
]

# Create a RAG system
rag = SimpleRAG(
    agent=sc.Agent(model="openai:gpt-4o"),
    documents=documents
)

# Run the RAG system
response = rag.run_sync("What is the capital of France and what is it known for?")
print(response)
```

## Understanding Complexity Scores

One of Scoras' unique features is its ability to track and report on the complexity of your agent workflows:

```python
# Create an agent with tools
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
    tools=[calculator],
    enable_scoring=True
)

# Run the agent
response = agent.run_sync("What is 42 divided by 6?")
print(response)

# Get the complexity score
score = agent.get_complexity_score()
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")
print(f"Component breakdown: {score['breakdown']}")
```

The complexity score helps you understand how sophisticated your agent workflow is, which can be useful for optimization and resource planning.

## Next Steps

Now that you've seen the basics, you can:

- Learn about [Basic Concepts](basic-concepts.md) in Scoras
- Explore [Core Features](../core-features/agents.md) in more detail
- Check out [Protocol Support](../protocols/mcp.md) for MCP and A2A
- Try some [Examples](../examples/basic.md) to see Scoras in action
