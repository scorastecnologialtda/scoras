# Basic Examples

This page provides basic examples of using the Scoras library. These examples demonstrate the fundamental features of Scoras, including creating agents, using tools, building RAG systems, and tracking complexity scores.

## Creating a Simple Agent

This example shows how to create and use a basic agent:

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
# Output: "The capital of France is Paris."

# Run the agent asynchronously
import asyncio

async def main():
    response = await agent.run("Tell me about quantum computing.")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## Using Different Model Providers

Scoras supports multiple model providers:

```python
import scoras as sc

# OpenAI
openai_agent = sc.Agent(model="openai:gpt-4o")
response = openai_agent.run_sync("What is the capital of France?")
print(f"OpenAI: {response}")

# Anthropic
anthropic_agent = sc.Agent(model="anthropic:claude-3-opus")
response = anthropic_agent.run_sync("What is the capital of France?")
print(f"Anthropic: {response}")

# Google Gemini
gemini_agent = sc.Agent(model="gemini:gemini-pro")
response = gemini_agent.run_sync("What is the capital of France?")
print(f"Gemini: {response}")
```

## Creating and Using Tools

This example demonstrates how to create and use tools with an agent:

```python
import scoras as sc

# Define a tool using the decorator
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

# Create an agent with the tool
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant with calculation abilities.",
    tools=[calculator]
)

# Run the agent with a query that will use the tool
response = agent.run_sync("What is 25 multiplied by 16?")
print(response)
# Output: "25 multiplied by 16 equals 400."

# Add another tool after creation
@sc.tool(name="weather", description="Get weather information", complexity="standard")
async def get_weather(location: str) -> dict:
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

agent.add_tool(get_weather)

# Run the agent with a query that will use the new tool
response = agent.run_sync("What's the weather like in New York?")
print(response)
```

## Building a Simple RAG System

This example shows how to create a basic RAG (Retrieval-Augmented Generation) system:

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
# Output: "The capital of France is Paris, known as the City of Light. Paris is famous for the Eiffel Tower, which was built in 1889."

# Add more documents
rag.add_documents([
    Document(content="The Louvre Museum in Paris is the world's largest art museum."),
    Document(content="Paris hosts the annual Tour de France cycling race finish.")
])

# Run with the expanded knowledge
response = rag.run_sync("What museums are in Paris?")
print(response)
```

## Creating a Simple Workflow

This example demonstrates creating a basic workflow:

```python
import scoras as sc
from pydantic import BaseModel

# Define the state model
class WorkflowState(BaseModel):
    query: str
    processed_query: str = ""
    search_results: list = []
    final_answer: str = ""

# Define node functions
async def preprocess(state):
    """Clean and prepare the query."""
    return {"processed_query": state.query.strip().lower()}

async def search(state):
    """Search for information based on the processed query."""
    # Simulate search
    results = [f"Result for {state.processed_query}", "Another result"]
    return {"search_results": results}

async def generate_answer(state):
    """Generate a final answer based on search results."""
    answer = f"Based on {len(state.search_results)} results, the answer is: This is information about {state.processed_query}."
    return {"final_answer": answer}

# Create a workflow graph
graph = sc.WorkflowGraph(state_type=WorkflowState)

# Add nodes
graph.add_node("start", lambda s: s, "simple")
graph.add_node("preprocess", preprocess, "standard")
graph.add_node("search", search, "standard")
graph.add_node("generate", generate_answer, "complex")
graph.add_node("end", lambda s: s, "simple")

# Add edges
graph.add_edge("start", "preprocess")
graph.add_edge("preprocess", "search")
graph.add_edge("search", "generate")
graph.add_edge("generate", "end")

# Compile the graph
workflow = graph.compile()

# Run the workflow
result = workflow.run_sync(WorkflowState(query="What is quantum computing?"))
print(result.final_answer)
```

## Tracking Complexity Scores

This example shows how to track and analyze complexity scores:

```python
import scoras as sc
import json

# Create an agent with scoring enabled
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
    enable_scoring=True
)

# Run the agent
response = agent.run_sync("What is the capital of France?")
print(response)

# Get the complexity score
score = agent.get_complexity_score()
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")

# Add tools to increase complexity
@sc.tool(name="calculator", complexity="simple")
async def calculator(operation: str, a: float, b: float) -> float:
    # Implementation...
    return 0

@sc.tool(name="weather", complexity="standard")
async def get_weather(location: str) -> dict:
    # Implementation...
    return {}

agent.add_tool(calculator)
agent.add_tool(get_weather)

# Get updated complexity score
updated_score = agent.get_complexity_score()
print(f"Updated complexity: {updated_score['complexity_rating']} (Score: {updated_score['total_score']})")

# Get detailed breakdown
print("Detailed score breakdown:")
print(json.dumps(updated_score, indent=2))
```

## Conversation Management

This example demonstrates managing a conversation with an agent:

```python
import scoras as sc

# Create an agent
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant."
)

# Start a conversation
agent.add_message("user", "Hello, who are you?")
response = agent.run_sync()
print(f"Assistant: {response}")

# Continue the conversation
agent.add_message("user", "What can you help me with?")
response = agent.run_sync()
print(f"Assistant: {response}")

# Ask about previous context
agent.add_message("user", "Can you remember what I asked you first?")
response = agent.run_sync()
print(f"Assistant: {response}")

# Reset the conversation
agent.reset_conversation()
agent.add_message("user", "Do you remember our previous conversation?")
response = agent.run_sync()
print(f"Assistant: {response}")
```

## Streaming Responses

This example shows how to stream responses from an agent:

```python
import scoras as sc
import asyncio

# Create an agent
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant."
)

# Stream a response
async def stream_example():
    print("Streaming response:")
    async for chunk in agent.stream("Tell me a short story about a robot."):
        print(chunk, end="", flush=True)
    print("\nStreaming complete!")

if __name__ == "__main__":
    asyncio.run(stream_example())
```

## Next Steps

- Check out [MCP Examples](mcp.md) for Model Context Protocol usage
- Explore [A2A Examples](a2a.md) for Agent-to-Agent communication
- Try [Advanced Examples](advanced.md) for more sophisticated use cases
