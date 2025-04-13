# Scoras

**Intelligent Agent Framework with Complexity Scoring**

Scoras is a powerful, intuitive framework for building intelligent agents with built-in complexity scoring. Inspired by PydanticAI and Langgraph but designed to be more accessible and comprehensive, Scoras provides everything you need to create sophisticated AI agents, RAG systems, and multi-agent workflows.

## Key Features

- **Integrated Complexity Scoring**: Automatically measure and understand the complexity of your agent workflows
- **Multi-Model Support**: Work with OpenAI, Anthropic, Google Gemini, and other LLM providers
- **Protocol Support**: Native integration with MCP (Model Context Protocol) and A2A (Agent-to-Agent) protocols
- **Intuitive API**: Simple, expressive interface for creating agents and tools
- **Advanced Graph-Based Workflows**: Create sophisticated agent workflows with conditional branching
- **Enhanced RAG Capabilities**: Build powerful retrieval-augmented generation systems
- **Structured Data Validation**: Leverages Pydantic-style validation for robust data handling
- **Comprehensive Tooling**: Extensive tool framework for agent capabilities

## Quick Example

```python
import scoras as sc

# Create a simple agent
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant."
)

# Run the agent
response = agent.run_sync("What is the capital of France?")
print(response)

# Check the complexity score
score = agent.get_complexity_score()
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")
```

## Understanding Complexity Scores

Scoras provides a unique complexity scoring system that helps you understand and manage the complexity of your agent workflows:

- **Nodes**: Basic processing units (1-1.5 points each)
- **Edges**: Connections between nodes (1.5-4 points each)
- **Tools**: Agent capabilities (1.4-3 points each)
- **Conditions**: Decision points (2.5 points each)

Complexity ratings:
- **Simple**: Score < 10
- **Moderate**: Score 10-25
- **Complex**: Score 25-50
- **Very Complex**: Score 50-100
- **Extremely Complex**: Score > 100

## License

Scoras is created by Anderson L. Amaral and is available under the MIT License.
