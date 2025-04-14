"""
Scoras __init__.py - Main module initialization

This module initializes the Scoras library and provides access to its components.
It includes support for MCP and A2A protocols.

Author: Anderson L. Amaral
"""

# Core components
from .core import (
    Agent,
    Message,
    Tool,
    tool,
    WorkflowGraph,
    ScoreTracker,
    ScorasConfig,
    RAG
)

# Protocol support
from .mcp import (
    create_mcp_server,
    MCPClient,
    MCPContext,
    MCPAgentAdapter,
    run_mcp_server
)

from .a2a import (
    create_agent_skill,
    create_a2a_server,
    A2AClient,
    A2AAgentAdapter,
    run_a2a_server
)

__version__ = "1.0.0"
__author__ = "Anderson L. Amaral"

# Set up package-level docstring
__doc__ = """
Scoras: Intelligent Agent Framework with Complexity Scoring

Scoras is a powerful, intuitive framework for building intelligent agents with built-in 
complexity scoring. It provides everything you need to create sophisticated AI agents, 
RAG systems, and multi-agent workflows.

Key Features:
- Integrated Complexity Scoring: Automatically measure workflow complexity
- Multi-Model Support: Work with OpenAI, Anthropic, Google Gemini, and other LLMs
- Protocol Support: Native integration with MCP and A2A protocols
- Intuitive API: Simple, expressive interface for creating agents and tools
- Advanced Graph-Based Workflows: Create sophisticated agent workflows
- Enhanced RAG Capabilities: Build powerful retrieval-augmented generation systems

Basic usage:
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
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']}")
```
"""
