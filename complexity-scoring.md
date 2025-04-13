# Complexity Scoring

One of Scoras' most distinctive features is its integrated complexity scoring system. This page explains how complexity scoring works and how to use it to understand and optimize your agent workflows.

## What is Complexity Scoring?

Complexity scoring in Scoras is a system that:

- Measures the computational and conceptual complexity of agent workflows
- Assigns points to different components based on their complexity
- Provides an overall complexity rating for the entire system
- Helps identify optimization opportunities and resource requirements

## Score Components

The complexity score is calculated based on four main components:

### Nodes (1-1.5 points each)

Nodes are basic processing units in workflows:

- **Simple Node**: 1 point
- **Standard Node**: 1.2 points
- **Complex Node**: 1.5 points

### Edges (1.5-4 points each)

Edges connect nodes and define the flow of execution:

- **Simple Edge**: 1.5 points
- **Conditional Edge**: 4 points (includes a condition)

### Tools (1.4-3 points each)

Tools extend agent capabilities:

- **Simple Tool**: 1.4 points
- **Standard Tool**: 2 points
- **Complex Tool**: 3 points

### Conditions (2.5 points each)

Conditions are decision points that determine execution paths:

- **Each Condition**: 2.5 points

## Complexity Ratings

Based on the total score, Scoras assigns a complexity rating:

- **Simple**: Score < 10
- **Moderate**: Score 10-25
- **Complex**: Score 25-50
- **Very Complex**: Score 50-100
- **Extremely Complex**: Score > 100

## Enabling Complexity Scoring

Complexity scoring is enabled by default but can be explicitly controlled:

```python
import scoras as sc

# Enable scoring for an agent
agent = sc.Agent(
    model="openai:gpt-4o",
    enable_scoring=True
)

# Enable scoring for a workflow
graph = sc.WorkflowGraph(
    state_type=MyState,
    enable_scoring=True
)

# Enable scoring for a RAG system
rag = sc.rag.SimpleRAG(
    agent=agent,
    documents=documents,
    enable_scoring=True
)

# Enable scoring for protocol adapters
mcp_adapter = sc.mcp.MCPAgentAdapter(
    agent=agent,
    enable_scoring=True
)
```

## Getting Complexity Scores

Retrieve complexity scores from any component:

```python
# Get agent complexity score
agent_score = agent.get_complexity_score()
print(f"Agent complexity: {agent_score['complexity_rating']} (Score: {agent_score['total_score']})")

# Get workflow complexity score
workflow_score = graph.get_complexity_score()
print(f"Workflow complexity: {workflow_score['complexity_rating']} (Score: {workflow_score['total_score']})")

# Get RAG system complexity score
rag_score = rag.get_complexity_score()
print(f"RAG complexity: {rag_score['complexity_rating']} (Score: {rag_score['total_score']})")

# Get protocol adapter complexity score
adapter_score = mcp_adapter.get_complexity_score()
print(f"Adapter complexity: {adapter_score['complexity_rating']} (Score: {adapter_score['total_score']})")
```

## Detailed Score Reports

Get detailed breakdowns of complexity scores:

```python
import json

# Get detailed score report
score_report = agent.get_complexity_score()
print(json.dumps(score_report, indent=2))

# Example output:
# {
#   "total_score": 15.5,
#   "complexity_rating": "Moderate",
#   "component_scores": {
#     "nodes": 3.0,
#     "edges": 4.5,
#     "tools": 8.0,
#     "conditions": 0.0
#   },
#   "component_counts": {
#     "nodes": 3,
#     "edges": 3,
#     "tools": 4,
#     "conditions": 0
#   },
#   "breakdown": {
#     "nodes": "3 nodes (3.0 points)",
#     "edges": "3 edges (4.5 points)",
#     "tools": "4 tools (8.0 points)",
#     "conditions": "0 conditions (0.0 points)"
#   }
# }
```

## Tracking Complexity Over Time

Monitor how complexity changes as your system evolves:

```python
import scoras as sc

# Create an agent with scoring
agent = sc.Agent(
    model="openai:gpt-4o",
    enable_scoring=True
)

# Get initial score
initial_score = agent.get_complexity_score()
print(f"Initial complexity: {initial_score['complexity_rating']} (Score: {initial_score['total_score']})")

# Add tools
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

# Get updated score
updated_score = agent.get_complexity_score()
print(f"Updated complexity: {updated_score['complexity_rating']} (Score: {updated_score['total_score']})")
```

## Complexity Optimization

Use complexity scores to optimize your systems:

```python
import scoras as sc

# Create a complex workflow
graph = sc.WorkflowGraph(
    state_type=MyState,
    enable_scoring=True
)

# Add many nodes and edges
# ...

# Check complexity
score = graph.get_complexity_score()
print(f"Initial complexity: {score['complexity_rating']} (Score: {score['total_score']})")

# Identify high-complexity components
if score["component_scores"]["conditions"] > 10:
    print("High condition complexity - consider simplifying decision logic")
    
if score["component_scores"]["tools"] > 20:
    print("High tool complexity - consider consolidating similar tools")

# Optimize by removing unnecessary components
# ...

# Check optimized complexity
optimized_score = graph.get_complexity_score()
print(f"Optimized complexity: {optimized_score['complexity_rating']} (Score: {optimized_score['total_score']})")
```

## Complexity Comparison

Compare complexity across different implementations:

```python
# Create two different implementations
implementation1 = sc.Agent(
    model="openai:gpt-4o",
    tools=[tool1, tool2, tool3],
    enable_scoring=True
)

implementation2 = sc.WorkflowGraph(
    state_type=MyState,
    enable_scoring=True
)
# Add nodes and edges
# ...

# Compare complexity
score1 = implementation1.get_complexity_score()
score2 = implementation2.get_complexity_score()

print(f"Implementation 1: {score1['complexity_rating']} (Score: {score1['total_score']})")
print(f"Implementation 2: {score2['complexity_rating']} (Score: {score2['total_score']})")

if score1["total_score"] < score2["total_score"]:
    print("Implementation 1 is less complex")
else:
    print("Implementation 2 is less complex")
```

## Protocol Integration

Complexity scoring works with protocol integrations:

```python
from scoras.mcp import create_mcp_server
from scoras.a2a import create_a2a_server

# Create servers with scoring enabled
mcp_server = create_mcp_server(
    name="ScorasServer",
    tools=[calculator, get_weather],
    enable_scoring=True
)

a2a_server = create_a2a_server(
    name="ScorasAgent",
    agent=agent,
    skills=[math_skill, research_skill],
    enable_scoring=True
)

# Get complexity scores
mcp_score = mcp_server.get_complexity_score()
a2a_score = a2a_server.get_complexity_score()

print(f"MCP Server complexity: {mcp_score['complexity_rating']} (Score: {mcp_score['total_score']})")
print(f"A2A Server complexity: {a2a_score['complexity_rating']} (Score: {a2a_score['total_score']})")
```

## Next Steps

- Learn about [Agents](agents.md) and their complexity
- Explore [Tools](tools.md) and how they affect complexity
- Understand [Workflows](workflows.md) for complex systems
- Dive into [RAG Systems](rag.md) and their complexity considerations
