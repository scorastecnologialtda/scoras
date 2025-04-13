# Workflows

Workflows in Scoras allow you to create sophisticated agent processes with conditional branching, parallel execution, and state management. This page explains how to design, implement, and optimize workflows.

## What are Scoras Workflows?

In Scoras, a workflow is a directed graph that represents a sequence of operations, where:

- **Nodes**: Processing steps that transform state
- **Edges**: Connections between nodes that define flow
- **Conditions**: Decision points that determine which path to take
- **State**: Data that flows through the workflow

## Creating a Basic Workflow

The simplest workflow connects a few processing steps:

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
    answer = f"Based on {len(state.search_results)} results, the answer is..."
    return {"final_answer": answer}

# Create a workflow graph
graph = sc.WorkflowGraph(state_type=WorkflowState)

# Add nodes with complexity ratings
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

## Conditional Branching

Add decision points to your workflow:

```python
# Define a condition function
def needs_search(state):
    """Determine if search is needed based on the query."""
    return "search" in state.processed_query or "find" in state.processed_query

# Add conditional edges
graph.add_edge("preprocess", "search", condition=needs_search)
graph.add_edge("preprocess", "generate", condition=lambda s: not needs_search(s))
```

## Parallel Execution

Execute multiple nodes in parallel:

```python
# Define parallel node functions
async def web_search(state):
    return {"web_results": ["Web result 1", "Web result 2"]}

async def database_lookup(state):
    return {"db_results": ["DB result 1", "DB result 2"]}

# Add parallel nodes
graph.add_node("web_search", web_search, "standard")
graph.add_node("db_lookup", database_lookup, "standard")

# Add edges for parallel execution
graph.add_edge("preprocess", "web_search")
graph.add_edge("preprocess", "db_lookup")

# Add join node
async def combine_results(state):
    all_results = state.web_results + state.db_results
    return {"search_results": all_results}

graph.add_node("combine", combine_results, "standard")
graph.add_edge("web_search", "combine")
graph.add_edge("db_lookup", "combine")
graph.add_edge("combine", "generate")
```

## Error Handling

Add error handling to your workflow:

```python
# Define error handling node
async def handle_error(state, error):
    """Handle errors in the workflow."""
    return {"final_answer": f"An error occurred: {str(error)}"}

# Add error handling
graph.add_error_handler("search", handle_error)
```

## Workflow Monitoring

Monitor the execution of your workflow:

```python
# Define a monitor function
def monitor(node_name, state, result):
    """Monitor workflow execution."""
    print(f"Executed node: {node_name}")
    print(f"Input state: {state}")
    print(f"Output: {result}")

# Add monitoring
graph.add_monitor(monitor)
```

## Complexity Scoring

Track and understand the complexity of your workflow:

```python
# Create a workflow graph with scoring enabled
graph = sc.WorkflowGraph(
    state_type=WorkflowState,
    enable_scoring=True
)

# Add nodes and edges
# ...

# Get the complexity score
score = graph.get_complexity_score()
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")

# Get detailed breakdown
print(json.dumps(score, indent=2))
```

## Integrating Agents in Workflows

Use agents within workflow nodes:

```python
import scoras as sc

# Create an agent
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant."
)

# Define a node function that uses the agent
async def agent_node(state):
    """Use an agent to process the query."""
    response = await agent.run(f"Answer this question: {state.query}")
    return {"final_answer": response}

# Add the node to the workflow
graph.add_node("agent_process", agent_node, "complex")
```

## Workflow Templates

Scoras provides templates for common workflow patterns:

```python
from scoras.workflows import RAGWorkflow, AgentChainWorkflow

# Create a RAG workflow
rag_workflow = RAGWorkflow(
    agent=sc.Agent(model="openai:gpt-4o"),
    documents=[Document(content="Example content")],
    retriever_type="semantic"
)

# Create an agent chain workflow
chain_workflow = AgentChainWorkflow(
    agents=[
        sc.Agent(model="openai:gpt-4o", system_prompt="You research information."),
        sc.Agent(model="anthropic:claude-3-opus", system_prompt="You write content."),
        sc.Agent(model="gemini:gemini-pro", system_prompt="You edit and improve content.")
    ]
)
```

## Protocol Integration

Workflows can be exposed via MCP and A2A protocols:

```python
from scoras.mcp import create_mcp_server
from scoras.a2a import create_a2a_server

# Create an MCP server with a workflow
mcp_server = create_mcp_server(
    name="WorkflowServer",
    workflow=my_workflow
)

# Create an A2A server with a workflow
a2a_server = create_a2a_server(
    name="WorkflowAgent",
    workflow=my_workflow
)
```

## Next Steps

- Learn about [Agents](agents.md) that can be used in workflows
- Explore [Tools](tools.md) for extending workflow capabilities
- Understand [RAG Systems](rag.md) for knowledge-enhanced workflows
- Dive into [Complexity Scoring](complexity-scoring.md) for workflow optimization
