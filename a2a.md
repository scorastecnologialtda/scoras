# A2A Protocol

The Agent-to-Agent (A2A) protocol is an open standard for communication between AI agents. Scoras provides comprehensive support for A2A, enabling your agents to communicate with other agents regardless of their underlying implementation.

## What is A2A?

A2A (Agent-to-Agent) is an open protocol developed by Google that standardizes communication between AI agents. It enables:

- **Cross-Framework Communication**: Agents built with different frameworks can interact
- **Skill Discovery**: Agents can discover and leverage each other's capabilities
- **Task Management**: Structured approach to sending tasks between agents
- **State Tracking**: Monitoring the progress of tasks across agent boundaries

## Scoras A2A Integration

Scoras provides a complete implementation of A2A with these key components:

- **A2A Servers**: Create servers that expose Scoras agents via A2A
- **A2A Clients**: Connect to A2A agents to use their capabilities
- **A2A Agent Adapters**: Adapt existing Scoras agents to use A2A
- **Complexity Scoring**: Track complexity across A2A interactions

## Agent Skills

In A2A, agents expose their capabilities as "skills":

```python
import scoras as sc
from scoras.a2a import create_agent_skill

# Define skills for your agent
math_skill = create_agent_skill(
    id="math",
    name="Mathematics",
    description="Perform mathematical calculations and solve problems",
    tags=["math", "calculation", "problem-solving"],
    examples=[
        "Calculate the derivative of f(x) = x^2 + 3x + 2",
        "Solve the equation 2x + 5 = 13"
    ],
    complexity="standard"
)

research_skill = create_agent_skill(
    id="research",
    name="Research",
    description="Find and analyze information on various topics",
    tags=["research", "information", "analysis"],
    complexity="complex"
)
```

## Creating an A2A Server

You can create an A2A server to expose your Scoras agent:

```python
import scoras as sc
from scoras.a2a import create_a2a_server, run_a2a_server

# Create an agent
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant with expertise in mathematics and research.",
    enable_scoring=True
)

# Create an A2A server
server = create_a2a_server(
    name="ScorasAgent",
    description="A versatile agent with multiple skills powered by Scoras",
    agent=agent,
    skills=[math_skill, research_skill],
    provider={
        "organization": "Scoras Project",
        "url": "https://scoras.example.com"
    },
    capabilities={
        "streaming": True,
        "push_notifications": False,
        "state_transition_history": True
    },
    authentication_schemes=["bearer"],
    enable_scoring=True
)

# Run the server
await run_a2a_server(server, host="0.0.0.0", port=8001)
```

## Using an A2A Client

Connect to A2A agents to use their capabilities:

```python
from scoras.a2a import A2AClient

# Create an A2A client
client = A2AClient(
    agent_url="http://localhost:8001",
    enable_scoring=True
)

# Get the agent card to see available skills
agent_card = await client.get_agent_card()
print(f"Agent name: {agent_card.name}")
print(f"Available skills: {[skill.name for skill in agent_card.skills]}")

# Send a task to the agent
task = await client.send_task(
    message="Calculate the area of a circle with radius 5 cm."
)
print(f"Task ID: {task.id}")
print(f"Task state: {task.state}")

# Get the task result
task_result = await client.get_task(task.id)
print(f"Response: {task_result.messages[-1].parts[0].text}")

# Get the complexity score
score = client.get_complexity_score()
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")
```

## A2A Agent Adapter

Adapt existing Scoras agents to use A2A agents:

```python
import scoras as sc
from scoras.a2a import A2AAgentAdapter

# Create an agent
agent = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant that can coordinate with other agents.",
    enable_scoring=True
)

# Create an A2A agent adapter
adapter = A2AAgentAdapter(
    agent=agent,
    enable_scoring=True
)

# Connect to an A2A agent
adapter.connect_to_agent("http://localhost:8001")

# Now the agent can delegate tasks to the connected A2A agent
response = await adapter.run("I need to calculate the area of a circle with radius 5 cm.")
print(response)

# Get the complexity score
score = adapter.get_complexity_score()
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")
```

## Multi-Agent Systems with A2A

A2A enables building sophisticated multi-agent systems:

```python
from scoras.a2a import A2AClient

# Connect to multiple specialized agents
math_agent = A2AClient(agent_url="http://localhost:8001")
research_agent = A2AClient(agent_url="http://localhost:8002")
writing_agent = A2AClient(agent_url="http://localhost:8003")

# Coordinate between agents
research_task = await research_agent.send_task(
    message="Find information about quantum computing"
)
research_result = await research_agent.get_task(research_task.id)
research_text = research_result.messages[-1].parts[0].text

writing_task = await writing_agent.send_task(
    message=f"Write a blog post based on this research: {research_text}"
)
writing_result = await writing_agent.get_task(writing_task.id)
final_text = writing_result.messages[-1].parts[0].text

print(final_text)
```

## Complexity Scoring with A2A

Scoras tracks complexity across A2A interactions:

```python
# Create an A2A server with scoring enabled
server = create_a2a_server(
    name="ScorasAgent",
    agent=agent,
    skills=[math_skill, research_skill],
    enable_scoring=True
)

# Handle tasks
await server.handle_task(task_id, "Calculate 5 * 7")
await server.handle_task(task_id, "Research quantum computing")

# Get detailed complexity report
score_report = server.get_complexity_score()
print(json.dumps(score_report, indent=2))
```

## A2A Specification Compliance

Scoras implements the full A2A specification, including:

- Agent cards and skill discovery
- Task management
- Message handling
- State transitions
- Authentication

For more details on the A2A specification, visit the [official A2A documentation](https://google.github.io/A2A/).

## Next Steps

- Check out the [A2A Examples](../examples/a2a.md) for more detailed usage
- Learn about [MCP Protocol](mcp.md) for model-tool communication
- Explore the [A2A API Reference](../api/a2a.md) for detailed documentation
