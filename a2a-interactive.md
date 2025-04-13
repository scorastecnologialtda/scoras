# Interactive A2A Example

This page demonstrates the Agent-to-Agent (A2A) protocol in action with an interactive example. You can experiment with different agents, send messages, and see how the complexity score changes as you interact with the system.

## Try It Yourself

Below is an interactive A2A demo that simulates a multi-agent system with a coordinator, math agent, and research agent. You can select different agents to interact with and see how tasks are delegated and processed.

<div id="a2a-example-demo" class="a2a-interactive-demo"></div>

## How It Works

The interactive demo above demonstrates the key components of the A2A protocol:

1. **Agent Cards**: The left panel shows the available agents and their skills.
2. **Conversation**: The center panel shows the conversation between you and the agents.
3. **Agent Selection**: You can select which agent to send your message to.
4. **Task Management**: The system tracks tasks and their states.
5. **Complexity Scoring**: The bottom panel shows the complexity score, which increases as you interact with more complex agents.

## Code Example

Here's how you would implement a similar multi-agent system using Scoras and the A2A protocol:

```python
import scoras as sc
from scoras.a2a import create_agent_skill, create_a2a_server, A2AClient
import asyncio

# Define skills for the agents
math_skill = create_agent_skill(
    id="math",
    name="Mathematics",
    description="Perform mathematical calculations and solve problems",
    tags=["math", "calculation", "problem-solving"],
    complexity="standard"
)

research_skill = create_agent_skill(
    id="research",
    name="Research",
    description="Find and analyze information on various topics",
    tags=["research", "information", "analysis"],
    complexity="complex"
)

async def setup_agents():
    # Create a math agent
    math_agent = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="You are a mathematics expert. You excel at solving math problems and explaining mathematical concepts.",
        enable_scoring=True
    )
    
    # Create a research agent
    research_agent = sc.Agent(
        model="anthropic:claude-3-opus",
        system_prompt="You are a research specialist. You excel at finding and analyzing information on various topics.",
        enable_scoring=True
    )
    
    # Create A2A servers for each agent
    math_server = create_a2a_server(
        name="MathAgent",
        description="A specialized agent for mathematics",
        agent=math_agent,
        skills=[math_skill],
        enable_scoring=True
    )
    
    research_server = create_a2a_server(
        name="ResearchAgent",
        description="A specialized agent for research",
        agent=research_agent,
        skills=[research_skill],
        enable_scoring=True
    )
    
    # Start the servers in the background
    math_task = asyncio.create_task(
        run_a2a_server(math_server, host="0.0.0.0", port=8001)
    )
    
    research_task = asyncio.create_task(
        run_a2a_server(research_server, host="0.0.0.0", port=8002)
    )
    
    return math_server, research_server, math_task, research_task

async def run_coordinator():
    # Set up the agents
    math_server, research_server, math_task, research_task = await setup_agents()
    
    # Create clients to connect to the agents
    math_client = A2AClient(
        agent_url="http://localhost:8001",
        enable_scoring=True
    )
    
    research_client = A2AClient(
        agent_url="http://localhost:8002",
        enable_scoring=True
    )
    
    # Process a user query
    user_query = "What is the area of a circle with radius 5 cm?"
    
    # Determine which agent to use
    if is_math_query(user_query):
        print(f"Delegating to Math Agent: {user_query}")
        
        # Send task to math agent
        math_task = await math_client.send_task(message=user_query)
        
        # Wait for the task to complete
        completed_task = await math_client.wait_for_task(math_task.id)
        
        # Get the response
        response = completed_task.messages[-1].parts[0].text
        print(f"Math Agent response: {response}")
        
        # Get complexity score
        score = math_client.get_complexity_score()
        print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")
    else:
        print(f"Delegating to Research Agent: {user_query}")
        
        # Send task to research agent
        research_task = await research_client.send_task(message=user_query)
        
        # Wait for the task to complete
        completed_task = await research_client.wait_for_task(research_task.id)
        
        # Get the response
        response = completed_task.messages[-1].parts[0].text
        print(f"Research Agent response: {response}")
        
        # Get complexity score
        score = research_client.get_complexity_score()
        print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")
    
    # Cancel the server tasks when done
    math_task.cancel()
    research_task.cancel()

def is_math_query(query):
    # Simple function to determine if a query is math-related
    math_keywords = ["calculate", "solve", "equation", "math", "formula", "area", "volume"]
    return any(keyword in query.lower() for keyword in math_keywords)

if __name__ == "__main__":
    asyncio.run(run_coordinator())
```

## Key Features

The A2A implementation in Scoras provides several key features:

- **Agent Skills**: Define agent capabilities as skills with complexity ratings
- **Server Creation**: Create A2A servers that expose agents via the protocol
- **Client Connection**: Connect to A2A agents to use their capabilities
- **Task Management**: Send tasks to agents and track their progress
- **Complexity Tracking**: Monitor the complexity of agent interactions
- **Multi-Agent Systems**: Build systems with multiple specialized agents

## Next Steps

- Check out the [MCP Interactive Example](mcp-interactive.md) to see how models interact with tools
- Learn more about the [A2A Protocol](../protocols/a2a.md)
- Explore the [A2A API Reference](../api/a2a.md)
