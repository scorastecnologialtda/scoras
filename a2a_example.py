"""
Example script demonstrating A2A (Agent-to-Agent) protocol support in Scoras.

This example shows how to:
1. Create an A2A server with a Scoras agent
2. Connect to other A2A agents with an A2A client
3. Send tasks and messages between agents
4. Track complexity scores throughout the process

Author: Anderson L. Amaral
"""

import asyncio
import json
import os
import uuid
from typing import Dict, Any, List

import scoras as sc
from scoras.a2a import (
    create_agent_skill,
    create_a2a_server,
    A2AClient,
    A2AAgentAdapter,
    run_a2a_server
)

# Define some skills for our A2A agent
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
    examples=[
        "Summarize the latest research on renewable energy",
        "Compare and contrast different machine learning algorithms"
    ],
    complexity="complex"
)

writing_skill = create_agent_skill(
    id="writing",
    name="Writing",
    description="Create and edit written content",
    tags=["writing", "content", "editing"],
    examples=[
        "Write a blog post about artificial intelligence",
        "Edit and improve this paragraph for clarity"
    ],
    complexity="standard"
)

async def run_a2a_server_example():
    """Run the A2A server example."""
    print("=== A2A Server Example ===")
    
    # Create an agent
    agent = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant with expertise in mathematics, research, and writing.",
        enable_scoring=True
    )
    
    # Create an A2A server
    server = create_a2a_server(
        name="ScorasAgent",
        description="A versatile agent with multiple skills powered by Scoras",
        agent=agent,
        skills=[math_skill, research_skill, writing_skill],
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
    
    # Print the agent card
    agent_card = server.get_agent_card()
    print("Agent Card:", json.dumps(agent_card.model_dump(), indent=2))
    
    # Simulate handling a task
    print("\nSimulating handling a task...")
    task_id = str(uuid.uuid4())
    task_data = {
        "id": task_id,
        "messages": [
            {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": "Calculate the area of a circle with radius 5 cm."
                    }
                ]
            }
        ]
    }
    
    # In a real application, this would be handled by the HTTP server
    # For this example, we'll call the handler directly
    response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": {
            "task": task_data
        },
        "id": "request1"
    })
    
    print("Response:", json.dumps(response, indent=2))
    
    # Get the complexity score
    score = server.get_complexity_score()
    print("\nServer Complexity Score:", json.dumps(score, indent=2))
    
    # In a real application, you would run the server with:
    # await run_a2a_server(server, host="0.0.0.0", port=8001)
    # For this example, we'll just print a message
    print("\nIn a real application, the server would be running at http://0.0.0.0:8001")

async def run_a2a_client_example():
    """Run the A2A client example."""
    print("\n=== A2A Client Example ===")
    
    # Create an A2A client
    # In a real application, this would connect to a running server
    # For this example, we'll simulate the connection
    client = A2AClient(
        agent_url="http://localhost:8001",
        enable_scoring=True
    )
    
    # Simulate getting agent card
    print("Connecting to A2A agent...")
    
    # Simulate sending a task
    print("\nSending a task to the agent...")
    try:
        # In a real application, this would call the agent
        # For this example, we'll simulate the response
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "state": "completed",
            "messages": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "What is the capital of France?"
                        }
                    ]
                },
                {
                    "role": "agent",
                    "parts": [
                        {
                            "type": "text",
                            "text": "The capital of France is Paris."
                        }
                    ]
                }
            ],
            "artifacts": [],
            "created_at": "2025-04-13T02:30:00Z",
            "updated_at": "2025-04-13T02:30:05Z"
        }
        print("Task:", json.dumps(task, indent=2))
    except Exception as e:
        print(f"Error: {e}")
    
    # Simulate sending a message to an existing task
    print("\nSending a follow-up message...")
    try:
        # In a real application, this would call the agent
        # For this example, we'll simulate the response
        updated_task = {
            "id": task_id,
            "state": "completed",
            "messages": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "What is the capital of France?"
                        }
                    ]
                },
                {
                    "role": "agent",
                    "parts": [
                        {
                            "type": "text",
                            "text": "The capital of France is Paris."
                        }
                    ]
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "What is the population of Paris?"
                        }
                    ]
                },
                {
                    "role": "agent",
                    "parts": [
                        {
                            "type": "text",
                            "text": "The population of Paris is approximately 2.2 million people in the city proper, and over 12 million people in the metropolitan area."
                        }
                    ]
                }
            ],
            "artifacts": [],
            "created_at": "2025-04-13T02:30:00Z",
            "updated_at": "2025-04-13T02:30:15Z"
        }
        print("Updated Task:", json.dumps(updated_task, indent=2))
    except Exception as e:
        print(f"Error: {e}")
    
    # Get the complexity score
    score = client.get_complexity_score()
    print("\nClient Complexity Score:", json.dumps(score, indent=2))

async def run_a2a_multi_agent_example():
    """Run the A2A multi-agent example."""
    print("\n=== A2A Multi-Agent Example ===")
    
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
    
    # Simulate connecting to multiple A2A agents
    print("Connecting to multiple A2A agents...")
    try:
        # In a real application, this would connect to running agents
        # For this example, we'll simulate the connections
        math_agent_id = "math_agent"
        research_agent_id = "research_agent"
        print(f"Connected to agents with IDs: {math_agent_id}, {research_agent_id}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Simulate sending a task to the math agent
    print("\nSending a task to the math agent...")
    try:
        # In a real application, this would call the agent
        # For this example, we'll simulate the response
        math_task_id = str(uuid.uuid4())
        math_task = {
            "id": math_task_id,
            "state": "completed",
            "messages": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "Solve the equation 3x + 7 = 22"
                        }
                    ]
                },
                {
                    "role": "agent",
                    "parts": [
                        {
                            "type": "text",
                            "text": "To solve the equation 3x + 7 = 22:\n\n3x + 7 = 22\n3x = 22 - 7\n3x = 15\nx = 15/3\nx = 5\n\nThe solution is x = 5."
                        }
                    ]
                }
            ],
            "artifacts": [],
            "created_at": "2025-04-13T02:30:00Z",
            "updated_at": "2025-04-13T02:30:05Z"
        }
        print("Math Task:", json.dumps(math_task, indent=2))
    except Exception as e:
        print(f"Error: {e}")
    
    # Simulate sending a task to the research agent
    print("\nSending a task to the research agent...")
    try:
        # In a real application, this would call the agent
        # For this example, we'll simulate the response
        research_task_id = str(uuid.uuid4())
        research_task = {
            "id": research_task_id,
            "state": "completed",
            "messages": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "Find information about quantum computing"
                        }
                    ]
                },
                {
                    "role": "agent",
                    "parts": [
                        {
                            "type": "text",
                            "text": "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. The devices that perform quantum computations are known as quantum computers.\n\nUnlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously due to superposition. This allows quantum computers to process a vast number of possibilities simultaneously.\n\nSome key concepts in quantum computing include:\n\n1. Superposition: Qubits can exist in multiple states at once\n2. Entanglement: Qubits can be correlated with each other\n3. Quantum gates: Operations that manipulate qubits\n4. Quantum algorithms: Specialized algorithms designed for quantum computers\n\nQuantum computers have the potential to solve certain problems much faster than classical computers, particularly in areas like cryptography, optimization, and simulation of quantum systems."
                        }
                    ]
                }
            ],
            "artifacts": [],
            "created_at": "2025-04-13T02:30:00Z",
            "updated_at": "2025-04-13T02:30:10Z"
        }
        print("Research Task:", json.dumps(research_task, indent=2))
    except Exception as e:
        print(f"Error: {e}")
    
    # Simulate coordinating between agents
    print("\nCoordinating between agents...")
    try:
        # In a real application, this would involve multiple agent interactions
        # For this example, we'll simulate the coordination
        
        # First, get results from both agents
        math_result = "x = 5"
        research_result = "Quantum computers use qubits instead of classical bits."
        
        # Then, have the main agent combine the results
        combined_result = f"I've consulted with specialized agents and found that:\n\n1. Math problem solution: {math_result}\n2. Quantum computing information: {research_result}\n\nWould you like more detailed information on either topic?"
        
        print("Coordinated Result:", combined_result)
    except Exception as e:
        print(f"Error: {e}")
    
    # Get the complexity score
    score = adapter.get_complexity_score()
    print("\nAdapter Complexity Score:", json.dumps(score, indent=2))

async def main():
    """Run all examples."""
    await run_a2a_server_example()
    await run_a2a_client_example()
    await run_a2a_multi_agent_example()

if __name__ == "__main__":
    asyncio.run(main())
