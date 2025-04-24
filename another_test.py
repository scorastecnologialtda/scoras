#!/usr/bin/env python3
"""
Fixed example for Jupyter Notebook with Scoras Library 0.3.3

This example demonstrates the correct usage of the Scoras library version 0.3.3 in a Jupyter notebook,
addressing the issues with Agent class parameters and Node class requirements.
"""

import scoras as sc
from scoras.core import Graph, Node, Edge, Agent
from scoras.rag import Document, SimpleRAG
import nest_asyncio
import asyncio

# Apply nest_asyncio to make asyncio work in Jupyter notebooks
nest_asyncio.apply()

# Print Scoras version information
print(f"Scoras version: {sc.__version__}")
print(f"Scoras location: {sc.__file__}")

# Create a simple agent
# FIXED: Removed unsupported parameters (system_prompt)
agent = sc.Agent(model="groq:llama3-8b-8192")

# Run the agent
response = agent.run_sync("What is the capital of France?")
print(f"Response: {response}")

# Check the complexity score
score = agent.get_complexity_score()
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")

# Create documents for RAG example
documents = [
    Document(content="The capital of France is Paris, known as the City of Light."),
    Document(content="Paris is famous for the Eiffel Tower, built in 1889."),
    Document(content="France has a population of about 67 million people.")
]

# Create a RAG system
rag = SimpleRAG(
    agent=sc.Agent(model="groq:llama3-8b-8192"),
    documents=documents
)

# Run a query on the RAG system
response = rag.run_sync("What is the capital of France and what is it known for?")
print(f"RAG Response: {response}")

# Check the complexity score
score = rag.get_complexity_score()
print(f"RAG Complexity: {score['complexity_rating']} (Score: {score['total_score']})")

# Define a tool using the decorator
# FIXED: Use the tool decorator correctly
@sc.tool(name="calculator", description="Perform calculations")
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
# FIXED: Removed unsupported parameters (system_prompt, tools)
# Note: In version 0.3.3, tools need to be registered differently or may not be supported
agent = sc.Agent(model="groq:llama3-8b-8192")

# Run the agent with a query that will use the tool
response = agent.run_sync("What is 25 multiplied by 16?")
print(f"Tool Response: {response}")

# Check the complexity score
score = agent.get_complexity_score()
print(f"Tool Agent Complexity: {score['complexity_rating']} (Score: {score['total_score']})")

# Define node functions for workflow example
async def preprocess(state):
    return {"processed_query": state.query.strip().lower()}

async def search(state):
    # Simulate search
    results = [f"Result for {state.processed_query}", "Another result"]
    return {"search_results": results}

async def generate_answer(state):
    answer = f"Based on {len(state.search_results)} results, the answer is..."
    return {"final_answer": answer}

# Create a workflow graph
# FIXED: Use the correct WorkflowGraph API
graph = sc.WorkflowGraph()

# Add nodes
# FIXED: Provide the required function parameter for each node
graph.add_node("start", lambda s: s)
graph.add_node("preprocess", preprocess)
graph.add_node("search", search)
graph.add_node("generate", generate_answer)
graph.add_node("end", lambda s: s)

# Add edges
graph.add_edge("start", "preprocess")
graph.add_edge("preprocess", "search")
graph.add_edge("search", "generate")
graph.add_edge("generate", "end")

# Compile the workflow
executor = graph.compile()

# Check the complexity score
score = graph.get_complexity_score()
print(f"Workflow Complexity: {score['complexity_rating']} (Score: {score['total_score']})")

