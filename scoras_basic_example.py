#!/usr/bin/env python3
"""
Scoras Library Basic Example

This example demonstrates the core functionality of the Scoras library,
including creating agents, using the RAG system, and working with tools.
"""

import asyncio
import os
from typing import Dict, Any

# Import Scoras components
import scoras as sc
from scoras.core import Graph, Node, Edge, Agent, Message, Tool
from scoras.rag import Document, SimpleRAG
from scoras.tools import tool


# Define a simple tool using the decorator
@tool(name="calculator", description="A simple calculator tool")
def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform a simple calculation.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
        
    Returns:
        Result of the calculation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else float('inf')
    else:
        raise ValueError(f"Unknown operation: {operation}")


# Define a simple workflow using Graph, Node, and Edge
def create_workflow() -> Graph:
    """
    Create a simple workflow graph.
    
    Returns:
        A Graph instance representing the workflow
    """
    # Create a graph
    graph = Graph("data_processing")
    
    # Add nodes
    graph.add_node(Node("extract"))
    graph.add_node(Node("transform"))
    graph.add_node(Node("load"))
    
    # Add edges
    graph.add_edge(Edge("extract", "transform"))
    graph.add_edge(Edge("transform", "load"))
    
    # Print complexity score
    complexity = graph.get_complexity_score()
    print(f"Workflow complexity score: {complexity['total_score']}")
    print(f"Workflow complexity components: {complexity['components']}")
    
    return graph


# Create a simple RAG system
def create_rag_system() -> SimpleRAG:
    """
    Create a simple RAG system with sample documents.
    
    Returns:
        A SimpleRAG instance
    """
    # Create an agent
    agent = Agent("openai:gpt-4")
    
    # Create a RAG system
    rag = SimpleRAG(agent)
    
    # Add documents
    doc1 = Document(
        content="Scoras is a Python library for creating intelligent agents with complexity scoring.",
        metadata={"source": "documentation", "topic": "overview"}
    )
    
    doc2 = Document(
        content="The name 'Scoras' comes from 'Score', and each graph (or edge) represents one score (score=1) to measure the complexity of agentic workflows.",
        metadata={"source": "documentation", "topic": "scoring"}
    )
    
    # Add documents to RAG
    rag.add_document(doc1)
    rag.add_document(doc2)
    
    # Print document IDs
    print(f"Document 1 ID: {doc1.id}")
    print(f"Document 2 ID: {doc2.id}")
    
    return rag


# Create and use messages
def demonstrate_messages() -> None:
    """
    Demonstrate creating and using Message objects.
    """
    # Create user message
    user_msg = Message(
        role="user",
        content="What is the Scoras library?",
        metadata={"timestamp": "2025-04-23T12:00:00Z"}
    )
    
    # Create assistant message
    assistant_msg = Message(
        role="assistant",
        content="Scoras is a Python library for creating intelligent agents with complexity scoring.",
        metadata={"timestamp": "2025-04-23T12:01:00Z", "model": "gpt-4"}
    )
    
    # Print messages
    print(f"User message: {user_msg.content}")
    print(f"Assistant message: {assistant_msg.content}")
    print(f"User message metadata: {user_msg.metadata}")


async def main() -> None:
    """
    Main function to demonstrate Scoras functionality.
    """
    print("\n" + "=" * 80)
    print("SCORAS LIBRARY BASIC EXAMPLE".center(80))
    print("=" * 80 + "\n")
    
    # Print Scoras version
    print(f"Scoras version: {sc.__version__}")
    print(f"Scoras location: {sc.__file__}\n")
    
    # Create and demonstrate a workflow
    print("\n--- Workflow Example ---")
    workflow = create_workflow()
    
    # Create and demonstrate a RAG system
    print("\n--- RAG System Example ---")
    rag = create_rag_system()
    
    # Run a query on the RAG system
    query = "What is Scoras?"
    print(f"\nRunning query: '{query}'")
    response = rag.run_sync(query)
    print(f"Response: {response}")
    
    # Demonstrate messages
    print("\n--- Message Example ---")
    demonstrate_messages()
    
    # Use the calculator tool
    print("\n--- Tool Example ---")
    result = calculator("add", 5, 3)
    print(f"Calculator tool - add(5, 3) = {result}")
    
    result = calculator("multiply", 4, 7)
    print(f"Calculator tool - multiply(4, 7) = {result}")
    
    # Print tool information
    print(f"Tool name: {calculator.tool_name}")
    print(f"Tool description: {calculator.tool_description}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETED SUCCESSFULLY".center(80))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
