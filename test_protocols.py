"""
Test script for validating MCP and A2A protocol implementations in Scoras.

This script tests:
1. Core functionality with scoring
2. MCP client and server functionality
3. A2A client and server functionality
4. Integration of protocols with the scoring system

Author: Anderson L. Amaral
"""

import asyncio
import json
import os
import unittest
from typing import Dict, Any, List

import scoras as sc
from scoras.mcp import (
    create_mcp_server,
    MCPClient,
    MCPContext,
    MCPAgentAdapter
)
from scoras.a2a import (
    create_agent_skill,
    create_a2a_server,
    A2AClient,
    A2AAgentAdapter
)

# Define test tools
@sc.tool(name="test_calculator", description="Test calculator tool", complexity="simple")
async def test_calculator(operation: str, a: float, b: float) -> float:
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

class TestScoras(unittest.TestCase):
    """Test cases for Scoras library."""
    
    def test_score_tracker(self):
        """Test the ScoreTracker functionality."""
        tracker = sc.core.ScoreTracker()
        
        # Add components
        tracker.add_node("simple")
        tracker.add_node("standard")
        tracker.add_node("complex")
        tracker.add_edge()
        tracker.add_edge(True)  # Conditional edge
        tracker.add_tool("simple")
        tracker.add_tool("standard")
        tracker.add_tool("complex")
        
        # Get the report
        report = tracker.get_report()
        
        # Verify the report
        self.assertIsNotNone(report)
        self.assertIn("total_score", report)
        self.assertIn("complexity_rating", report)
        self.assertIn("component_scores", report)
        self.assertIn("component_counts", report)
        self.assertIn("breakdown", report)
        
        # Verify component counts
        self.assertEqual(report["component_counts"]["nodes"], 3)
        self.assertEqual(report["component_counts"]["edges"], 2)
        self.assertEqual(report["component_counts"]["tools"], 3)
        self.assertEqual(report["component_counts"]["conditions"], 1)
        
        # Verify complexity rating
        self.assertIn(report["complexity_rating"], 
                     ["Simple", "Moderate", "Complex", "Very Complex", "Extremely Complex"])
        
        print(f"Score Tracker Test: {report['complexity_rating']} (Score: {report['total_score']})")
    
    def test_agent_with_scoring(self):
        """Test an agent with scoring enabled."""
        # Create an agent
        agent = sc.Agent(
            model="openai:gpt-4o",  # Note: This won't actually call the API in tests
            system_prompt="You are a helpful assistant.",
            tools=[test_calculator],
            enable_scoring=True
        )
        
        # Add a tool
        @sc.tool(name="test_tool", description="Another test tool", complexity="standard")
        async def test_tool(input: str) -> str:
            return f"Processed: {input}"
        
        agent.add_tool(test_tool)
        
        # Get the complexity score
        score = agent.get_complexity_score()
        
        # Verify the score
        self.assertIsNotNone(score)
        self.assertIn("total_score", score)
        self.assertIn("complexity_rating", score)
        
        # Verify tool count
        self.assertEqual(score["component_counts"]["tools"], 2)
        
        print(f"Agent Test: {score['complexity_rating']} (Score: {score['total_score']})")
    
    def test_workflow_graph_with_scoring(self):
        """Test a workflow graph with scoring enabled."""
        from pydantic import BaseModel
        
        # Define a state model
        class TestState(BaseModel):
            input: str
            output: str = ""
        
        # Create a workflow graph
        graph = sc.WorkflowGraph(
            state_type=TestState,
            enable_scoring=True
        )
        
        # Add nodes and edges
        graph.add_node("start", lambda s: s, "simple")
        graph.add_node("process", lambda s: {"output": f"Processed: {s.input}"}, "standard")
        graph.add_node("end", lambda s: s, "simple")
        
        graph.add_edge("start", "process")
        graph.add_edge("process", "end")
        
        # Add a conditional edge
        graph.add_edge(
            "process", 
            "end", 
            condition=lambda s: len(s.input) > 10
        )
        
        # Get the complexity score
        score = graph.get_complexity_score()
        
        # Verify the score
        self.assertIsNotNone(score)
        self.assertIn("total_score", score)
        self.assertIn("complexity_rating", score)
        
        # Verify component counts
        self.assertEqual(score["component_counts"]["nodes"], 3)
        self.assertEqual(score["component_counts"]["edges"], 3)
        self.assertEqual(score["component_counts"]["conditions"], 1)
        
        print(f"Workflow Graph Test: {score['complexity_rating']} (Score: {score['total_score']})")

class TestMCPProtocol(unittest.TestCase):
    """Test cases for MCP protocol support."""
    
    def test_mcp_server_creation(self):
        """Test creating an MCP server."""
        # Create an MCP server
        server = create_mcp_server(
            name="TestServer",
            description="Test MCP server",
            tools=[test_calculator],
            capabilities=["tools"],
            enable_scoring=True
        )
        
        # Verify server info
        info = server.get_server_info()
        self.assertEqual(info["name"], "TestServer")
        self.assertEqual(info["description"], "Test MCP server")
        self.assertIn("tools_count", info)
        self.assertEqual(info["tools_count"], 1)
        
        # Verify available tools
        tools = server.get_available_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "test_calculator")
        
        # Verify complexity score
        score = server.get_complexity_score()
        self.assertIsNotNone(score)
        self.assertIn("total_score", score)
        self.assertIn("complexity_rating", score)
        
        print(f"MCP Server Test: {score['complexity_rating']} (Score: {score['total_score']})")
    
    def test_mcp_client_creation(self):
        """Test creating an MCP client."""
        # Create an MCP client
        client = MCPClient(
            server_url="http://localhost:8000",  # Not actually connecting in tests
            enable_scoring=True
        )
        
        # Verify complexity score
        score = client.get_complexity_score()
        self.assertIsNotNone(score)
        self.assertIn("total_score", score)
        self.assertIn("complexity_rating", score)
        
        print(f"MCP Client Test: {score['complexity_rating']} (Score: {score['total_score']})")
    
    def test_mcp_agent_adapter(self):
        """Test the MCP agent adapter."""
        # Create an agent
        agent = sc.Agent(
            model="openai:gpt-4o",  # Not actually calling the API in tests
            system_prompt="You are a helpful assistant.",
            enable_scoring=True
        )
        
        # Create an MCP agent adapter
        adapter = MCPAgentAdapter(
            agent=agent,
            enable_scoring=True
        )
        
        # Verify complexity score
        score = adapter.get_complexity_score()
        self.assertIsNotNone(score)
        self.assertIn("total_score", score)
        self.assertIn("complexity_rating", score)
        
        print(f"MCP Agent Adapter Test: {score['complexity_rating']} (Score: {score['total_score']})")

class TestA2AProtocol(unittest.TestCase):
    """Test cases for A2A protocol support."""
    
    def test_a2a_server_creation(self):
        """Test creating an A2A server."""
        # Create an agent
        agent = sc.Agent(
            model="openai:gpt-4o",  # Not actually calling the API in tests
            system_prompt="You are a helpful assistant.",
            enable_scoring=True
        )
        
        # Create skills
        math_skill = create_agent_skill(
            id="math",
            name="Mathematics",
            description="Perform mathematical calculations",
            complexity="standard"
        )
        
        # Create an A2A server
        server = create_a2a_server(
            name="TestAgent",
            description="Test A2A agent",
            agent=agent,
            skills=[math_skill],
            enable_scoring=True
        )
        
        # Verify agent card
        card = server.get_agent_card()
        self.assertEqual(card.name, "TestAgent")
        self.assertEqual(card.description, "Test A2A agent")
        self.assertEqual(len(card.skills), 1)
        self.assertEqual(card.skills[0].name, "Mathematics")
        
        # Verify complexity score
        score = server.get_complexity_score()
        self.assertIsNotNone(score)
        self.assertIn("total_score", score)
        self.assertIn("complexity_rating", score)
        
        print(f"A2A Server Test: {score['complexity_rating']} (Score: {score['total_score']})")
    
    def test_a2a_client_creation(self):
        """Test creating an A2A client."""
        # Create an A2A client
        client = A2AClient(
            agent_url="http://localhost:8001",  # Not actually connecting in tests
            enable_scoring=True
        )
        
        # Verify complexity score
        score = client.get_complexity_score()
        self.assertIsNotNone(score)
        self.assertIn("total_score", score)
        self.assertIn("complexity_rating", score)
        
        print(f"A2A Client Test: {score['complexity_rating']} (Score: {score['total_score']})")
    
    def test_a2a_agent_adapter(self):
        """Test the A2A agent adapter."""
        # Create an agent
        agent = sc.Agent(
            model="openai:gpt-4o",  # Not actually calling the API in tests
            system_prompt="You are a helpful assistant.",
            enable_scoring=True
        )
        
        # Create an A2A agent adapter
        adapter = A2AAgentAdapter(
            agent=agent,
            enable_scoring=True
        )
        
        # Verify complexity score
        score = adapter.get_complexity_score()
        self.assertIsNotNone(score)
        self.assertIn("total_score", score)
        self.assertIn("complexity_rating", score)
        
        print(f"A2A Agent Adapter Test: {score['complexity_rating']} (Score: {score['total_score']})")

class TestProtocolIntegration(unittest.TestCase):
    """Test cases for protocol integration with Scoras."""
    
    def test_protocol_compatibility(self):
        """Test that protocols are compatible with core Scoras features."""
        # Create an agent with tools
        agent = sc.Agent(
            model="openai:gpt-4o",  # Not actually calling the API in tests
            system_prompt="You are a helpful assistant.",
            tools=[test_calculator],
            enable_scoring=True
        )
        
        # Create MCP and A2A adapters
        mcp_adapter = MCPAgentAdapter(agent=agent, enable_scoring=True)
        a2a_adapter = A2AAgentAdapter(agent=agent, enable_scoring=True)
        
        # Verify MCP adapter score
        mcp_score = mcp_adapter.get_complexity_score()
        self.assertIsNotNone(mcp_score)
        self.assertIn("total_score", mcp_score)
        
        # Verify A2A adapter score
        a2a_score = a2a_adapter.get_complexity_score()
        self.assertIsNotNone(a2a_score)
        self.assertIn("total_score", a2a_score)
        
        print(f"MCP Integration Test: {mcp_score['complexity_rating']} (Score: {mcp_score['total_score']})")
        print(f"A2A Integration Test: {a2a_score['complexity_rating']} (Score: {a2a_score['total_score']})")
    
    def test_combined_protocols(self):
        """Test using both protocols together."""
        # Create an agent
        agent = sc.Agent(
            model="openai:gpt-4o",  # Not actually calling the API in tests
            system_prompt="You are a helpful assistant.",
            tools=[test_calculator],
            enable_scoring=True
        )
        
        # Create MCP server and A2A server
        mcp_server = create_mcp_server(
            name="MCPServer",
            description="MCP server",
            tools=[test_calculator],
            enable_scoring=True
        )
        
        math_skill = create_agent_skill(
            id="math",
            name="Mathematics",
            description="Perform mathematical calculations",
            complexity="standard"
        )
        
        a2a_server = create_a2a_server(
            name="A2AAgent",
            description="A2A agent",
            agent=agent,
            skills=[math_skill],
            enable_scoring=True
        )
        
        # Verify both servers have valid complexity scores
        mcp_score = mcp_server.get_complexity_score()
        a2a_score = a2a_server.get_complexity_score()
        
        self.assertIsNotNone(mcp_score)
        self.assertIsNotNone(a2a_score)
        
        print(f"Combined Protocols Test - MCP: {mcp_score['complexity_rating']} (Score: {mcp_score['total_score']})")
        print(f"Combined Protocols Test - A2A: {a2a_score['complexity_rating']} (Score: {a2a_score['total_score']})")

def run_tests():
    """Run all tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests()
