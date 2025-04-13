"""
Test module for the Scoras library.

This module contains unit tests to verify the functionality of the Scoras library,
including the scoring system, agents, RAG, and tools.

Author: Anderson L. Amaral
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock

from pydantic import BaseModel, Field

import scoras as sc
from scoras.core import Agent, Tool, Message, WorkflowGraph, CompiledWorkflow, RAG, ScoreTracker
from scoras.agents import ExpertAgent, CreativeAgent, RAGAgent, MultiAgentSystem, AgentTeam
from scoras.rag import Document, SimpleRetriever, VectorRetriever, HybridRetriever
from scoras.tools import ToolRegistry, register_tool, ToolChain, ToolRouter


class TestCore(unittest.TestCase):
    """Tests for the core functionality of the library."""
    
    def test_agent_initialization(self):
        """Test the initialization of an agent."""
        agent = Agent(
            model="openai:gpt-4o",
            system_prompt="You are a helpful assistant.",
            tools=[]
        )
        
        self.assertEqual(agent.model, "openai:gpt-4o")
        self.assertEqual(agent.system_prompt, "You are a helpful assistant.")
        self.assertEqual(agent.tools, [])
    
    def test_tool_initialization(self):
        """Test the initialization of a tool."""
        def dummy_function(param1: str, param2: int) -> str:
            return f"{param1} {param2}"
        
        tool = Tool(
            name="dummy_tool",
            description="A test tool",
            function=dummy_function
        )
        
        self.assertEqual(tool.name, "dummy_tool")
        self.assertEqual(tool.description, "A test tool")
        self.assertEqual(tool.function, dummy_function)
        
        # Verify parameters were extracted correctly
        self.assertIn("param1", tool.parameters)
        self.assertIn("param2", tool.parameters)
    
    def test_message_initialization(self):
        """Test the initialization of a message."""
        message = Message(
            role="user",
            content="Hello, world!"
        )
        
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello, world!")
        
        # Test conversion to dictionary
        message_dict = message.to_dict()
        self.assertEqual(message_dict["role"], "user")
        self.assertEqual(message_dict["content"], "Hello, world!")
    
    def test_workflow_graph_initialization(self):
        """Test the initialization of a workflow graph."""
        class TestState(BaseModel):
            value: int
        
        graph = WorkflowGraph(state_type=TestState)
        
        self.assertEqual(graph.state_type, TestState)
        self.assertEqual(graph.nodes, {})
        self.assertEqual(graph.edges, {})
    
    def test_score_tracker(self):
        """Test the score tracker functionality."""
        tracker = ScoreTracker()
        
        # Add components
        tracker.add_node()
        tracker.add_node("complex")
        tracker.add_edge()
        tracker.add_edge(True)
        tracker.add_tool()
        tracker.add_tool("complex")
        
        # Check scores
        self.assertGreater(tracker.total_score, 0)
        self.assertGreater(tracker.components["nodes"], 0)
        self.assertGreater(tracker.components["edges"], 0)
        self.assertGreater(tracker.components["tools"], 0)
        self.assertGreater(tracker.components["conditions"], 0)
        
        # Check counts
        self.assertEqual(tracker.component_counts["nodes"], 2)
        self.assertEqual(tracker.component_counts["edges"], 2)
        self.assertEqual(tracker.component_counts["tools"], 2)
        self.assertEqual(tracker.component_counts["conditions"], 1)
        
        # Check report
        report = tracker.get_report()
        self.assertIn("total_score", report)
        self.assertIn("complexity_rating", report)
        self.assertIn("component_scores", report)
        self.assertIn("component_counts", report)
        self.assertIn("breakdown", report)
    
    @patch('scoras.core.ModelProvider')
    def test_agent_run_sync(self, mock_provider):
        """Test the synchronous execution of an agent."""
        # Configure the mock
        mock_instance = MagicMock()
        mock_instance.generate.return_value = "Test response"
        mock_provider.return_value = mock_instance
        
        # Create and run the agent
        agent = Agent(model="test:model")
        agent.provider = mock_instance
        
        result = agent.run_sync("Hello")
        
        # Verify the generate method was called
        mock_instance.generate.assert_called_once()
        self.assertEqual(result, "Test response")


class TestAgents(unittest.TestCase):
    """Tests for the specialized agents."""
    
    def test_expert_agent_initialization(self):
        """Test the initialization of an expert agent."""
        agent = ExpertAgent(
            model="openai:gpt-4o",
            domain="science",
            expertise_level="advanced"
        )
        
        self.assertEqual(agent.model, "openai:gpt-4o")
        self.assertEqual(agent.domain, "science")
        self.assertEqual(agent.expertise_level, "advanced")
        self.assertIn("expert in science", agent.system_prompt)
    
    def test_creative_agent_initialization(self):
        """Test the initialization of a creative agent."""
        agent = CreativeAgent(
            model="openai:gpt-4o",
            creative_mode="experimental",
            style_guide="Use vivid imagery."
        )
        
        self.assertEqual(agent.model, "openai:gpt-4o")
        self.assertEqual(agent.creative_mode, "experimental")
        self.assertEqual(agent.style_guide, "Use vivid imagery.")
        self.assertIn("creative assistant", agent.system_prompt)
        self.assertIn("Style Guide", agent.system_prompt)
    
    def test_rag_agent_initialization(self):
        """Test the initialization of a RAG agent."""
        def dummy_retriever(query: str) -> list:
            return ["Document 1", "Document 2"]
        
        agent = RAGAgent(
            model="openai:gpt-4o",
            retriever=dummy_retriever,
            citation_style="inline"
        )
        
        self.assertEqual(agent.model, "openai:gpt-4o")
        self.assertEqual(agent.retriever, dummy_retriever)
        self.assertEqual(agent.citation_style, "inline")
        self.assertIn("specialized agent", agent.system_prompt)
        self.assertIn("inline citations", agent.system_prompt)
    
    def test_multi_agent_system_initialization(self):
        """Test the initialization of a multi-agent system."""
        agents = {
            "agent1": Agent(model="openai:gpt-4o"),
            "agent2": Agent(model="anthropic:claude-3-opus")
        }
        
        system = MultiAgentSystem(agents)
        
        self.assertEqual(system.agents, agents)
        self.assertIn("agent1", system.agents)
        self.assertIn("agent2", system.agents)
    
    def test_agent_team_initialization(self):
        """Test the initialization of an agent team."""
        coordinator = Agent(model="openai:gpt-4o")
        specialists = {
            "specialist1": Agent(model="openai:gpt-4o"),
            "specialist2": Agent(model="anthropic:claude-3-opus")
        }
        
        team = AgentTeam(coordinator=coordinator, specialists=specialists)
        
        self.assertEqual(team.coordinator, coordinator)
        self.assertEqual(team.specialists, specialists)
        self.assertEqual(len(team.conversation_history), 0)


class TestRAG(unittest.TestCase):
    """Tests for the RAG system."""
    
    def test_document_initialization(self):
        """Test the initialization of a document."""
        doc = Document(
            content="Test content",
            metadata={"source": "test"}
        )
        
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata["source"], "test")
    
    def test_simple_retriever(self):
        """Test the simple retriever."""
        docs = [
            Document(content="Python is a programming language."),
            Document(content="Java is another programming language."),
            Document(content="Scoras is a library for creating agents.")
        ]
        
        retriever = SimpleRetriever(docs)
        
        results = retriever("Python programming")
        
        # Verify the document about Python is in the results
        self.assertTrue(any("Python" in r for r in results))
        
        # Verify the order is correct (most relevant first)
        self.assertIn("Python", results[0])
    
    def test_vector_retriever(self):
        """Test the vector retriever."""
        docs = [
            Document(content="Python is a programming language."),
            Document(content="Java is another programming language."),
            Document(content="Scoras is a library for creating agents.")
        ]
        
        # Simple mock embedding function
        def embedding_function(text):
            if "Python" in text:
                return [1.0, 0.0, 0.0]
            elif "Java" in text:
                return [0.0, 1.0, 0.0]
            else:
                return [0.0, 0.0, 1.0]
        
        retriever = VectorRetriever(docs, embedding_function)
        
        # Query with embedding similar to Python
        results = retriever("Python code")
        
        # Verify the document about Python is in the results
        self.assertTrue(any("Python" in r for r in results))
    
    def test_hybrid_retriever(self):
        """Test the hybrid retriever."""
        docs = [
            Document(content="Python is a programming language."),
            Document(content="Java is another programming language."),
            Document(content="Scoras is a library for creating agents.")
        ]
        
        simple_retriever = SimpleRetriever(docs)
        
        # Simple mock embedding function
        def embedding_function(text):
            if "Python" in text:
                return [1.0, 0.0, 0.0]
            elif "Java" in text:
                return [0.0, 1.0, 0.0]
            else:
                return [0.0, 0.0, 1.0]
        
        vector_retriever = VectorRetriever(docs, embedding_function)
        
        hybrid_retriever = HybridRetriever(
            retrievers=[simple_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        
        results = hybrid_retriever("Python programming")
        
        # Verify the document about Python is in the results
        self.assertTrue(any("Python" in r for r in results))


class TestTools(unittest.TestCase):
    """Tests for the tools functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        ToolRegistry.clear()
    
    def test_tool_registry(self):
        """Test the tool registry."""
        @register_tool(name="test_tool", description="Test tool")
        def test_function(param: str) -> str:
            return f"Test: {param}"
        
        # Verify the tool was registered
        self.assertIn("test_tool", [t.name for t in ToolRegistry.list()])
        
        # Verify the tool can be retrieved by name
        tool = ToolRegistry.get("test_tool")
        self.assertEqual(tool.name, "test_tool")
        self.assertEqual(tool.description, "Test tool")
    
    def test_tool_execution(self):
        """Test the execution of a tool."""
        def test_function(a: int, b: int) -> int:
            return a + b
        
        tool = Tool(
            name="sum",
            description="Sum two numbers",
            function=test_function
        )
        
        # Execute the tool synchronously
        result = asyncio.run(tool.execute(a=5, b=3))
        
        self.assertEqual(result, 8)
    
    def test_tool_chain(self):
        """Test the tool chain functionality."""
        def tool1_function(input: str) -> dict:
            return {"processed": input.upper()}
        
        def tool2_function(processed: str) -> str:
            return f"Final: {processed}"
        
        tool1 = Tool(
            name="tool1",
            description="First tool",
            function=tool1_function
        )
        
        tool2 = Tool(
            name="tool2",
            description="Second tool",
            function=tool2_function
        )
        
        chain = ToolChain(
            tools=[tool1, tool2],
            name="test_chain",
            description="Test chain"
        )
        
        # Execute the chain synchronously
        result = chain.execute_sync("hello")
        
        self.assertEqual(result, "Final: HELLO")
        
        # Check complexity score
        score = chain.get_complexity_score()
        self.assertIsNotNone(score)
        self.assertGreater(score["total_score"], 0)
    
    def test_tool_router(self):
        """Test the tool router functionality."""
        def tool1_function(input: str) -> str:
            return input.upper()
        
        def tool2_function(input: str) -> str:
            return input.lower()
        
        tool1 = Tool(
            name="upper",
            description="Convert to uppercase",
            function=tool1_function
        )
        
        tool2 = Tool(
            name="lower",
            description="Convert to lowercase",
            function=tool2_function
        )
        
        def router_function(inputs):
            if "uppercase" in inputs.get("mode", "").lower():
                return "upper"
            else:
                return "lower"
        
        router = ToolRouter(
            tools={"upper": tool1, "lower": tool2},
            router_function=router_function,
            name="text_case_router",
            description="Router for text case conversion"
        )
        
        # Execute the router synchronously
        result1 = router.execute_sync(input="Hello", mode="uppercase")
        result2 = router.execute_sync(input="Hello", mode="lowercase")
        
        self.assertEqual(result1, "HELLO")
        self.assertEqual(result2, "hello")
        
        # Check complexity score
        score = router.get_complexity_score()
        self.assertIsNotNone(score)
        self.assertGreater(score["total_score"], 0)


class TestWorkflowGraph(unittest.TestCase):
    """Tests for the workflow graph functionality."""
    
    def test_graph_creation_and_execution(self):
        """Test creating and executing a workflow graph."""
        # Create a graph
        graph = sc.Graph(state_type=dict)
        
        # Define node functions
        def process_input(state):
            return {"processed": state["input"].upper()}
        
        def analyze(state):
            return {"analysis": f"Analysis of {state['processed']}"}
        
        # Add nodes and edges
        graph.add_node("process_input", process_input)
        graph.add_node("analyze", analyze)
        graph.add_edge("start", "process_input")
        graph.add_edge("process_input", "analyze")
        graph.add_edge("analyze", "end")
        
        # Compile and run the graph
        compiled = graph.compile()
        result = compiled.run_sync({"input": "hello"})
        
        # Verify the result
        self.assertEqual(result["processed"], "HELLO")
        self.assertEqual(result["analysis"], "Analysis of HELLO")
        
        # Check complexity score
        score = graph.get_complexity_score()
        self.assertIsNotNone(score)
        self.assertGreater(score["total_score"], 0)
    
    def test_conditional_edges(self):
        """Test workflow graph with conditional edges."""
        # Create a graph
        graph = sc.Graph(state_type=dict)
        
        # Define node functions
        def process(state):
            return {"value": state["input"] * 2}
        
        def path_a(state):
            return {"result": f"Path A:
(Content truncated due to size limit. Use line ranges to read in chunks)