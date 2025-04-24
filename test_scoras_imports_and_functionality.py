#!/usr/bin/env python3
"""
Scoras Library Functionality Test Script

This script tests both imports and basic functionality of the Scoras library
to verify that components are properly installed and working correctly.
"""

import sys
import importlib
import inspect
from typing import Dict, List, Any, Optional, Tuple
import asyncio

def test_import(module_name: str, components: List[str]) -> Tuple[List[str], List[str]]:
    """
    Test importing specific components from a module.
    
    Args:
        module_name: Name of the module to import from
        components: List of component names to import
        
    Returns:
        Tuple of (successful imports, failed imports)
    """
    successful = []
    failed = []
    
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported module: {module_name}")
        
        for component in components:
            try:
                if hasattr(module, component):
                    # Get the component
                    comp = getattr(module, component)
                    # Check if it's a class, function, or other object
                    comp_type = "class" if inspect.isclass(comp) else "function" if inspect.isfunction(comp) else "object"
                    successful.append(f"{component} ({comp_type})")
                else:
                    failed.append(f"{component} (not found in {module_name})")
            except Exception as e:
                failed.append(f"{component} (error: {str(e)})")
    except ImportError as e:
        print(f"❌ Failed to import module: {module_name} - {str(e)}")
        failed.extend([f"{component} (module not found)" for component in components])
    
    return successful, failed

def print_results(module_name: str, successful: List[str], failed: List[str]) -> None:
    """
    Print the results of import tests for a module.
    
    Args:
        module_name: Name of the module
        successful: List of successfully imported components
        failed: List of failed imports
    """
    print(f"\n--- {module_name} Import Results ---")
    
    if successful:
        print("Successfully imported:")
        for item in successful:
            print(f"  ✅ {item}")
    
    if failed:
        print("Failed to import:")
        for item in failed:
            print(f"  ❌ {item}")

async def test_functionality() -> None:
    """
    Test basic functionality of key Scoras components.
    """
    print("\n" + "=" * 80)
    print("SCORAS FUNCTIONALITY TESTS".center(80))
    print("=" * 80 + "\n")
    
    functionality_tests = []
    
    # Test 1: Create and use a Node
    try:
        from scoras.core import Node
        
        # Define a simple function for the node
        async def test_function(data):
            return f"Processed: {data}"
        
        # Create a node
        node = Node("test_node", test_function)
        
        # Execute the node
        result = await node.execute("test_data")
        
        if result == "Processed: test_data":
            print("✅ Node functionality test passed")
            functionality_tests.append(True)
        else:
            print(f"❌ Node functionality test failed: unexpected result '{result}'")
            functionality_tests.append(False)
    except Exception as e:
        print(f"❌ Node functionality test failed with error: {str(e)}")
        functionality_tests.append(False)
    
    # Test 2: Create and use a Document
    try:
        from scoras.rag import Document
        
        # Create a document
        doc = Document(content="Test document content", metadata={"source": "test"})
        
        # Check if the document has an ID
        if hasattr(doc, "id") and doc.id:
            print("✅ Document functionality test passed (has ID)")
            functionality_tests.append(True)
        else:
            print("❌ Document functionality test failed: missing ID attribute")
            functionality_tests.append(False)
    except Exception as e:
        print(f"❌ Document functionality test failed with error: {str(e)}")
        functionality_tests.append(False)
    
    # Test 3: Create a Graph
    try:
        from scoras.core import Graph, Node, Edge
        
        # Create a graph
        graph = Graph("test_graph")
        
        # Define simple functions for nodes
        async def extract_function(data):
            return f"Extracted: {data}"
            
        async def transform_function(data):
            return f"Transformed: {data}"
        
        # Add nodes
        graph.add_node(Node("extract", extract_function))
        graph.add_node(Node("transform", transform_function))
        
        # Add an edge
        graph.add_edge(Edge("extract", "transform"))
        
        # Get complexity score
        score = graph.get_complexity_score()
        
        if "total_score" in score and "complexity_rating" in score:
            print(f"✅ Graph functionality test passed (complexity score: {score['total_score']})")
            functionality_tests.append(True)
        else:
            print("❌ Graph functionality test failed: invalid complexity score")
            functionality_tests.append(False)
    except Exception as e:
        print(f"❌ Graph functionality test failed with error: {str(e)}")
        functionality_tests.append(False)
    
    # Test 4: Create a tool using the decorator
    try:
        from scoras.tools import tool
        
        # Create a tool using the decorator
        @tool(name="test_tool", description="A test tool")
        def test_tool(a: int, b: int) -> int:
            return a + b
        
        # Check if the tool has the expected attributes
        if (hasattr(test_tool, "tool_name") and test_tool.tool_name == "test_tool" and
            hasattr(test_tool, "tool_description") and test_tool.tool_description == "A test tool"):
            print("✅ Tool decorator functionality test passed")
            functionality_tests.append(True)
        else:
            print("❌ Tool decorator functionality test failed: missing attributes")
            functionality_tests.append(False)
    except Exception as e:
        print(f"❌ Tool decorator functionality test failed with error: {str(e)}")
        functionality_tests.append(False)
    
    # Calculate success rate
    success_rate = (sum(functionality_tests) / len(functionality_tests)) * 100 if functionality_tests else 0
    
    print("\n" + "=" * 80)
    print(f"FUNCTIONALITY TEST SUMMARY".center(80))
    print("=" * 80)
    print(f"Total tests: {len(functionality_tests)}")
    print(f"Passed: {sum(functionality_tests)}")
    print(f"Failed: {len(functionality_tests) - sum(functionality_tests)}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if all(functionality_tests):
        print("\n✅ All functionality tests passed!")
    else:
        print("\n⚠️ Some functionality tests failed. See details above.")

async def main() -> None:
    """
    Main function to test Scoras library imports and functionality.
    """
    print("\n" + "=" * 80)
    print("SCORAS LIBRARY IMPORT AND FUNCTIONALITY TEST".center(80))
    print("=" * 80 + "\n")
    
    # Print Python version
    print(f"Python version: {sys.version}")
    
    # Try to import scoras and get version
    try:
        import scoras
        print(f"Scoras version: {scoras.__version__}")
        print(f"Scoras location: {scoras.__file__}\n")
    except ImportError as e:
        print(f"❌ Failed to import scoras: {str(e)}")
        print("Please make sure the Scoras library is installed.")
        return
    except AttributeError:
        print("⚠️ Scoras imported but version information not found.")
    
    # Test core module imports
    core_components = [
        "Graph", "Node", "Edge", 
        "ScoringMixin", "ComplexityScore",
        "WorkflowGraph", "WorkflowExecutor"
    ]
    core_success, core_failed = test_import("scoras.core", core_components)
    print_results("Core Module", core_success, core_failed)
    
    # Test agents module imports
    agents_components = [
        "Agent", "Message", "Tool",
        "ExpertAgent", "CreativeAgent", "MultiAgentSystem"
    ]
    agents_success, agents_failed = test_import("scoras.agents", agents_components)
    print_results("Agents Module", agents_success, agents_failed)
    
    # Test RAG module imports
    rag_components = [
        "Document", "SimpleRAG", "ContextualRAG",
        "Retriever", "SimpleRetriever", "SemanticRetriever"
    ]
    rag_success, rag_failed = test_import("scoras.rag", rag_components)
    print_results("RAG Module", rag_success, rag_failed)
    
    # Test tools module imports
    tools_components = ["tool"]
    tools_success, tools_failed = test_import("scoras.tools", tools_components)
    print_results("Tools Module", tools_success, tools_failed)
    
    # Test protocol module imports
    mcp_components = ["MCPServer", "MCPClient", "MCPSkill"]
    mcp_success, mcp_failed = test_import("scoras.mcp", mcp_components)
    print_results("MCP Protocol Module", mcp_success, mcp_failed)
    
    a2a_components = ["A2AAgent", "A2ANetwork", "A2AHub"]
    a2a_success, a2a_failed = test_import("scoras.a2a", a2a_components)
    print_results("A2A Protocol Module", a2a_success, a2a_failed)
    
    # Calculate overall success rate for imports
    all_components = len(core_components) + len(agents_components) + len(rag_components) + len(tools_components) + len(mcp_components) + len(a2a_components)
    all_successful = len(core_success) + len(agents_success) + len(rag_success) + len(tools_success) + len(mcp_success) + len(a2a_success)
    success_rate = (all_successful / all_components) * 100 if all_components > 0 else 0
    
    print("\n" + "=" * 80)
    print(f"IMPORT TEST SUMMARY".center(80))
    print("=" * 80)
    print(f"Total components tested: {all_components}")
    print(f"Successfully imported: {all_successful}")
    print(f"Failed to import: {all_components - all_successful}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if all_successful == all_components:
        print("\n✅ All components imported successfully!")
    else:
        print("\n⚠️ Some components failed to import. See details above.")
    
    # Test functionality
    await test_functionality()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED".center(80))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
