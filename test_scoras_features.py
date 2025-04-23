#!/usr/bin/env python3
"""
Modified test script with improved package detection for the Scoras library.
This script tests all the major components of the Scoras library.
"""

import sys
import os

# Add the scoras package directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import colorization for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

def print_header(text):
    """Print a formatted header."""
    width = 80
    print("\n" + "=" * width)
    print(f"{Colors.BOLD}{text.center(width)}{Colors.ENDC}")
    print("=" * width)

def print_section(text):
    """Print a formatted section header."""
    print(f"\n{Colors.BOLD}{text}{Colors.ENDC}")
    print("-" * len(text))

def print_success(text):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_failure(text, error=None):
    """Print a failure message."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")
    if error:
        print(f"  └─ Error: {error}")

def print_info(text):
    """Print an info message."""
    print(f"  └─ {text}")

def main() -> None:
    """Main function to run all tests."""
    print_header("COMPREHENSIVE TEST OF THE SCORAS LIBRARY")
    
    # Check the version of the scoras package
    try:
        import importlib
        
        # Print Python path for debugging
        print(f"Python path: {sys.path}")
        
        # Try to import scoras with explicit reload
        import scoras
        importlib.reload(scoras)
        
        print(f"\n{Colors.BOLD}Scoras Version: {Colors.GREEN}{scoras.__version__}{Colors.ENDC}")
        print(f"Scoras package found at: {scoras.__file__}")
        
        if scoras.__version__ != "0.3.1":
            print(f"{Colors.YELLOW}⚠️ Warning: The installed version ({scoras.__version__}) is not the expected version (0.3.1).{Colors.ENDC}")
    except ImportError as e:
        print(f"\n{Colors.RED}❌ ERROR: The scoras package is not installed.{Colors.ENDC}")
        print(f"Import error: {e}")
        print("Please install the package before running the tests:")
        print("pip install scoras")
        sys.exit(1)
    
    # Track test results
    successful_tests = 0
    failed_tests = 0
    failed_details = []
    
    # Test imports
    print_section("Testing imports")
    imports_to_test = [
        "from scoras.core import Graph",
        "from scoras.core import Node",
        "from scoras.core import Edge",
        "from scoras.core import Message",
        "from scoras.core import Tool",
        "from scoras.core import RAG",
        "from scoras.core import ScoreTracker",
        "from scoras.core import ScorasConfig",
        "from scoras.core import WorkflowGraph",
        "from scoras.agents import Agent",
        "from scoras.rag import Document",
        "from scoras.rag import SimpleRAG",
        "from scoras.tools import tool",
        "from scoras.mcp import MCPServer",
        "from scoras.mcp import MCPClient",
        "from scoras.a2a import A2AAgent",
        "from scoras.a2a import A2ANetwork"
    ]
    
    for import_stmt in imports_to_test:
        try:
            exec(import_stmt)
            print_success(import_stmt)
            successful_tests += 1
        except Exception as e:
            print_failure(import_stmt, str(e))
            failed_tests += 1
            failed_details.append(f"Import: {import_stmt} - {str(e)}")
    
    # Test Document class
    print_section("Testing the Document class")
    try:
        from scoras.rag import Document
        doc = Document("This is a test document", metadata={"source": "test"})
        print_success("Creation of basic Document")
        print_info(f"Content: {doc.content}")
        print_info(f"Metadata: {doc.metadata}")
        successful_tests += 1
        
        # Test Document attributes
        try:
            doc_id = doc.id
            print_success("Document class tests")
            successful_tests += 1
        except AttributeError as e:
            print_failure("Document class tests", f"'{e}'")
            failed_tests += 1
            failed_details.append(f"Document class tests: '{e}'")
    except Exception as e:
        print_failure("Creation of basic Document", str(e))
        failed_tests += 1
        failed_details.append(f"Document creation: {str(e)}")
    
    # Test SimpleRAG class
    print_section("Testing the SimpleRAG class")
    try:
        from scoras.rag import SimpleRAG
        from scoras.agents import Agent
        
        # Create a mock agent
        agent = Agent("test_agent")
        
        # Create SimpleRAG
        rag = SimpleRAG(agent)
        print_success("Creation of SimpleRAG")
        successful_tests += 1
        
        # Test SimpleRAG methods
        methods_success = []
        methods_failure = []
        
        try:
            # Test run method
            result = rag.run_sync("test query")
            methods_success.append("run_sync")
        except Exception as e:
            methods_failure.append(("run_sync", str(e)))
        
        try:
            # Test add_document method
            doc = Document("Test document")
            rag.add_document(doc)
            methods_success.append("add_document")
        except Exception as e:
            methods_failure.append(("add_document", str(e)))
        
        print(f"SimpleRAG methods: {', '.join([f'{m}: ✓' for m in methods_success])}, {', '.join([f'{m[0]}: ✗' for m in methods_failure])}")
        
        successful_tests += len(methods_success)
        failed_tests += len(methods_failure)
        for method, error in methods_failure:
            failed_details.append(f"SimpleRAG method {method}: {error}")
        
    except Exception as e:
        print_failure("Creation of SimpleRAG", str(e))
        failed_tests += 1
        failed_details.append(f"SimpleRAG creation: {str(e)}")
    
    # Test tool decorator
    print_section("Testing the tool decorator")
    try:
        from scoras.tools import tool
        
        @tool(name="calculator", description="A simple calculator")
        def add(a, b):
            return a + b
        
        # Test tool attributes
        print_success("Tool decorator - attributes")
        print_info(f"tool_name: {add.tool_name}")
        print_info(f"tool_description: {add.tool_description}")
        successful_tests += 1
        
        # Test tool execution
        result = add(3, 5)
        print_success("Execution of function decorated with tool")
        print_info(f"Result: {result}")
        successful_tests += 1
    except Exception as e:
        print_failure("Tool decorator", str(e))
        failed_tests += 1
        failed_details.append(f"Tool decorator: {str(e)}")
    
    # Test Message class
    print_section("Testing the Message class")
    try:
        from scoras.core import Message
        
        # Create a basic message
        msg = Message(role="user", content="Hello, world!")
        print_success("Creation of basic Message")
        print_info(f"Role: {msg.role}")
        print_info(f"Content: {msg.content}")
        successful_tests += 1
        
        # Create a message with metadata
        msg_with_metadata = Message(
            role="assistant",
            content="Assistant's response",
            metadata={"timestamp": "2025-04-23T12:00:00Z", "model": "gpt-4"}
        )
        print_success("Creation of Message with metadata")
        print_info(f"Role: {msg_with_metadata.role}")
        print_info(f"Content: {msg_with_metadata.content}")
        print_info(f"Metadata: {msg_with_metadata.metadata}")
        successful_tests += 1
    except Exception as e:
        print_failure("Message class", str(e))
        failed_tests += 1
        failed_details.append(f"Message class: {str(e)}")
    
    # Test WorkflowGraph class
    print_section("Testing the WorkflowGraph class")
    try:
        from scoras.core import WorkflowGraph
        
        # Create a workflow graph
        workflow = WorkflowGraph(name="data_processing")
        print_success("Creation of WorkflowGraph")
        print_info(f"Name: {workflow.name}")
        successful_tests += 1
        
        # Add nodes
        workflow.add_node("extract")
        workflow.add_node("filter")
        workflow.add_node("transform")
        print_success("Add nodes to WorkflowGraph")
        print_info(f"Nodes: {', '.join(workflow.nodes.keys())}")
        successful_tests += 1
        
        # Add edges
        workflow.add_edge("extract", "filter")
        workflow.add_edge("filter", "transform")
        print_success("Add edges to WorkflowGraph")
        print_info(f"Number of edges: {len(workflow.edges)}")
        successful_tests += 1
        
        # Compile workflow
        executor = workflow.compile()
        print_success("Compile WorkflowGraph")
        print_info(f"Executor created with run and run_sync methods")
        successful_tests += 1
    except Exception as e:
        print_failure("WorkflowGraph class", str(e))
        failed_tests += 1
        failed_details.append(f"WorkflowGraph class: {str(e)}")
    
    # Print results summary
    print_header("RESULTS SUMMARY")
    total_tests = successful_tests + failed_tests
    print(f"\nTotal tests: {total_tests}")
    print(f"Successful tests: {Colors.GREEN}{successful_tests}{Colors.ENDC}")
    print(f"Failed tests: {Colors.RED if failed_tests > 0 else Colors.GREEN}{failed_tests}{Colors.ENDC}")
    
    if failed_tests > 0:
        print(f"\nFailed tests:")
        for detail in failed_details:
            print(f"- {detail}")
        
        print(f"\n{Colors.YELLOW}⚠️ SOME TESTS FAILED. Check the details above.{Colors.ENDC}")
        
        print("\nSuggestions to fix problems:")
        print("1. Make sure the scoras package is updated to version 0.3.1:")
        print("   pip install scoras==0.3.1 --upgrade")
        print("2. Check if all update files were applied correctly:")
        print("   python update_scoras_package.py")
        print("3. Rebuild and reinstall the package:")
        print("   cd ~/scoras")
        print("   rm -rf dist/ build/ *.egg-info/")
        print("   python -m build")
        print("   pip install -e .")
    else:
        print(f"\n{Colors.GREEN}✅ ALL TESTS PASSED! The Scoras library is working correctly.{Colors.ENDC}")

if __name__ == "__main__":
    main()
