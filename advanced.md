# Advanced Examples

This page provides advanced examples of using the Scoras library. These examples demonstrate sophisticated features and complex use cases, including specialized agents, multi-agent systems, advanced RAG implementations, and complex workflows.

## Specialized Agents

This example demonstrates using specialized agent types for specific use cases:

```python
import scoras as sc
from scoras.agents import ExpertAgent, CreativeAgent, RAGAgent
from scoras.rag import Document

# Create an expert agent for a specialized domain
expert = ExpertAgent(
    model="anthropic:claude-3-opus",
    domain="medicine",
    expertise_level="advanced",
    system_prompt="You are a medical expert specializing in cardiology.",
    enable_scoring=True
)

# Run the expert agent
expert_response = expert.run_sync("Explain the difference between systolic and diastolic heart failure.")
print("Expert Agent Response:")
print(expert_response)
print(f"Complexity: {expert.get_complexity_score()['complexity_rating']}")

# Create a creative agent for content generation
creative = CreativeAgent(
    model="openai:gpt-4o",
    creativity_level="high",
    system_prompt="You are a creative writer with a unique and imaginative style.",
    enable_scoring=True
)

# Run the creative agent
creative_response = creative.run_sync("Write a short poem about artificial intelligence.")
print("\nCreative Agent Response:")
print(creative_response)
print(f"Complexity: {creative.get_complexity_score()['complexity_rating']}")

# Create a RAG agent with built-in retrieval
documents = [
    Document(content="The capital of France is Paris, known as the City of Light."),
    Document(content="Paris is famous for the Eiffel Tower, built in 1889."),
    Document(content="France has a population of about 67 million people.")
]

rag_agent = RAGAgent(
    model="gemini:gemini-pro",
    documents=documents,
    retrieval_type="semantic",
    system_prompt="You are a knowledgeable assistant that provides accurate information based on your knowledge base.",
    enable_scoring=True
)

# Run the RAG agent
rag_response = rag_agent.run_sync("What is Paris known for?")
print("\nRAG Agent Response:")
print(rag_response)
print(f"Complexity: {rag_agent.get_complexity_score()['complexity_rating']}")
```

## Multi-Agent System

This example shows how to create a multi-agent system for collaborative problem-solving:

```python
import scoras as sc
from scoras.agents import MultiAgentSystem

# Create individual agents
researcher = sc.Agent(
    model="openai:gpt-4o",
    system_prompt="You are a research specialist. Your role is to find and analyze information on various topics.",
    enable_scoring=True
)

writer = sc.Agent(
    model="anthropic:claude-3-opus",
    system_prompt="You are a writing expert. Your role is to create clear, engaging, and well-structured content.",
    enable_scoring=True
)

fact_checker = sc.Agent(
    model="gemini:gemini-pro",
    system_prompt="You are a fact-checking specialist. Your role is to verify information for accuracy and reliability.",
    enable_scoring=True
)

# Create a multi-agent system
system = MultiAgentSystem(
    agents={
        "researcher": researcher,
        "writer": writer,
        "fact_checker": fact_checker
    },
    coordinator_prompt="""
    You are a coordinator that manages a team of specialized agents:
    - Researcher: Finds and analyzes information
    - Writer: Creates well-structured content
    - Fact-checker: Verifies information accuracy
    
    Your job is to:
    1. Break down tasks into appropriate sub-tasks for each agent
    2. Send sub-tasks to the right agents
    3. Integrate their outputs into a cohesive final result
    4. Ensure the final output is accurate, well-written, and comprehensive
    """,
    enable_scoring=True
)

# Run the system
result = system.run_sync("Create a well-researched article about quantum computing, focusing on recent breakthroughs and future applications.")
print("Multi-Agent System Result:")
print(result)

# Get complexity scores
system_score = system.get_complexity_score()
researcher_score = researcher.get_complexity_score()
writer_score = writer.get_complexity_score()
fact_checker_score = fact_checker.get_complexity_score()

print("\nComplexity Scores:")
print(f"System: {system_score['complexity_rating']} (Score: {system_score['total_score']})")
print(f"Researcher: {researcher_score['complexity_rating']} (Score: {researcher_score['total_score']})")
print(f"Writer: {writer_score['complexity_rating']} (Score: {writer_score['total_score']})")
print(f"Fact-checker: {fact_checker_score['complexity_rating']} (Score: {fact_checker_score['total_score']})")
```

## Advanced RAG with Semantic Chunking

This example demonstrates advanced RAG techniques with semantic chunking:

```python
import scoras as sc
from scoras.rag import Document, SemanticChunker, ContextualRAG

# Sample long document
long_document = """
# Quantum Computing: An Overview

Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. The devices that perform quantum computations are known as quantum computers.

## Quantum Bits (Qubits)

Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits. A qubit can exist in a state of 0, 1, or any quantum superposition of these states. This allows quantum computers to process a vast number of possibilities simultaneously.

## Quantum Superposition

Superposition refers to the quantum phenomenon where a quantum system can exist in multiple states simultaneously. When measured, a qubit in superposition collapses to either 0 or 1 according to probability.

## Quantum Entanglement

Entanglement is a quantum phenomenon where pairs or groups of particles are generated or interact in ways such that the quantum state of each particle cannot be described independently of the others. Measuring one particle instantaneously affects its entangled partners, regardless of the distance separating them.

## Quantum Gates

Quantum gates are the building blocks of quantum circuits. They perform operations on qubits, similar to how classical logic gates operate on bits. Common quantum gates include the Hadamard gate, Pauli gates, and CNOT gate.

## Quantum Algorithms

Quantum algorithms are designed to run on quantum computers and can solve certain problems much faster than classical algorithms. Notable examples include:

- Shor's algorithm for factoring large numbers
- Grover's algorithm for searching unsorted databases
- Quantum Fourier Transform
- Quantum machine learning algorithms

## Applications of Quantum Computing

Quantum computing has potential applications in various fields:

1. Cryptography: Breaking and creating more secure encryption
2. Drug discovery: Simulating molecular structures
3. Optimization problems: Finding optimal solutions in complex systems
4. Machine learning: Enhancing AI capabilities
5. Material science: Designing new materials with specific properties
6. Financial modeling: Analyzing complex financial data

## Challenges in Quantum Computing

Despite its potential, quantum computing faces several challenges:

- Decoherence: Quantum states are fragile and can collapse due to environmental interactions
- Error correction: Quantum error correction is complex and resource-intensive
- Scalability: Building large-scale quantum computers is technically challenging
- Hardware limitations: Current quantum computers have limited qubit counts and high error rates

## Current State of Quantum Computing

As of 2025, quantum computers are still in the early stages of development. Companies like IBM, Google, Microsoft, and several startups are working on building increasingly powerful quantum computers. Quantum supremacy (the point where quantum computers outperform classical computers for specific tasks) has been demonstrated, but practical quantum advantage for real-world problems remains a goal for the future.
"""

# Create a semantic chunker
chunker = SemanticChunker(
    chunk_size=200,
    overlap=50,
    embedding_model="openai:text-embedding-3-small"
)

# Process the document
document = Document(content=long_document)
chunks = chunker.process([document])

print(f"Created {len(chunks)} semantic chunks")
for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks
    print(f"\nChunk {i+1} (first 100 chars): {chunk.content[:100]}...")

# Create a contextual RAG system
rag = ContextualRAG(
    agent=sc.Agent(model="openai:gpt-4o"),
    documents=chunks,
    context_window_size=5,
    enable_scoring=True
)

# Run queries with context adaptation
query1 = "What are qubits and how do they differ from classical bits?"
response1 = rag.run_sync(query1)
print(f"\nQuery: {query1}")
print(f"Response: {response1}")

# The next query builds on the context of the previous one
query2 = "How does superposition relate to this?"
response2 = rag.run_sync(query2)
print(f"\nQuery: {query2}")
print(f"Response: {response2}")

# Get complexity score
score = rag.get_complexity_score()
print(f"\nComplexity: {score['complexity_rating']} (Score: {score['total_score']})")
```

## Complex Workflow with Conditional Branching

This example demonstrates a complex workflow with conditional branching:

```python
import scoras as sc
from pydantic import BaseModel
import asyncio
import json

# Define the state model
class QueryState(BaseModel):
    query: str
    query_type: str = ""
    factual_answer: str = ""
    creative_answer: str = ""
    final_answer: str = ""

# Define node functions
async def classify_query(state):
    """Determine if the query requires a factual or creative response."""
    factual_keywords = ["what", "who", "when", "where", "how", "why", "explain", "define"]
    creative_keywords = ["imagine", "create", "write", "design", "story", "poem", "creative"]
    
    query_lower = state.query.lower()
    
    # Check for factual keywords
    if any(keyword in query_lower for keyword in factual_keywords):
        return {"query_type": "factual"}
    
    # Check for creative keywords
    elif any(keyword in query_lower for keyword in creative_keywords):
        return {"query_type": "creative"}
    
    # Default to factual
    else:
        return {"query_type": "factual"}

async def generate_factual_answer(state):
    """Generate a factual answer using a knowledge-focused agent."""
    factual_agent = sc.Agent(
        model="anthropic:claude-3-opus",
        system_prompt="You are a factual assistant that provides accurate, concise information."
    )
    
    response = await factual_agent.run(f"Provide a factual answer to: {state.query}")
    return {"factual_answer": response}

async def generate_creative_answer(state):
    """Generate a creative answer using a creativity-focused agent."""
    creative_agent = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="You are a creative assistant that provides imaginative, engaging responses."
    )
    
    response = await creative_agent.run(f"Provide a creative response to: {state.query}")
    return {"creative_answer": response}

async def combine_answers(state):
    """Combine factual and creative answers when both are available."""
    if state.factual_answer and state.creative_answer:
        return {"final_answer": f"Factual perspective: {state.factual_answer}\n\nCreative perspective: {state.creative_answer}"}
    elif state.factual_answer:
        return {"final_answer": state.factual_answer}
    elif state.creative_answer:
        return {"final_answer": state.creative_answer}
    else:
        return {"final_answer": "No answer could be generated."}

async def format_factual_answer(state):
    """Format the factual answer as the final answer."""
    return {"final_answer": state.factual_answer}

async def format_creative_answer(state):
    """Format the creative answer as the final answer."""
    return {"final_answer": state.creative_answer}

# Create a workflow graph
graph = sc.WorkflowGraph(
    state_type=QueryState,
    enable_scoring=True
)

# Add nodes
graph.add_node("start", lambda s: s, "simple")
graph.add_node("classify", classify_query, "standard")
graph.add_node("factual", generate_factual_answer, "complex")
graph.add_node("creative", generate_creative_answer, "complex")
graph.add_node("combine", combine_answers, "standard")
graph.add_node("format_factual", format_factual_answer, "simple")
graph.add_node("format_creative", format_creative_answer, "simple")
graph.add_node("end", lambda s: s, "simple")

# Add edges with conditions
graph.add_edge("start", "classify")

# Conditional edges based on query type
graph.add_edge(
    "classify", "factual", 
    condition=lambda s: s.query_type == "factual"
)
graph.add_edge(
    "classify", "creative", 
    condition=lambda s: s.query_type == "creative"
)
graph.add_edge(
    "classify", "factual", 
    condition=lambda s: s.query_type == "both"
)
graph.add_edge(
    "classify", "creative", 
    condition=lambda s: s.query_type == "both"
)

# Edges from processing to formatting
graph.add_edge(
    "factual", "format_factual", 
    condition=lambda s: s.query_type == "factual"
)
graph.add_edge(
    "creative", "format_creative", 
    condition=lambda s: s.query_type == "creative"
)
graph.add_edge(
    "factual", "combine", 
    condition=lambda s: s.query_type == "both"
)
graph.add_edge(
    "creative", "combine", 
    condition=lambda s: s.query_type == "both"
)

# Final edges to end
graph.add_edge("format_factual", "end")
graph.add_edge("format_creative", "end")
graph.add_edge("combine", "end")

# Add error handling
async def handle_error(state, error):
    """Handle errors in the workflow."""
    return {"final_answer": f"An error occurred: {str(error)}"}

graph.add_error_handler("factual", handle_error)
graph.add_error_handler("creative", handle_error)

# Compile the graph
workflow = graph.compile()

# Run the workflow with different queries
async def run_examples():
    # Factual query
    factual_query = "What is quantum computing?"
    factual_result = await workflow.run(QueryState(query=factual_query))
    print(f"Query: {factual_query}")
    print(f"Query Type: {factual_result.query_type}")
    print(f"Answer: {factual_result.final_answer[:200]}...\n")
    
    # Creative query
    creative_query = "Write a short poem about artificial intelligence."
    creative_result = await workflow.run(QueryState(query=creative_query))
    print(f"Query: {creative_query}")
    print(f"Query Type: {creative_result.query_type}")
    print(f"Answer: {creative_result.final_answer[:200]}...\n")
    
    # Get complexity score
    score = graph.get_complexity_score()
    print(f"Workflow Complexity: {score['complexity_rating']} (Score: {score['total_score']})")
    print("Detailed score breakdown:")
    print(json.dumps(score, indent=2))

# Run the examples
asyncio.run(run_examples())
```

## Tool Chains and Tool Routers

This example demonstrates advanced tool management with chains and routers:

```python
import scoras as sc
from scoras.tools import ToolChain, ToolRouter
import asyncio

# Define individual tools
@sc.tool(name="calculator", description="Perform calculations", complexity="simple")
async def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic arithmetic operations."""
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

@sc.tool(name="temperature_converter", description="Convert between temperature units", complexity="simple")
async def temperature_converter(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between Celsius,
(Content truncated due to size limit. Use line ranges to read in chunks)