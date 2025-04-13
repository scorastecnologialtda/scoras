# RAG Systems

Retrieval-Augmented Generation (RAG) systems enhance agents with document knowledge, allowing them to provide more accurate and informed responses. This page explains how to create and optimize RAG systems in Scoras.

## What are Scoras RAG Systems?

In Scoras, a RAG system combines:

- **Documents**: Text content with optional metadata
- **Retrievers**: Components that find relevant documents
- **Agents**: LLM-powered components that generate responses
- **Scoring**: Complexity tracking across the RAG pipeline

## Creating a Basic RAG System

The simplest RAG system connects documents to an agent:

```python
import scoras as sc
from scoras.rag import Document, SimpleRAG

# Create documents
documents = [
    Document(content="The capital of France is Paris, known as the City of Light."),
    Document(content="Paris is famous for the Eiffel Tower, built in 1889."),
    Document(content="France has a population of about 67 million people.")
]

# Create a RAG system
rag = SimpleRAG(
    agent=sc.Agent(model="openai:gpt-4o"),
    documents=documents
)

# Run the RAG system
response = rag.run_sync("What is the capital of France and what is it known for?")
print(response)
```

## Document Management

Manage documents with metadata and source tracking:

```python
from scoras.rag import Document, DocumentStore

# Create documents with metadata
documents = [
    Document(
        content="The capital of France is Paris, known as the City of Light.",
        metadata={
            "source": "geography_textbook",
            "page": 42,
            "topic": "France",
            "reliability": "high"
        }
    ),
    Document(
        content="Paris is famous for the Eiffel Tower, built in 1889.",
        metadata={
            "source": "travel_guide",
            "page": 15,
            "topic": "Paris",
            "reliability": "high"
        }
    )
]

# Create a document store
doc_store = DocumentStore()

# Add documents to the store
doc_store.add_documents(documents)

# Query documents
france_docs = doc_store.query(metadata_filter={"topic": "France"})
high_reliability_docs = doc_store.query(metadata_filter={"reliability": "high"})
```

## Document Processing

Process documents for better retrieval:

```python
from scoras.rag import TextSplitter, SemanticChunker

# Split text into chunks
text_splitter = TextSplitter(chunk_size=200, chunk_overlap=50)
chunks = text_splitter.split("Long document content...")

# Create documents from chunks
chunk_docs = [Document(content=chunk) for chunk in chunks]

# Use semantic chunking for content-aware splitting
semantic_chunker = SemanticChunker(chunk_size=200, overlap=50)
semantic_chunks = semantic_chunker.process([
    Document(content="Long document content...")
])
```

## Retrieval Methods

Choose from different retrieval methods:

```python
from scoras.rag import (
    SimpleRAG,
    SemanticRAG,
    HybridRAG,
    ContextualRAG
)

# Simple keyword-based retrieval
simple_rag = SimpleRAG(
    agent=sc.Agent(model="openai:gpt-4o"),
    documents=documents
)

# Semantic retrieval using embeddings
semantic_rag = SemanticRAG(
    agent=sc.Agent(model="openai:gpt-4o"),
    documents=documents,
    embedding_model="openai:text-embedding-3-small"
)

# Hybrid retrieval (combines keyword and semantic)
hybrid_rag = HybridRAG(
    agent=sc.Agent(model="openai:gpt-4o"),
    documents=documents,
    embedding_model="openai:text-embedding-3-small",
    keyword_weight=0.3,
    semantic_weight=0.7
)

# Contextual retrieval (adapts based on query context)
contextual_rag = ContextualRAG(
    agent=sc.Agent(model="openai:gpt-4o"),
    documents=documents,
    context_window_size=5
)
```

## Custom Retrievers

Create custom retrievers for specialized needs:

```python
from scoras.rag import Retriever

class MyCustomRetriever(Retriever):
    def __init__(self, documents, **kwargs):
        super().__init__(documents, **kwargs)
        # Custom initialization
        
    async def retrieve(self, query, top_k=3):
        # Custom retrieval logic
        # ...
        return [self.documents[0], self.documents[1]]

# Use the custom retriever
from scoras.rag import RAG

custom_rag = RAG(
    agent=sc.Agent(model="openai:gpt-4o"),
    retriever=MyCustomRetriever(documents)
)
```

## RAG with Citations

Add citations to RAG responses:

```python
from scoras.rag import CitationRAG

# Create a RAG system with citations
citation_rag = CitationRAG(
    agent=sc.Agent(model="openai:gpt-4o"),
    documents=documents,
    citation_style="inline"  # or "footnote", "endnote"
)

# Run with citations
response = citation_rag.run_sync("What is the capital of France?")
print(response)
# Output: "The capital of France is Paris [1], known as the City of Light [1].
# Sources: [1] geography_textbook, page 42"
```

## Streaming RAG

Stream responses from RAG systems:

```python
# Create a RAG system
rag = SimpleRAG(
    agent=sc.Agent(model="openai:gpt-4o"),
    documents=documents
)

# Stream the response
async for chunk in rag.stream("What is the capital of France?"):
    print(chunk, end="", flush=True)
```

## Complexity Scoring

Track and understand the complexity of your RAG system:

```python
# Create a RAG system with scoring enabled
rag = SimpleRAG(
    agent=sc.Agent(model="openai:gpt-4o"),
    documents=documents,
    enable_scoring=True
)

# Run the RAG system
response = rag.run_sync("What is the capital of France?")

# Get the complexity score
score = rag.get_complexity_score()
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")

# Get detailed breakdown
print(json.dumps(score, indent=2))
```

## RAG Agents

Use specialized agents for RAG:

```python
from scoras.agents import RAGAgent

# Create a RAG agent
rag_agent = RAGAgent(
    model="openai:gpt-4o",
    documents=documents,
    retrieval_type="semantic",
    embedding_model="openai:text-embedding-3-small"
)

# Run the agent
response = rag_agent.run_sync("What is the capital of France?")
```

## Protocol Integration

RAG systems can be exposed via MCP and A2A protocols:

```python
from scoras.mcp import create_mcp_server
from scoras.a2a import create_a2a_server

# Create an MCP server with a RAG system
mcp_server = create_mcp_server(
    name="RAGServer",
    rag_system=my_rag
)

# Create an A2A server with a RAG agent
a2a_server = create_a2a_server(
    name="RAGAgent",
    agent=rag_agent
)
```

## Next Steps

- Learn about [Agents](agents.md) that power RAG systems
- Explore [Tools](tools.md) for extending RAG capabilities
- Understand [Workflows](workflows.md) for complex RAG pipelines
- Dive into [Complexity Scoring](complexity-scoring.md) for RAG optimization
