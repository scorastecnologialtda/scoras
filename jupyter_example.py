# Copy and paste this code into your Jupyter notebook to fix the asyncio error

# First, install nest_asyncio if you don't have it
# !pip install nest_asyncio

import scoras as sc
from scoras.rag import Document, SimpleRAG
import nest_asyncio
import asyncio

# This is the key fix for Jupyter notebooks
nest_asyncio.apply()

# Create documents
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

# Method 1: Now run_sync works properly with nest_asyncio
response = rag.run_sync("What is the capital of France and what is it known for?")
print(response)

# Check the complexity score
score = rag.get_complexity_score()
print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")

# Method 2: You can also use the async version directly in Jupyter
# In a separate cell, you can do:
# 
# async def async_example():
#     response = await rag.run("What is another question?")
#     return response
#
# await async_example() 