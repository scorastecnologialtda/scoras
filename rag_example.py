import scoras as sc
from scoras.rag import Document, SimpleRAG
import nest_asyncio
import asyncio

# Apply nest_asyncio to allow nested event loops (needed for Jupyter notebooks)
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

# Option 1: Using run_sync (now works with nest_asyncio)
def use_run_sync():
    response = rag.run_sync("What is the capital of France and what is it known for?")
    print(response)
    
    # Check the complexity score
    score = rag.get_complexity_score()
    print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")

# Option 2: Using the async method directly
async def use_async_method():
    response = await rag.run("What is the capital of France and what is it known for?")
    print(response)
    
    # Check the complexity score
    score = rag.get_complexity_score()
    print(f"Complexity: {score['complexity_rating']} (Score: {score['total_score']})")

# For running in a script (not in Jupyter)
if __name__ == "__main__":
    # Either method will work now
    use_run_sync()
    
    # Or run the async function
    # asyncio.run(use_async_method()) 