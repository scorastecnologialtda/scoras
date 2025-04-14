"""
Example of a basic usage of the Scoras library.

This script demonstrates how to create and use a simple agent with the Scoras library,
including the scoring system for measuring workflow complexity.

Author: Anderson L. Amaral
"""

import asyncio
from pydantic import BaseModel, Field

import scoras as sc
from scoras.tools import register_tool


# Define a result model
class WeatherResult(BaseModel):
    city: str = Field(..., description="Name of the city")
    temperature: float = Field(..., description="Temperature in Celsius")
    condition: str = Field(..., description="Current weather condition")
    humidity: float = Field(..., description="Humidity percentage")


# Create a tool
@register_tool(name="calculate", description="Performs mathematical calculations", complexity="simple")
async def calculate(expression: str) -> float:
    """Calculates the result of a mathematical expression."""
    # Safe implementation using ast for evaluation
    import ast
    import operator
    
    # Define allowed operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    
    def eval_expr(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        else:
            raise ValueError(f"Unsupported operation: {type(node).__name__}")
    
    try:
        return eval_expr(ast.parse(expression, mode='eval').body)
    except Exception as e:
        raise ValueError(f"Error calculating expression: {str(e)}")


async def main():
    # Create a simple agent
    agent = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="You are a helpful and concise assistant."
    )
    
    # Run the agent
    response = await agent.run("What is the capital of France?")
    print(f"Response: {response}")
    
    # Check the complexity score
    score = agent.get_complexity_score()
    print(f"Agent complexity: {score['complexity_rating']} (Score: {score['total_score']:.1f})")
    
    # Create an agent with typed result
    weather_agent = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="You are an assistant specialized in weather information.",
        result_type=WeatherResult
    )
    
    # Run the agent with typed result
    result = await weather_agent.run("How is the weather in New York today?")
    print(f"City: {result.city}")
    print(f"Temperature: {result.temperature}°C")
    print(f"Condition: {result.condition}")
    print(f"Humidity: {result.humidity}%")
    
    # Create an agent with tools
    calculator_agent = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="You are a math assistant.",
        tools=[calculate]
    )
    
    # Run the agent with tools
    response = await calculator_agent.run("What is 15 * 7 + 22?")
    print(f"Calculation result: {response}")
    
    # Check the complexity score
    score = calculator_agent.get_complexity_score()
    print(f"Calculator agent complexity: {score['complexity_rating']} (Score: {score['total_score']:.1f})")
    
    # Create a simple RAG system
    from scoras.rag import Document, create_rag_system
    
    documents = [
        Document(content="France is a country in Western Europe with a population of 67 million."),
        Document(content="Paris is the capital of France, founded in 3rd century BC."),
        Document(content="Lyon is the third-largest city in France, with about 500,000 inhabitants.")
    ]
    
    rag_system = create_rag_system(
        agent=sc.Agent(model="openai:gpt-4o"),
        documents=documents
    )
    
    # Run the RAG system
    response = await rag_system.run("What is the capital of France and when was it founded?")
    print(f"RAG response: {response}")
    
    # Check the complexity score
    score = rag_system.get_complexity_score()
    print(f"RAG system complexity: {score['complexity_rating']} (Score: {score['total_score']:.1f})")
    
    # Create a workflow graph
    class State(BaseModel):
        message: str
        counter: int = 0
    
    graph = sc.Graph(state_type=State)
    
    # Define nodes
    def process(state: State) -> dict:
        return {"message": f"Processed: {state.message}"}
    
    def increment(state: State) -> dict:
        return {"counter": state.counter + 1}
    
    def check(state: State) -> bool:
        return state.counter >= 3
    
    # Add nodes and edges
    graph.add_node("process", process)
    graph.add_node("increment", increment)
    graph.add_edge("start", "process")
    graph.add_edge("process", "increment")
    graph.add_edge("increment", "process", lambda s: s.counter < 3)
    graph.add_edge("increment", "end", check)
    
    # Compile and run the graph
    compiled_graph = graph.compile()
    result = await compiled_graph.run({"message": "Hello, world!"})
    
    print(f"Graph result: {result}")
    
    # Check the complexity score
    score = graph.get_complexity_score()
    print(f"Graph complexity: {score['complexity_rating']} (Score: {score['total_score']:.1f})")
    
    # Visualize the graph
    print("\nGraph visualization:")
    print(graph.visualize(format="text"))


if __name__ == "__main__":
    asyncio.run(main())
