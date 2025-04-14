"""
Advanced example of the Scoras library.

This script demonstrates advanced features of the Scoras library,
including specialized agents, multi-agent systems, complex tools,
and detailed workflow complexity scoring.

Author: Anderson L. Amaral
"""

import asyncio
from typing import Dict, List, Any
from pydantic import BaseModel, Field

import scoras as sc
from scoras.tools import create_http_tool, create_python_tool, ToolChain, ToolRouter


# Define models for structured data
class ResearchResult(BaseModel):
    query: str = Field(..., description="Original search query")
    findings: List[str] = Field(..., description="Key findings from research")
    sources: List[str] = Field(..., description="Sources of information")


class AnalysisResult(BaseModel):
    text: str = Field(..., description="Text being analyzed")
    sentiment: str = Field(..., description="Sentiment of the text (positive, negative, neutral)")
    keywords: List[str] = Field(..., description="Key words or phrases identified")
    summary: str = Field(..., description="Summary of the text")


# Functions for tools
def analyze_text(text: str) -> Dict[str, Any]:
    """Analyzes text and returns information about it."""
    # Simulated implementation
    import random
    
    sentiments = ["positive", "negative", "neutral"]
    words = text.lower().split()
    keywords = list(set([p for p in words if len(p) > 4]))[:5]
    
    return {
        "text": text,
        "sentiment": random.choice(sentiments),
        "keywords": keywords,
        "summary": f"This is a simulated summary of: {text[:50]}..."
    }


async def search_web(query: str, num_results: int = 3) -> Dict[str, Any]:
    """Searches for information on the web."""
    # Simulated implementation
    await asyncio.sleep(1)  # Simulate network delay
    
    results = [
        f"Result {i+1} for '{query}': Simulated information about {query}."
        for i in range(num_results)
    ]
    
    sources = [
        f"https://example.com/result{i+1}" for i in range(num_results)
    ]
    
    return {
        "query": query,
        "findings": results,
        "sources": sources
    }


async def main():
    # Create specialized agents
    expert_agent = sc.agents.ExpertAgent(
        model="openai:gpt-4o",
        domain="science",
        expertise_level="advanced"
    )
    
    creative_agent = sc.agents.CreativeAgent(
        model="openai:gpt-4o",
        creative_mode="experimental",
        style_guide="Use vivid imagery and metaphors."
    )
    
    # Create tools
    analysis_tool = create_python_tool(
        function=analyze_text,
        name="analyze_text",
        description="Analyzes text and returns information about it"
    )
    
    search_tool = create_python_tool(
        function=search_web,
        name="search_web",
        description="Searches for information on the web",
        is_async=True
    )
    
    # Create a tool chain
    tool_chain = ToolChain(
        tools=[search_tool, analysis_tool],
        name="research_and_analyze",
        description="Searches for information and then analyzes it"
    )
    
    # Create a tool router
    def router_function(inputs):
        if "query" in inputs:
            return "search"
        elif "text" in inputs:
            return "analyze"
        else:
            return "chain"
    
    router = ToolRouter(
        tools={
            "search": search_tool,
            "analyze": analysis_tool,
            "chain": tool_chain.to_tool()
        },
        router_function=router_function,
        name="smart_router"
    )
    
    # Create agents with tools
    researcher_agent = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="You are a research assistant specialized in finding information.",
        tools=[search_tool],
        result_type=ResearchResult
    )
    
    analyst_agent = sc.Agent(
        model="openai:gpt-4o",
        system_prompt="You are an analyst specialized in text analysis.",
        tools=[analysis_tool],
        result_type=AnalysisResult
    )
    
    # Create a multi-agent system
    system = sc.agents.MultiAgentSystem({
        "expert": expert_agent,
        "creative": creative_agent,
        "researcher": researcher_agent,
        "analyst": analyst_agent
    })
    
    # Run individual agents
    expert_response = await system.run("expert", "Explain quantum entanglement in simple terms.")
    print(f"Expert response: {expert_response}\n")
    
    creative_response = await system.run("creative", "Write a short poem about artificial intelligence.")
    print(f"Creative response: {creative_response}\n")
    
    # Run agents with tools
    research_result = await system.run("researcher", "What are the latest developments in quantum computing?")
    print(f"Research query: {research_result.query}")
    print(f"Research findings: {len(research_result.findings)} items found")
    for i, finding in enumerate(research_result.findings, 1):
        print(f"  {i}. {finding}")
    print(f"Sources: {research_result.sources}\n")
    
    # Run a sequence of agents
    print("Running agent sequence: researcher -> analyst")
    results = await system.run_sequence(
        "Analyze the latest developments in quantum computing",
        ["researcher", "analyst"]
    )
    
    # The final result is from the analyst
    analysis = results[-1]
    print(f"Analysis of: {analysis.text[:50]}...")
    print(f"Sentiment: {analysis.sentiment}")
    print(f"Keywords: {', '.join(analysis.keywords)}")
    print(f"Summary: {analysis.summary}\n")
    
    # Create an agent team
    team = sc.agents.AgentTeam(
        coordinator=expert_agent,
        specialists={
            "researcher": researcher_agent,
            "analyst": analyst_agent,
            "creative": creative_agent
        }
    )
    
    # Run the agent team
    team_response = await team.run(
        "I need comprehensive information about quantum computing, analyzed and presented creatively."
    )
    print(f"Team response: {team_response}\n")
    
    # Create a workflow graph
    class ResearchState(BaseModel):
        topic: str
        research_results: List[str] = []
        analysis_results: Dict[str, Any] = {}
        final_report: str = ""
    
    graph = sc.Graph(state_type=ResearchState)
    
    # Define node functions
    async def research_topic(state: ResearchState) -> Dict[str, Any]:
        print(f"Researching: {state.topic}")
        # Simulate research
        results = await search_web(state.topic)
        return {"research_results": results["findings"]}
    
    def analyze_results(state: ResearchState) -> Dict[str, Any]:
        print(f"Analyzing {len(state.research_results)} research results")
        # Combine research results into a single text
        combined_text = "\n".join(state.research_results)
        # Analyze the text
        analysis = analyze_text(combined_text)
        return {"analysis_results": analysis}
    
    def generate_report(state: ResearchState) -> Dict[str, Any]:
        print("Generating final report")
        # Create a report based on analysis
        report = f"REPORT ON: {state.topic}\n\n"
        report += f"SUMMARY: {state.analysis_results['summary']}\n\n"
        report += "KEY FINDINGS:\n"
        for result in state.research_results:
            report += f"- {result}\n"
        report += f"\nSENTIMENT: {state.analysis_results['sentiment']}\n"
        report += f"KEYWORDS: {', '.join(state.analysis_results['keywords'])}"
        
        return {"final_report": report}
    
    # Add nodes to the graph
    graph.add_node("research", research_topic, "complex")
    graph.add_node("analyze", analyze_results)
    graph.add_node("report", generate_report)
    
    # Add edges
    graph.add_edge("start", "research")
    graph.add_edge("research", "analyze")
    graph.add_edge("analyze", "report")
    graph.add_edge("report", "end")
    
    # Compile and run the graph
    compiled_graph = graph.compile()
    final_state = await compiled_graph.run({"topic": "quantum computing"})
    
    print("\nWorkflow Graph Results:")
    print(f"Final report:\n{final_state.final_report}\n")
    
    # Get complexity scores for all components
    print("\nCOMPLEXITY SCORES:")
    
    expert_score = expert_agent.get_complexity_score()
    print(f"Expert Agent: {expert_score['complexity_rating']} (Score: {expert_score['total_score']:.1f})")
    
    researcher_score = researcher_agent.get_complexity_score()
    print(f"Researcher Agent: {researcher_score['complexity_rating']} (Score: {researcher_score['total_score']:.1f})")
    
    system_score = system.get_complexity_score()
    print(f"Multi-Agent System: {system_score['complexity_rating']} (Score: {system_score['total_score']:.1f})")
    
    team_score = team.get_complexity_score()
    print(f"Agent Team: {team_score['complexity_rating']} (Score: {team_score['total_score']:.1f})")
    
    chain_score = tool_chain.get_complexity_score()
    print(f"Tool Chain: {chain_score['complexity_rating']} (Score: {chain_score['total_score']:.1f})")
    
    router_score = router.get_complexity_score()
    print(f"Tool Router: {router_score['complexity_rating']} (Score: {router_score['total_score']:.1f})")
    
    graph_score = graph.get_complexity_score()
    print(f"Workflow Graph: {graph_score['complexity_rating']} (Score: {graph_score['total_score']:.1f})")
    
    # Visualize the graph
    print("\nGraph visualization:")
    print(graph.visualize(format="text"))
    
    print("\nMermaid diagram:")
    print(graph.visualize(format="mermaid"))


if __name__ == "__main__":
    asyncio.run(main())
