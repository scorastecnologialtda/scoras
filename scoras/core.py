# Add this class to your core.py file

class Agent(ScoringMixin):
    """Represents an AI agent with complexity scoring."""
    
    def __init__(self, model: str, system_prompt: str = None):
        super().__init__()
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.tools = []
        self.add_complexity(2.0, "Agent creation")
    
    def add_tool(self, tool):
        """Add a tool to the agent."""
        self.tools.append(tool)
        self.add_complexity(1.0, f"Added tool: {getattr(tool, 'name', 'unnamed')}")
        return tool
    
    def run_sync(self, query: str) -> str:
        """Synchronous version of run method (simplified for minimal implementation)."""
        self.add_complexity(1.0, "Agent execution")
        return f"Response to: {query}"
    
    async def run(self, query: str) -> str:
        """Run the agent with a query (simplified for minimal implementation)."""
        self.add_complexity(1.0, "Agent execution")
        return f"Response to: {query}"
    
    def get_complexity_score(self) -> Dict[str, Any]:
        """Get the complexity score of the agent, including all tools."""
        base_score = super().get_complexity_score()
        
        # Add complexity from tools if they have scoring
        tool_score = 0
        for tool in self.tools:
            if hasattr(tool, 'get_complexity_score'):
                tool_score += tool.get_complexity_score()["total_score"]
            else:
                tool_score += 1.0  # Default score for tools without scoring
        
        total_score = base_score["total_score"] + tool_score
        
        # Determine complexity rating based on total score
        if total_score < 5:
            rating = "Simple"
        elif total_score < 10:
            rating = "Standard"
        elif total_score < 20:
            rating = "Complex"
        elif total_score < 50:
            rating = "Very Complex"
        else:
            rating = "Extremely Complex"
        
        return {
            "total_score": total_score,
            "complexity_rating": rating,
            "components": {
                "base": base_score["total_score"],
                "tools": tool_score
            }
        }
    
    def __repr__(self):
        return f"Agent(model='{self.model}', tools={len(self.tools)})"
