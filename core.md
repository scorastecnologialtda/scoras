# API Reference

This section provides detailed API documentation for the Scoras library. Each module's API is documented separately for easier reference.

## Core API

The Core API includes the fundamental components of Scoras, including the base classes and utilities.

### Agent Class

```python
class Agent:
    def __init__(
        self, 
        model: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Tool]] = None,
        enable_scoring: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Scoras Agent.
        
        Args:
            model: Model identifier in format "provider:model_name" (e.g., "openai:gpt-4o")
            system_prompt: Optional system prompt to guide the agent's behavior
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate in responses
            tools: Optional list of tools available to the agent
            enable_scoring: Whether to track complexity scoring
            metadata: Optional metadata for the agent
        """
        
    async def run(self, query: Optional[str] = None) -> str:
        """
        Run the agent asynchronously.
        
        Args:
            query: Optional query to process. If None, uses conversation history.
            
        Returns:
            Agent's response as a string
        """
        
    def run_sync(self, query: Optional[str] = None) -> str:
        """
        Run the agent synchronously.
        
        Args:
            query: Optional query to process. If None, uses conversation history.
            
        Returns:
            Agent's response as a string
        """
        
    async def stream(self, query: Optional[str] = None) -> AsyncIterator[str]:
        """
        Stream the agent's response asynchronously.
        
        Args:
            query: Optional query to process. If None, uses conversation history.
            
        Yields:
            Chunks of the agent's response as they are generated
        """
        
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Role of the message sender (e.g., "user", "assistant")
            content: Content of the message
        """
        
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        
    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent.
        
        Args:
            tool: Tool to add
        """
        
    def get_complexity_score(self) -> Dict[str, Any]:
        """
        Get the complexity score for the agent.
        
        Returns:
            Dictionary containing complexity score information
        """
```

### Tool Decorator

```python
def tool(
    name: str, 
    description: str, 
    complexity: str = "standard"
) -> Callable:
    """
    Decorator to create a tool from a function.
    
    Args:
        name: Name of the tool
        description: Description of what the tool does
        complexity: Complexity level ("simple", "standard", "complex")
        
    Returns:
        Decorated function as a Tool
    """
```

### WorkflowGraph Class

```python
class WorkflowGraph:
    def __init__(
        self, 
        state_type: Type[BaseModel],
        enable_scoring: bool = True
    ):
        """
        Initialize a workflow graph.
        
        Args:
            state_type: Pydantic model class defining the workflow state
            enable_scoring: Whether to track complexity scoring
        """
        
    def add_node(
        self, 
        name: str, 
        function: Callable, 
        complexity: str = "standard"
    ) -> None:
        """
        Add a node to the workflow graph.
        
        Args:
            name: Unique name for the node
            function: Function to execute at this node
            complexity: Complexity level ("simple", "standard", "complex")
        """
        
    def add_edge(
        self, 
        from_node: str, 
        to_node: str, 
        condition: Optional[Callable] = None
    ) -> None:
        """
        Add an edge between nodes.
        
        Args:
            from_node: Source node name
            to_node: Destination node name
            condition: Optional condition function that determines if this edge should be followed
        """
        
    def add_error_handler(
        self, 
        node_name: str, 
        handler: Callable
    ) -> None:
        """
        Add an error handler for a node.
        
        Args:
            node_name: Name of the node to handle errors for
            handler: Function to handle errors
        """
        
    def add_monitor(self, monitor: Callable) -> None:
        """
        Add a monitoring function to the workflow.
        
        Args:
            monitor: Function to call for monitoring workflow execution
        """
        
    def compile(self) -> "Workflow":
        """
        Compile the graph into an executable workflow.
        
        Returns:
            Executable Workflow object
        """
        
    def get_complexity_score(self) -> Dict[str, Any]:
        """
        Get the complexity score for the workflow.
        
        Returns:
            Dictionary containing complexity score information
        """
```

### Workflow Class

```python
class Workflow:
    async def run(self, initial_state: BaseModel) -> BaseModel:
        """
        Run the workflow asynchronously.
        
        Args:
            initial_state: Initial state for the workflow
            
        Returns:
            Final state after workflow execution
        """
        
    def run_sync(self, initial_state: BaseModel) -> BaseModel:
        """
        Run the workflow synchronously.
        
        Args:
            initial_state: Initial state for the workflow
            
        Returns:
            Final state after workflow execution
        """
        
    def get_complexity_score(self) -> Dict[str, Any]:
        """
        Get the complexity score for the workflow.
        
        Returns:
            Dictionary containing complexity score information
        """
```

## Additional APIs

For detailed documentation on other APIs, please refer to the specific API pages:

- [Agents API](agents.md): Specialized agent types and multi-agent systems
- [Tools API](tools.md): Tool chains, routers, and HTTP tools
- [RAG API](rag.md): Document management, retrievers, and RAG systems
- [MCP API](mcp.md): Model Context Protocol client and server
- [A2A API](a2a.md): Agent-to-Agent Protocol client and server
