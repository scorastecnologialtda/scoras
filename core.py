"""
Core module for the Scoras library with MCP and A2A protocol support.

This module contains the core functionality of the Scoras library,
including the scoring system for measuring workflow complexity.

Author: Anderson L. Amaral
"""

import asyncio
import inspect
import json
import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Type, Union, TypeVar, get_type_hints

from pydantic import BaseModel, Field, create_model

# Configure logging
logger = logging.getLogger(__name__)

class ScorasConfig:
    """Global configuration for the Scoras library."""
    
    # Default model provider
    DEFAULT_MODEL_PROVIDER = "openai"
    
    # Scoring weights
    SCORE_WEIGHTS = {
        "nodes": {
            "simple": 1.0,
            "standard": 1.0,
            "complex": 1.5
        },
        "edges": {
            "standard": 1.5,
            "conditional": 4.0
        },
        "tools": {
            "simple": 1.4,
            "standard": 2.0,
            "complex": 3.0
        },
        "conditions": {
            "standard": 2.5
        }
    }
    
    # Complexity thresholds
    COMPLEXITY_THRESHOLDS = {
        "simple": 10,
        "moderate": 25,
        "complex": 50,
        "very_complex": 100
    }
    
    # Protocol support
    ENABLE_MCP = True
    ENABLE_A2A = True
    
    # Default protocol ports
    DEFAULT_MCP_PORT = 8000
    DEFAULT_A2A_PORT = 8001

class ScoreTracker:
    """Tracks complexity scores for workflows."""
    
    def __init__(self):
        """Initialize a score tracker."""
        self.components = {
            "nodes": 0,
            "edges": 0,
            "tools": 0,
            "conditions": 0
        }
        
        self.component_counts = {
            "nodes": 0,
            "edges": 0,
            "tools": 0,
            "conditions": 0
        }
    
    def add_node(self, complexity: str = "standard") -> None:
        """
        Add a node to the score.
        
        Args:
            complexity: Complexity of the node (simple, standard, complex)
        """
        weight = ScorasConfig.SCORE_WEIGHTS["nodes"].get(complexity, 1.0)
        self.components["nodes"] += weight
        self.component_counts["nodes"] += 1
    
    def add_edge(self, conditional: bool = False) -> None:
        """
        Add an edge to the score.
        
        Args:
            conditional: Whether the edge is conditional
        """
        edge_type = "conditional" if conditional else "standard"
        weight = ScorasConfig.SCORE_WEIGHTS["edges"].get(edge_type, 1.5)
        self.components["edges"] += weight
        self.component_counts["edges"] += 1
        
        if conditional:
            self.components["conditions"] += ScorasConfig.SCORE_WEIGHTS["conditions"]["standard"]
            self.component_counts["conditions"] += 1
    
    def add_tool(self, complexity: str = "standard") -> None:
        """
        Add a tool to the score.
        
        Args:
            complexity: Complexity of the tool (simple, standard, complex)
        """
        weight = ScorasConfig.SCORE_WEIGHTS["tools"].get(complexity, 2.0)
        self.components["tools"] += weight
        self.component_counts["tools"] += 1
    
    @property
    def total_score(self) -> float:
        """Get the total score."""
        return sum(self.components.values())
    
    def get_complexity_rating(self) -> str:
        """Get the complexity rating based on the total score."""
        score = self.total_score
        
        if score < ScorasConfig.COMPLEXITY_THRESHOLDS["simple"]:
            return "Simple"
        elif score < ScorasConfig.COMPLEXITY_THRESHOLDS["moderate"]:
            return "Moderate"
        elif score < ScorasConfig.COMPLEXITY_THRESHOLDS["complex"]:
            return "Complex"
        elif score < ScorasConfig.COMPLEXITY_THRESHOLDS["very_complex"]:
            return "Very Complex"
        else:
            return "Extremely Complex"
    
    def get_report(self) -> Dict[str, Any]:
        """Get a detailed report of the score."""
        return {
            "total_score": self.total_score,
            "complexity_rating": self.get_complexity_rating(),
            "component_scores": self.components.copy(),
            "component_counts": self.component_counts.copy(),
            "breakdown": {
                "nodes": f"{self.component_counts['nodes']} nodes ({self.components['nodes']:.1f} points)",
                "edges": f"{self.component_counts['edges']} edges ({self.components['edges']:.1f} points)",
                "tools": f"{self.component_counts['tools']} tools ({self.components['tools']:.1f} points)",
                "conditions": f"{self.component_counts['conditions']} conditions ({self.components['conditions']:.1f} points)"
            }
        }

class Message(BaseModel):
    """Message exchanged between a user and an agent."""
    
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")
    
    def to_dict(self) -> Dict[str, str]:
        """Convert the message to a dictionary."""
        return {
            "role": self.role,
            "content": self.content
        }

class Tool:
    """Tool that can be used by an agent."""
    
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        complexity: str = "standard"
    ):
        """
        Initialize a tool.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            function: Function to execute
            parameters: Parameters of the function
            complexity: Complexity of the tool (simple, standard, complex)
        """
        self.name = name
        self.description = description
        self.function = function
        self.complexity = complexity
        
        # Extract parameters from function signature if not provided
        if parameters is None:
            self.parameters = self._extract_parameters()
        else:
            self.parameters = parameters
    
    def _extract_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Extract parameters from the function signature."""
        parameters = {}
        signature = inspect.signature(self.function)
        type_hints = get_type_hints(self.function)
        
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            
            param_info = {
                "type": "string",
                "description": f"Parameter '{name}'"
            }
            
            # Get type information
            if name in type_hints:
                type_hint = type_hints[name]
                if type_hint == int:
                    param_info["type"] = "integer"
                elif type_hint == float:
                    param_info["type"] = "number"
                elif type_hint == bool:
                    param_info["type"] = "boolean"
                elif type_hint == list or getattr(type_hint, "__origin__", None) == list:
                    param_info["type"] = "array"
                elif type_hint == dict or getattr(type_hint, "__origin__", None) == dict:
                    param_info["type"] = "object"
            
            # Get default value
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
            
            parameters[name] = param_info
        
        return parameters
    
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool.
        
        Args:
            **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function
        """
        if asyncio.iscoroutinefunction(self.function):
            return await self.function(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.function(**kwargs))

def tool(name: Optional[str] = None, description: Optional[str] = None, complexity: str = "standard"):
    """
    Decorator to create a tool from a function.
    
    Args:
        name: Name of the tool (defaults to function name)
        description: Description of the tool (defaults to function docstring)
        complexity: Complexity of the tool (simple, standard, complex)
        
    Returns:
        Decorated function as a Tool
    """
    def decorator(func):
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool for {tool_name}"
        
        return Tool(
            name=tool_name,
            description=tool_description,
            function=func,
            complexity=complexity
        )
    
    return decorator

class ModelProvider:
    """Base class for model providers."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize a model provider.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional arguments for the provider
        """
        self.model_name = model_name
        self.kwargs = kwargs
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response from the model.
        
        Args:
            messages: List of messages
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated response
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from the model with tool calling.
        
        Args:
            messages: List of messages
            tools: List of tools
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated response with tool calls
        """
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIProvider(ModelProvider):
    """Provider for OpenAI models."""
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response from an OpenAI model.
        
        Args:
            messages: List of messages
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated response
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("OpenAI package is required. Install it with 'pip install openai'.")
        
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **{**self.kwargs, **kwargs}
        )
        
        return response.choices[0].message.content
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from an OpenAI model with tool calling.
        
        Args:
            messages: List of messages
            tools: List of tools
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated response with tool calls
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("OpenAI package is required. Install it with 'pip install openai'.")
        
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            **{**self.kwargs, **kwargs}
        )
        
        message = response.choices[0].message
        
        result = {
            "content": message.content,
            "tool_calls": []
        }
        
        if hasattr(message, "tool_calls") and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                }
                for tool_call in message.tool_calls
            ]
        
        return result

class AnthropicProvider(ModelProvider):
    """Provider for Anthropic models."""
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response from an Anthropic model.
        
        Args:
            messages: List of messages
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated response
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package is required. Install it with 'pip install anthropic'.")
        
        client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        # Convert messages to Anthropic format
        anthropic_messages = []
        for message in messages:
            role = "user" if message["role"] == "user" else "assistant"
            anthropic_messages.append({
                "role": role,
                "content": message["content"]
            })
        
        response = await client.messages.create(
            model=self.model_name,
            messages=anthropic_messages,
            **{**self.kwargs, **kwargs}
        )
        
        return response.content[0].text
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from an Anthropic model with tool calling.
        
        Args:
            messages: List of messages
            tools: List of tools
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated response with tool calls
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package is required. Install it with 'pip install anthropic'.")
        
        client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        # Convert messages to Anthropic format
        anthropic_messages = []
        for message in messages:
            role = "user" if message["role"] == "user" else "assistant"
            anthropic_messages.append({
                "role": role,
                "content": message["content"]
            })
        
        # Convert tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append({
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "input_schema": {
                    "type": "object",
                    "properties": tool["function"].get("parameters", {}).get("properties", {}),
                    "required": tool["function"].get("parameters", {}).get("required", [])
                }
            })
        
        response = await client.messages.create(
            model=self.model_name,
            messages=anthropic_messages,
            tools=anthropic_tools,
            **{**self.kwargs, **kwargs}
        )
        
        result = {
            "content": response.content[0].text,
            "tool_calls": []
        }
        
        # Extract tool calls
        for block in response.content:
            if block.type == "tool_use":
                result["tool_calls"].append({
                    "id": str(uuid.uuid4()),
                    "name": block.name,
                    "arguments": block.input
                })
        
        return result

class GeminiProvider(ModelProvider):
    """Provider for Google Gemini models."""
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response from a Gemini model.
        
        Args:
            messages: List of messages
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated response
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package is required. Install it with 'pip install google-generativeai'.")
        
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        
        model = genai.GenerativeModel(self.model_name)
        
        # Convert messages to Gemini format
        gemini_messages = []
        for message in messages:
            role = "user" if message["role"] == "user" else "model"
            gemini_messages.append({
                "role": role,
                "parts": [{"text": message["content"]}]
            })
        
        response = await model.generate_content_async(gemini_messages, **{**self.kwargs, **kwargs})
        
        return response.text
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from a Gemini model with tool calling.
        
        Args:
            messages: List of messages
            tools: List of tools
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated response with tool calls
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package is required. Install it with 'pip install google-generativeai'.")
        
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        
        model = genai.GenerativeModel(self.model_name)
        
        # Convert messages to Gemini format
        gemini_messages = []
        for message in messages:
            role = "user" if message["role"] == "user" else "model"
            gemini_messages.append({
                "role": role,
                "parts": [{"text": message["content"]}]
            })
        
        # Convert tools to Gemini format
        gemini_tools = []
        for tool in tools:
            gemini_tools.append({
                "function_declarations": [
                    {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"].get("parameters", {})
                    }
                ]
            })
        
        response = await model.generate_content_async(
            gemini_messages,
            tools=gemini_tools,
            **{**self.kwargs, **kwargs}
        )
        
        result = {
            "content": response.text,
            "tool_calls": []
        }
        
        # Extract tool calls
        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            result["tool_calls"].append({
                                "id": str(uuid.uuid4()),
                                "name": part.function_call.name,
                                "arguments": part.function_call.args
                            })
        
        return result

def get_provider(model: str) -> ModelProvider:
    """
    Get a model provider based on the model name.
    
    Args:
        model: Model name in the format "provider:model_name"
        
    Returns:
        Model provider
    """
    if ":" in model:
        provider_name, model_name = model.split(":", 1)
    else:
        provider_name = ScorasConfig.DEFAULT_MODEL_PROVIDER
        model_name = model
    
    if provider_name == "openai":
        return OpenAIProvider(model_name)
    elif provider_name == "anthropic":
        return AnthropicProvider(model_name)
    elif provider_name == "gemini":
        return GeminiProvider(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")

class Agent:
    """Agent that can generate responses and use tools."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        result_type: Optional[Type[BaseModel]] = None,
        enable_scoring: bool = True,
        **kwargs
    ):
        """
        Initialize an agent.
        
        Args:
            model: Model name in the format "provider:model_name"
            system_prompt: System prompt for the agent
            tools: Tools available to the agent
            result_type: Expected result type (Pydantic model)
            enable_scoring: Whether to enable complexity scoring
            **kwargs: Additional arguments for the model provider
        """
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.result_type = result_type
        self.enable_scoring = enable_scoring
        self.kwargs = kwargs
        self.conversation_history = []
        self.score_tracker = ScoreTracker() if enable_scoring else None
        
        # Initialize the provider if model is provided
        self.provider = get_provider(model) if model else None
        
        # Add score for agent creation
        if self.score_tracker:
            self.score_tracker.add_node("standard")
            
            # Add tools
            for tool in self.tools:
                self.score_tracker.add_tool(tool.complexity)
    
    async def run(self, user_input: str, **kwargs) -> Any:
        """
        Run the agent with user input.
        
        Args:
            user_input: User input
            **kwargs: Additional arguments for the model provider
            
        Returns:
            Generated response
        """
        if not self.provider:
            raise ValueError("Model provider not initialized. Provide a model name when creating the agent.")
        
        # Add score for agent execution
        if self.score_tracker:
            self.score_tracker.add_edge()
        
        # Add system prompt if provided
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add user input
        messages.append({"role": "user", "content": user_input})
        
        # Check if tools are available
        if self.tools:
            # Convert tools to the format expected by the provider
            tool_defs = []
            for tool in self.tools:
                tool_defs.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                name: {
                                    "type": info.get("type", "string"),
                                    "description": info.get("description", "")
                                }
                                for name, info in tool.parameters.items()
                            },
                            "required": [
                                name for name, info in tool.parameters.items()
                                if "default" not in info
                            ]
                        }
                    }
                })
            
            # Generate response with tools
            response = await self.provider.generate_with_tools(
                messages=messages,
                tools=tool_defs,
                **{**self.kwargs, **kwargs}
            )
            
            # Check if tool calls are present
            if response["tool_calls"]:
                # Add score for tool calling
                if self.score_tracker:
                    self.score_tracker.add_edge()
                    self.score_tracker.add_node("complex")
                
                # Execute tools
                tool_results = []
                for tool_call in response["tool_calls"]:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]
                    
                    # Find the tool
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    if tool:
                        try:
                            # Execute the tool
                            result = await tool.execute(**tool_args)
                            
                            # Add the tool call and result to the conversation
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": tool_call["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": json.dumps(tool_args)
                                        }
                                    }
                                ]
                            })
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                            })
                            
                            tool_results.append({
                                "name": tool_name,
                                "result": result
                            })
                        except Exception as e:
                            logger.exception(f"Error executing tool {tool_name}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": f"Error: {str(e)}"
                            })
                            
                            tool_results.append({
                                "name": tool_name,
                                "error": str(e)
                            })
                
                # Generate final response
                final_response = await self.provider.generate(
                    messages=messages,
                    **{**self.kwargs, **kwargs}
                )
                
                # Add the final response to the conversation
                messages.append({"role": "assistant", "content": final_response})
                
                # Update conversation history
                self.conversation_history = messages
                
                # Parse result if result_type is provided
                if self.result_type:
                    try:
                        return self.result_type.model_validate_json(final_response)
                    except Exception as e:
                        logger.exception(f"Error parsing result as {self.result_type.__name__}")
                        return final_response
                
                return final_response
            
            # No tool calls, use the content
            content = response["content"]
            
            # Add the response to the conversation
            messages.append({"role": "assistant", "content": content})
            
            # Update conversation history
            self.conversation_history = messages
            
            # Parse result if result_type is provided
            if self.result_type:
                try:
                    return self.result_type.model_validate_json(content)
                except Exception as e:
                    logger.exception(f"Error parsing result as {self.result_type.__name__}")
                    return content
            
            return content
        else:
            # No tools, generate a simple response
            response = await self.provider.generate(
                messages=messages,
                **{**self.kwargs, **kwargs}
            )
            
            # Add the response to the conversation
            messages.append({"role": "assistant", "content": response})
            
            # Update conversation history
            self.conversation_history = messages
            
            # Parse result if result_type is provided
            if self.result_type:
                try:
                    return self.result_type.model_validate_json(response)
                except Exception as e:
                    logger.exception(f"Error parsing result as {self.result_type.__name__}")
                    return response
            
            return response
    
    def run_sync(self, user_input: str, **kwargs) -> Any:
        """
        Run the agent synchronously.
        
        Args:
            user_input: User input
            **kwargs: Additional arguments for the model provider
            
        Returns:
            Generated response
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.run(user_input, **kwargs))
    
    def add_tool(self, tool: Tool) -> 'Agent':
        """
        Add a tool to the agent.
        
        Args:
            tool: Tool to add
            
        Returns:
            Self for chaining
        """
        self.tools.append(tool)
        
        # Add score for tool addition
        if self.score_tracker:
            self.score_tracker.add_tool(tool.complexity)
        
        return self
    
    def add_system_prompt(self, prompt: str) -> 'Agent':
        """
        Add or update the system prompt.
        
        Args:
            prompt: System prompt
            
        Returns:
            Self for chaining
        """
        self.system_prompt = prompt
        return self
    
    def clear_history(self) -> 'Agent':
        """
        Clear the conversation history.
        
        Returns:
            Self for chaining
        """
        self.conversation_history = []
        return self
    
    def get_complexity_score(self) -> Optional[Dict[str, Any]]:
        """Get the complexity score of the agent."""
        if not self.score_tracker:
            return None
        
        return self.score_tracker.get_report()

class WorkflowNode:
    """Node in a workflow graph."""
    
    def __init__(
        self,
        function: Callable,
        complexity: str = "standard"
    ):
        """
        Initialize a workflow node.
        
        Args:
            function: Function to execute
            complexity: Complexity of the node (simple, standard, complex)
        """
        self.function = function
        self.complexity = complexity
    
    async def execute(self, state: Any) -> Dict[str, Any]:
        """
        Execute the node.
        
        Args:
            state: Current state
            
        Returns:
            Updates to the state
        """
        if asyncio.iscoroutinefunction(self.function):
            return await self.function(state)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.function(state))

class WorkflowEdge:
    """Edge in a workflow graph."""
    
    def __init__(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[[Any], bool]] = None
    ):
        """
        Initialize a workflow edge.
        
        Args:
            source: Source node name
            target: Target node name
            condition: Condition function
        """
        self.source = source
        self.target = target
        self.condition = condition
    
    async def check_condition(self, state: Any) -> bool:
        """
        Check if the condition is met.
        
        Args:
            state: Current state
            
        Returns:
            Whether the condition is met
        """
        if not self.condition:
            return True
        
        if asyncio.iscoroutinefunction(self.condition):
            return await self.condition(state)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.condition(state))

class WorkflowGraph:
    """Graph for defining workflows."""
    
    def __init__(
        self,
        state_type: Type,
        enable_scoring: bool = True
    ):
        """
        Initialize a workflow graph.
        
        Args:
            state_type: Type of the state
            enable_scoring: Whether to enable complexity scoring
        """
        self.state_type = state_type
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[str, List[WorkflowEdge]] = {}
        self.enable_scoring = enable_scoring
        self.score_tracker = ScoreTracker() if enable_scoring else None
        
        # Add score for graph creation
        if self.score_tracker:
            self.score_tracker.add_node("standard")
    
    def add_node(
        self,
        name: str,
        function: Callable,
        complexity: str = "standard"
    ) -> 'WorkflowGraph':
        """
        Add a node to the graph.
        
        Args:
            name: Name of the node
            function: Function to execute
            complexity: Complexity of the node (simple, standard, complex)
            
        Returns:
            Self for chaining
        """
        self.nodes[name] = WorkflowNode(function, complexity)
        
        # Add score for node addition
        if self.score_tracker:
            self.score_tracker.add_node(complexity)
        
        return self
    
    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[[Any], bool]] = None
    ) -> 'WorkflowGraph':
        """
        Add an edge to the graph.
        
        Args:
            source: Source node name
            target: Target node name
            condition: Condition function
            
        Returns:
            Self for chaining
        """
        if source not in self.edges:
            self.edges[source] = []
        
        self.edges[source].append(WorkflowEdge(source, target, condition))
        
        # Add score for edge addition
        if self.score_tracker:
            self.score_tracker.add_edge(condition is not None)
        
        return self
    
    def compile(self) -> 'CompiledWorkflow':
        """
        Compile the graph into an executable workflow.
        
        Returns:
            Compiled workflow
        """
        return CompiledWorkflow(self)
    
    def get_complexity_score(self) -> Optional[Dict[str, Any]]:
        """Get the complexity score of the graph."""
        if not self.score_tracker:
            return None
        
        return self.score_tracker.get_report()
    
    def visualize(self, format: str = "text") -> str:
        """
        Visualize the graph.
        
        Args:
            format: Output format (text, mermaid)
            
        Returns:
            Visualization of the graph
        """
        if format == "mermaid":
            return self._visualize_mermaid()
        else:
            return self._visualize_text()
    
    def _visualize_text(self) -> str:
        """Visualize the graph as text."""
        lines = ["Workflow Graph:"]
        
        # Add nodes
        lines.append("\nNodes:")
        for name, node in self.nodes.items():
            lines.append(f"  - {name} ({node.complexity})")
        
        # Add edges
        lines.append("\nEdges:")
        for source, edges in self.edges.items():
            for edge in edges:
                condition = " [conditional]" if edge.condition else ""
                lines.append(f"  - {source} -> {edge.target}{condition}")
        
        return "\n".join(lines)
    
    def _visualize_mermaid(self) -> str:
        """Visualize the graph as a Mermaid diagram."""
        lines = ["```mermaid", "graph TD;"]
        
        # Add nodes
        for name in self.nodes:
            lines.append(f"    {name}[{name}];")
        
        # Add edges
        for source, edges in self.edges.items():
            for edge in edges:
                if edge.condition:
                    lines.append(f"    {source} -->|condition| {edge.target};")
                else:
                    lines.append(f"    {source} --> {edge.target};")
        
        lines.append("```")
        
        return "\n".join(lines)

class CompiledWorkflow:
    """Compiled workflow that can be executed."""
    
    def __init__(self, graph: WorkflowGraph):
        """
        Initialize a compiled workflow.
        
        Args:
            graph: Workflow graph
        """
        self.graph = graph
        self.state_type = graph.state_type
    
    async def run(self, initial_state: Any) -> Any:
        """
        Run the workflow.
        
        Args:
            initial_state: Initial state
            
        Returns:
            Final state
        """
        # Convert initial state to the state type if needed
        if not isinstance(initial_state, self.state_type):
            state = self.state_type.model_validate(initial_state)
        else:
            state = initial_state
        
        # Start from the "start" node
        current_node = "start"
        
        # Execute until we reach the "end" node
        while current_node != "end":
            # Get outgoing edges
            edges = self.graph.edges.get(current_node, [])
            
            # Find the first edge whose condition is met
            next_node = None
            for edge in edges:
                if await edge.check_condition(state):
                    next_node = edge.target
                    break
            
            if not next_node:
                raise ValueError(f"No valid edge from node '{current_node}'")
            
            # Execute the next node if it's not "end"
            if next_node != "end":
                node = self.graph.nodes.get(next_node)
                if not node:
                    raise ValueError(f"Node '{next_node}' not found")
                
                # Execute the node
                updates = await node.execute(state)
                
                # Update the state
                if updates:
                    if isinstance(state, BaseModel):
                        # For Pydantic models, use model_copy and update
                        state = state.model_copy(update=updates)
                    elif isinstance(state, dict):
                        # For dictionaries, update directly
                        state.update(updates)
            
            # Move to the next node
            current_node = next_node
        
        return state
    
    def run_sync(self, initial_state: Any) -> Any:
        """
        Run the workflow synchronously.
        
        Args:
            initial_state: Initial state
            
        Returns:
            Final state
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.run(initial_state))

class RAG:
    """Base class for Retrieval Augmented Generation."""
    
    def __init__(
        self,
        agent: Agent,
        enable_scoring: bool = True
    ):
        """
        Initialize a RAG system.
        
        Args:
            agent: Agent to use for generation
            enable_scoring: Whether to enable complexity scoring
        """
        self.agent = agent
        self.enable_scoring = enable_scoring
        self.score_tracker = ScoreTracker() if enable_scoring else None
        
        # Add score for RAG creation
        if self.score_tracker:
            self.score_tracker.add_node("complex")
    
    async def retrieve(self, query: str) -> List[str]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def generate(self, query: str, documents: List[str]) -> str:
        """
        Generate a response based on retrieved documents.
        
        Args:
            query: Query string
            documents: Retrieved documents
            
        Returns:
            Generated response
        """
        # Add score for generation
        if self.score_tracker:
            self.score_tracker.add_edge()
        
        # Create a prompt with the documents
        context = "\n\n".join(documents)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Run the agent
        return await self.agent.run(prompt)
    
    async def run(self, query: str) -> str:
        """
        Run the RAG system.
        
        Args:
            query: Query string
            
        Returns:
            Generated response
        """
        # Add score for RAG execution
        if self.score_tracker:
            self.score_tracker.add_edge()
        
        # Retrieve documents
        documents = await self.retrieve(query)
        
        # Generate response
        return await self.generate(query, documents)
    
    def run_sync(self, query: str) -> str:
        """
        Run the RAG system synchronously.
        
        Args:
            query: Query string
            
        Returns:
            Generated response
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.run(query))
    
    def get_complexity_score(self) -> Optional[Dict[str, Any]]:
        """Get the complexity score of the RAG system."""
        if not self.score_tracker:
            return None
        
        # Add agent score if available
        if hasattr(self.agent, "get_complexity_score") and callable(self.agent.get_complexity_score):
            agent_score = self.agent.get_complexity_score()
            if agent_score:
                for component_type, score in agent_score["component_scores"].items():
                    self.score_tracker.components[component_type] += score
                
                for component_type, count in agent_score["component_counts"].items():
                    self.score_tracker.component_counts[component_type] += count
        
        return self.score_tracker.get_report()
