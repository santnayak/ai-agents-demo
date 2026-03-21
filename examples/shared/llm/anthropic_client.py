"""Anthropic LLM client implementation."""
import os
import json
from typing import List, Dict, Any, Optional
from anthropic import Anthropic

from .base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    """Anthropic API client implementation."""
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        super().__init__(model, temperature, max_tokens)
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """Convert messages to Anthropic format, extracting system message."""
        system = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return system, anthropic_messages
    
    def _convert_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert OpenAI-style tools to Anthropic format."""
        if not tools:
            return None
        
        anthropic_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
        
        return anthropic_tools
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat completion request to Anthropic."""
        system, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)
        
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }
        
        if system:
            params["system"] = system
        
        if anthropic_tools:
            params["tools"] = anthropic_tools
        
        response = self.client.messages.create(**params)
        
        # Extract usage information
        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, 'input_tokens', 0),
                "completion_tokens": getattr(response.usage, 'output_tokens', 0),
                "total_tokens": getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0)
            }
        
        result = {
            "content": "",
            "role": "assistant",
            "tool_calls": [],
            "usage": usage
        }
        
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
        
        return result
    
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """Stream chat completion responses from Anthropic."""
        system, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)
        
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }
        
        if system:
            params["system"] = system
        
        if anthropic_tools:
            params["tools"] = anthropic_tools
        
        with self.client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text
