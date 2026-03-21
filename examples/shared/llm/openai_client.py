"""OpenAI LLM client implementation."""
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI

from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI API client implementation."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        super().__init__(model, temperature, max_tokens)
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat completion request to OpenAI."""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        response = self.client.chat.completions.create(**params)
        
        result = {
            "content": response.choices[0].message.content,
            "role": response.choices[0].message.role,
            "tool_calls": []
        }
        
        if response.choices[0].message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in response.choices[0].message.tool_calls
            ]
        
        # Extract token usage
        if hasattr(response, 'usage') and response.usage:
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        return result
    
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """Stream chat completion responses from OpenAI."""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **kwargs
        }
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        stream = self.client.chat.completions.create(**params)
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
