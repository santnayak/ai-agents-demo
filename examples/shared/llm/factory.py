"""Factory for creating LLM clients."""
from typing import Optional
from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient


class LLMFactory:
    """Factory class for creating LLM clients."""
    
    @staticmethod
    def create_client(
        provider: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ) -> BaseLLMClient:
        """Create an LLM client based on provider.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', or 'gemini')
            model: Model name (uses default if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: API key for the provider
            
        Returns:
            BaseLLMClient instance
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()
        
        if provider == "openai":
            model = model or "gpt-4"
            return OpenAIClient(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key
            )
        elif provider == "anthropic":
            model = model or "claude-3-5-sonnet-20241022"
            return AnthropicClient(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key
            )
        elif provider == "gemini":
            model = model or "gemini-1.5-flash"
            return GeminiClient(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', or 'gemini'.")
