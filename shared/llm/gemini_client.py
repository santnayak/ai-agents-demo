"""Google Gemini LLM client implementation."""
import os
import json
from typing import List, Dict, Any, Optional
import google.generativeai as genai

from .base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """Google Gemini API client implementation."""
    
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        super().__init__(model, temperature, max_tokens)

        print(model)
        
        # Configure Gemini
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        # Store model name
        self.model_name = model
        
        # Generation config
        self.generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
    
    def _convert_messages_to_gemini(self, messages: List[Dict[str, str]]) -> tuple:
        """Convert messages to Gemini format.
        
        Returns:
            Tuple of (history, latest_message)
        """
        system_instruction = None
        history = []
        latest_message = None
        
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"] or ""
            
            if role == "system":
                # Save system instruction to prepend to first user message
                system_instruction = content
            elif role == "user":
                if i == len(messages) - 1:
                    # Last message - prepend system instruction if exists and no history yet
                    if system_instruction and not history:
                        latest_message = f"{system_instruction}\n\nUser: {content}"
                    else:
                        latest_message = content
                else:
                    message_content = content
                    # If this is the first user message and we have system instruction, prepend it
                    if system_instruction and not history:
                        message_content = f"{system_instruction}\n\nUser: {content}"
                        system_instruction = None
                    history.append({"role": "user", "parts": [message_content]})
            elif role == "assistant":
                # Handle assistant messages (both content and tool calls)
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # Assistant is calling tools - represent this in history
                    tool_names = [tc["function"]["name"] for tc in tool_calls]
                    history.append({
                        "role": "model",
                        "parts": [f"I will use these tools: {', '.join(tool_names)}"]
                    })
                elif content:  # Only add non-empty text responses
                    history.append({"role": "model", "parts": [content]})
            elif role == "tool":
                # Append tool results as user messages with context
                if content:
                    tool_name = msg.get("name", "unknown")
                    history.append({
                        "role": "user",
                        "parts": [f"Tool '{tool_name}' returned: {content}"]
                    })
        
        return history, latest_message
    
    def _convert_tools_to_gemini(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List]:
        """Convert OpenAI-style tools to Gemini format.
        
        Based on: https://ai.google.dev/gemini-api/docs/function-calling
        Uses dict-based schema format for compatibility.
        """
        if not tools:
            return None
        
        from google.generativeai.types import FunctionDeclaration, Tool
        
        gemini_functions = []
        
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                
                # Convert parameters schema - use dict format for compatibility
                parameters = func.get("parameters", {})
                properties = parameters.get("properties", {})
                required = parameters.get("required", [])
                
                # Build Gemini-compatible properties dict
                gemini_properties = {}
                
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get("type", "string").upper()
                    
                    # Gemini uses "INTEGER", "STRING", "NUMBER", etc.
                    gemini_properties[prop_name] = {
                        "type": prop_type,
                        "description": prop_info.get("description", "")
                    }
                
                # Build parameters dict in Gemini format
                gemini_params = {
                    "type": "OBJECT",
                    "properties": gemini_properties
                }
                
                if required:
                    gemini_params["required"] = required
                
                # Create function declaration
                func_decl = FunctionDeclaration(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters=gemini_params
                )
                gemini_functions.append(func_decl)
        
        if gemini_functions:
            return [Tool(function_declarations=gemini_functions)]
        return None
    
    def _extract_tool_calls(self, response) -> List[Dict[str, Any]]:
        """Extract tool calls from Gemini response.
        
        Gemini returns function calls in the response.candidates[0].content.parts
        when the model decides to use a function.
        """
        tool_calls = []
        
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for i, part in enumerate(candidate.content.parts):
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            
                            # Convert args to dict (Gemini uses a proto dict)
                            args_dict = {}
                            if hasattr(fc, 'args') and fc.args:
                                # fc.args is a proto MapComposite, iterate properly
                                for key in fc.args:
                                    args_dict[key] = fc.args[key]
                            
                            # Create unique ID for this tool call
                            tool_call_id = f"call_{abs(hash(fc.name + str(i)))}"
                            
                            tool_calls.append({
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": json.dumps(args_dict)
                                }
                            })
        except Exception as e:
            # Log error but don't crash
            import sys
            print(f"Warning: Error extracting tool calls: {e}", file=sys.stderr)
            pass
        
        return tool_calls
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat completion request to Gemini."""
        history, latest_message = self._convert_messages_to_gemini(messages)
        gemini_tools = self._convert_tools_to_gemini(tools)
        
        # Create model instance
        try:
            # Create model with tools if provided (required for function calling)
            if gemini_tools:
                model = genai.GenerativeModel(
                    self.model_name,
                    tools=gemini_tools
                )
            else:
                model = genai.GenerativeModel(self.model_name)
   
            # Start chat with history
            chat = model.start_chat(history=history)
            
            # Send message
            response = chat.send_message(
                latest_message or "Continue",
                generation_config=self.generation_config
            )
            
            # Extract content
            content = ""
            try:
                if hasattr(response, 'text'):
                    content = response.text
            except ValueError:
                # Text not available when there are tool calls
                pass
            except Exception:
                pass
            
            # Extract tool calls
            tool_calls = self._extract_tool_calls(response)
            
            # Debug: Check if we got code generation instead of function calls
            if not tool_calls and content and ("tool_code" in content or "default_api" in content):
                # Model is generating code instead of function calls
                # This can happen with certain Gemini models
                import sys
                print(f"\n⚠️ Warning: Gemini is generating code instead of function calls.", file=sys.stderr)
                print(f"   Model: {self.model_name}", file=sys.stderr)
                print(f"   Consider using 'gemini-1.5-flash' or 'gemini-1.5-pro' for better function calling support.", file=sys.stderr)
                print(f"   Response: {content[:200]}", file=sys.stderr)
            
            result = {
                "content": content,
                "role": "assistant",
                "tool_calls": tool_calls
            }
            
            return result
            
        except Exception as e:
            # Handle errors gracefully
            return {
                "content": f"Error calling Gemini: {str(e)}",
                "role": "assistant",
                "tool_calls": []
            }
    
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """Stream chat completion responses from Gemini."""
        history, latest_message = self._convert_messages_to_gemini(messages)
        gemini_tools = self._convert_tools_to_gemini(tools)
        
        # Create model instance
        try:
            model = genai.GenerativeModel(self.model_name)
            
            # Start chat with history
            chat = model.start_chat(history=history)
            
            # Stream response
            response_stream = chat.send_message(
                latest_message or "Continue",
                generation_config=self.generation_config,
                tools=gemini_tools,
                stream=True
            )
            
            for chunk in response_stream:
                try:
                    if hasattr(chunk, 'text') and chunk.text:
                        yield chunk.text
                except Exception:
                    pass
                    
        except Exception as e:
            yield f"Error: {str(e)}"
