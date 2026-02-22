"""Simple agent implementation with basic tools."""
import json
import sys
import uuid
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm.factory import LLMFactory
from shared.schemas.messages import ConversationHistory, Message
from shared.schemas.tool_schema import ToolRegistry
from shared.tracing.tracer import Tracer
from shared.tracing.logger import AgentLogger
from typing import Optional, Dict, Any


class SimpleAgent:
    """A simple agent that can use tools to complete tasks."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_iterations: int = 10,
        verbose: bool = True,
        session_id: Optional[str] = None,
        auto_save_traces: bool = True,
        traces_dir: str = "traces"
    ):
        """Initialize the simple agent.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', or 'gemini')
            model: Model name
            temperature: Sampling temperature
            max_iterations: Maximum number of agent iterations
            verbose: Whether to log execution details
            session_id: Session identifier for grouping traces (auto-generated if None)
            auto_save_traces: Whether to automatically save traces after each run
            traces_dir: Directory to save traces
        """
        self.llm = LLMFactory.create_client(
            provider=provider,
            model=model,
            temperature=temperature
        )
        
        self.tool_registry = ToolRegistry()
        self.conversation = ConversationHistory()
        self.session_id = session_id or str(uuid.uuid4())
        self.tracer = Tracer(session_id=self.session_id)
        self.logger = AgentLogger(level="INFO" if verbose else "WARNING")
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.temperature = temperature
        self.auto_save_traces = auto_save_traces
        self.traces_dir = Path(traces_dir)
        self.run_count = 0
        
        # Create traces directory if it doesn't exist
        if self.auto_save_traces:
            self.traces_dir.mkdir(parents=True, exist_ok=True)
        
        # Add system message
        system_prompt = """You are a helpful AI assistant with access to tools.
When you need to use a tool, you will be provided with the tool call interface.
Always think step by step and use tools when necessary to help the user.
Provide clear and concise responses."""
        
        self.conversation.add_system_message(system_prompt)
    
    def register_tool(self, tool):
        """Register a tool with the agent."""
        self.tool_registry.register(tool)
    
    def run(self, user_input: str) -> str:
        """Run the agent with user input.
        
        Args:
            user_input: User's message/query
            
        Returns:
            Agent's final response
        """
        self.logger.log_agent_start("SimpleAgent")
        self.tracer.start_trace(metadata={"user_input": user_input})
        
        try:
            # Add user message
            self.conversation.add_user_message(user_input)
            
            iteration = 0
            while iteration < self.max_iterations:
                iteration += 1
                
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Iteration {iteration}")
                    print(f"{'='*60}")
                
                # Call LLM
                response = self._call_llm()
                
                # Check if there are tool calls
                if response.get("tool_calls"):
                    # Execute tools
                    self._execute_tools(response["tool_calls"])
                else:
                    # No more tool calls, return final response
                    final_response = response.get("content", "")
                    self.conversation.add_assistant_message(final_response)
                    
                    self.tracer.end_trace()
                    self.logger.log_agent_end("SimpleAgent")
                    
                    # Auto-save trace if enabled
                    if self.auto_save_traces:
                        self._save_current_trace()
                    
                    if self.verbose:
                        print(f"\n{'='*60}")
                        print("Final Response:")
                        print(f"{'='*60}")
                        print(final_response)
                    
                    return final_response
            
            # Max iterations reached
            error_msg = f"Maximum iterations ({self.max_iterations}) reached"
            self.tracer.fail_trace(error_msg)
            self.logger.error(error_msg)
            
            if self.auto_save_traces:
                self._save_current_trace()
            
            return "I apologize, but I've reached my processing limit. Please try breaking down your request."
            
        except Exception as e:
            self.tracer.fail_trace(str(e))
            self.logger.error(f"Agent execution failed: {str(e)}")
            
            if self.auto_save_traces:
                self._save_current_trace()
            
            raise
    
    def _call_llm(self) -> Dict[str, Any]:
        """Call the LLM with current conversation history."""
        messages = self.conversation.to_list()
        tools = self.tool_registry.get_all_schemas()
        
        self.logger.log_llm_call(
            model=self.llm.model,
            message_count=len(messages)
        )
        
        # Capture full LLM request payload and response
        with self.tracer.trace_step(
            "llm_call",
            input_data={
                "messages": messages,
                "tools": tools if tools else [],
                "model": self.llm.model,
                "temperature": self.temperature
            }
        ) as step:
            response = self.llm.chat(
                messages=messages,
                tools=tools if tools else None
            )
            # Capture full response
            step.output_data = {
                "content": response.get("content"),
                "tool_calls": response.get("tool_calls"),
                "full_response": response
            }
        
        self.logger.log_llm_response(
            model=self.llm.model,
            response_length=len(response.get("content", ""))
        )
        
        if self.verbose:
            print(f"\n🤖 Assistant: {response.get('content', '(tool calls)')}")
            if response.get("tool_calls"):
                print(f"🔧 Tool calls requested: {len(response['tool_calls'])}")
        
        return response
    
    def _execute_tools(self, tool_calls: list):
        """Execute tool calls and add results to conversation.
        
        Args:
            tool_calls: List of tool call dictionaries
        """
        # Add assistant message with tool calls
        self.conversation.add_assistant_message("", tool_calls=tool_calls)
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args_str = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]
            
            try:
                # Parse arguments
                tool_args = json.loads(tool_args_str)
                
                self.logger.log_tool_call(tool_name, tool_args)
                
                if self.verbose:
                    print(f"\n🔧 Executing tool: {tool_name}")
                    print(f"   Arguments: {tool_args}")
                
                # Execute tool
                with self.tracer.trace_step(
                    "tool_call",
                    input_data={
                        "tool": tool_name,
                        "arguments": tool_args,
                        "tool_call_id": tool_call_id
                    }
                ) as step:
                    result = self.tool_registry.execute_tool(tool_name, **tool_args)
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                    # Capture full result (not truncated)
                    step.output_data = {
                        "result": result,
                        "result_str": result_str
                    }
                
                self.logger.log_tool_result(tool_name, result_str)
                
                if self.verbose:
                    print(f"   ✅ Result: {result_str[:200]}...")
                
                # Add tool result to conversation
                self.conversation.add_tool_message(
                    content=result_str,
                    tool_call_id=tool_call_id,
                    name=tool_name
                )
                
            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                self.logger.error(error_msg)
                
                if self.verbose:
                    print(f"   ❌ Error: {error_msg}")
                
                # Add error message to conversation
                self.conversation.add_tool_message(
                    content=error_msg,
                    tool_call_id=tool_call_id,
                    name=tool_name
                )
    
    def get_tracer(self):
        """Get the tracer instance for saving/inspecting traces."""
        return self.tracer
    
    def save_trace(self, filepath: str) -> bool:
        """Save the execution trace to a file.
        
        Args:
            filepath: Path to save the trace file
            
        Returns:
            True if successful, False otherwise
        """
        return self.tracer.save_trace_to_file(filepath)
    
    def _save_current_trace(self):
        """Save the current trace with auto-generated filename."""
        self.run_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{self.session_id[:8]}_run_{self.run_count:03d}_{timestamp}.json"
        filepath = self.traces_dir / filename
        
        success = self.tracer.save_trace_to_file(str(filepath))
        if success and self.verbose:
            print(f"\n💾 Trace saved: {filepath}")
