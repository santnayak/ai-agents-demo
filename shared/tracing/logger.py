"""Logger for agent execution."""
import logging
import sys
from typing import Optional
from datetime import datetime


class AgentLogger:
    """Custom logger for agent execution."""
    
    def __init__(self, name: str = "agent", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.logger.info(full_message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.logger.debug(full_message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.logger.warning(full_message)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.logger.error(full_message)
    
    def log_agent_start(self, agent_name: str):
        """Log agent start."""
        self.info(f"🤖 Starting agent: {agent_name}")
    
    def log_agent_end(self, agent_name: str):
        """Log agent end."""
        self.info(f"✅ Agent completed: {agent_name}")
    
    def log_tool_call(self, tool_name: str, args: Optional[dict] = None):
        """Log tool call."""
        args_str = str(args) if args else ""
        self.info(f"🔧 Calling tool: {tool_name}", args=args_str)
    
    def log_tool_result(self, tool_name: str, result: Optional[str] = None):
        """Log tool result."""
        result_preview = result[:100] if result else "None"
        self.info(f"✓ Tool result: {tool_name}", result=result_preview)
    
    def log_llm_call(self, model: str, message_count: int):
        """Log LLM call."""
        self.info(f"💭 Calling LLM: {model}", messages=message_count)
    
    def log_llm_response(self, model: str, response_length: int):
        """Log LLM response."""
        self.info(f"✓ LLM response: {model}", length=response_length)
