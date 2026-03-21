"""Message schemas for agent communication."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Message:
    """Represents a message in a conversation."""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool responses
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        result = {
            "role": self.role,
            "content": self.content
        }
        
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        
        if self.name:
            result["name"] = self.name
        
        return result


@dataclass
class ConversationHistory:
    """Manages conversation history."""
    messages: List[Message] = field(default_factory=list)
    
    def add_message(self, message: Message):
        """Add a message to history."""
        self.messages.append(message)
    
    def add_user_message(self, content: str):
        """Add a user message."""
        self.add_message(Message(role="user", content=content))
    
    def add_assistant_message(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None):
        """Add an assistant message."""
        self.add_message(Message(role="assistant", content=content, tool_calls=tool_calls))
    
    def add_system_message(self, content: str):
        """Add a system message."""
        self.add_message(Message(role="system", content=content))
    
    def add_tool_message(self, content: str, tool_call_id: str, name: str):
        """Add a tool result message."""
        self.add_message(Message(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name
        ))
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert conversation history to list of dicts."""
        return [msg.to_dict() for msg in self.messages]
    
    def get_last_n_messages(self, n: int) -> List[Dict[str, Any]]:
        """Get the last n messages."""
        return [msg.to_dict() for msg in self.messages[-n:]]
    
    def clear(self):
        """Clear conversation history."""
        self.messages.clear()
