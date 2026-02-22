"""Trace schemas for tracking agent execution."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


@dataclass
class TraceStep:
    """Represents a single step in agent execution."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: str = ""  # 'llm_call', 'tool_call', 'reasoning', 'error'
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata
        }


@dataclass
class AgentTrace:
    """Represents a complete trace of agent execution."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    steps: List[TraceStep] = field(default_factory=list)
    status: str = "running"  # 'running', 'completed', 'failed'
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: TraceStep):
        """Add a step to the trace."""
        self.steps.append(step)
    
    def complete(self):
        """Mark trace as completed."""
        self.status = "completed"
        self.end_time = datetime.now()
    
    def fail(self, error_message: str):
        """Mark trace as failed."""
        self.status = "failed"
        self.error = error_message
        self.end_time = datetime.now()
    
    def get_duration_ms(self) -> Optional[float]:
        """Get total duration in milliseconds."""
        if self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status,
            "error": self.error,
            "duration_ms": self.get_duration_ms(),
            "metadata": self.metadata
        }
