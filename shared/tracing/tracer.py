"""Tracer for tracking agent execution."""
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

from ..schemas.trace_schema import AgentTrace, TraceStep


class Tracer:
    """Tracer for tracking agent execution steps."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.current_trace: Optional[AgentTrace] = None
        self.session_id = session_id
    
    def start_trace(self, metadata: Optional[Dict[str, Any]] = None) -> AgentTrace:
        """Start a new trace."""
        self.current_trace = AgentTrace(
            session_id=self.session_id,
            metadata=metadata or {}
        )
        return self.current_trace
    
    def end_trace(self):
        """End the current trace."""
        if self.current_trace:
            self.current_trace.complete()
    
    def fail_trace(self, error_message: str):
        """Mark the current trace as failed."""
        if self.current_trace:
            self.current_trace.fail(error_message)
    
    @contextmanager
    def trace_step(
        self,
        step_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing a step.
        
        Usage:
            with tracer.trace_step("llm_call", input_data={"prompt": "..."}):
                # Do work
                pass
        """
        step = TraceStep(
            step_type=step_type,
            input_data=input_data,
            metadata=metadata or {}
        )
        
        start_time = time.time()
        
        try:
            yield step
        except Exception as e:
            step.output_data = {"error": str(e)}
            step.metadata["error"] = True
            raise
        finally:
            end_time = time.time()
            step.duration_ms = (end_time - start_time) * 1000
            
            if self.current_trace:
                self.current_trace.add_step(step)
    
    def add_step(
        self,
        step_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a step to the current trace."""
        step = TraceStep(
            step_type=step_type,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        
        if self.current_trace:
            self.current_trace.add_step(step)
    
    def get_trace(self) -> Optional[AgentTrace]:
        """Get the current trace."""
        return self.current_trace
    
    def save_trace_to_file(self, filepath: str) -> bool:
        """Save the current trace to a JSON file.
        
        Args:
            filepath: Path to save the trace file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_trace:
            return False
        
        try:
            # Convert trace to dict
            trace_dict = self.current_trace.to_dict()
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(trace_dict, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving trace: {e}")
            return False
    
    def get_trace_json(self) -> Optional[str]:
        """Get the current trace as JSON string.
        
        Returns:
            JSON string of the trace or None
        """
        if not self.current_trace:
            return None
        
        try:
            trace_dict = self.current_trace.to_dict()
            return json.dumps(trace_dict, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error converting trace to JSON: {e}")
            return None
    
    def get_trace_summary(self) -> Optional[Dict[str, Any]]:
        """Get a summary of the current trace.
        
        Returns:
            Dictionary with trace summary
        """
        if not self.current_trace:
            return None
        
        trace = self.current_trace
        
        # Count step types
        step_counts = {}
        total_duration = 0
        
        for step in trace.steps:
            step_type = step.step_type
            step_counts[step_type] = step_counts.get(step_type, 0) + 1
            if step.duration_ms:
                total_duration += step.duration_ms
        
        return {
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "status": trace.status,
            "total_steps": len(trace.steps),
            "step_types": step_counts,
            "total_duration_ms": total_duration,
            "overall_duration_ms": trace.get_duration_ms(),
            "start_time": trace.start_time.isoformat(),
            "end_time": trace.end_time.isoformat() if trace.end_time else None,
            "error": trace.error
        }
