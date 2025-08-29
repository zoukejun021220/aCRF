"""Debug management utilities"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DebugManager:
    """Manages debug information and conversations"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.debug_conversations = [] if debug_mode else None
        self.current_conversation = None
        
    def start_conversation(self, field_data: Dict[str, Any]):
        """Start a new debug conversation"""
        if not self.debug_mode:
            return
            
        self.current_conversation = {
            "field": field_data,
            "steps": [],
            "timestamp": datetime.now().isoformat()
        }
        
    def log_step(self, step_name: str, result: Any, confidence: float = None):
        """Log a step in the current conversation"""
        if not self.debug_mode or not self.current_conversation:
            return
            
        step_data = {
            "step": step_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        if confidence is not None:
            step_data["confidence"] = confidence
            
        self.current_conversation["steps"].append(step_data)
        
    def log_llm_call(self, prompt: str, response: str, context: str = None):
        """Log an LLM call"""
        if not self.debug_mode or not self.current_conversation:
            return
            
        llm_data = {
            "type": "llm_call",
            "context": context,
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        if "llm_calls" not in self.current_conversation:
            self.current_conversation["llm_calls"] = []
            
        self.current_conversation["llm_calls"].append(llm_data)
        
    def end_conversation(self):
        """End current conversation and save it"""
        if not self.debug_mode or not self.current_conversation:
            return
            
        self.debug_conversations.append(self.current_conversation)
        self.current_conversation = None
        
    def save_debug_json(self, output_path: str, additional_data: Dict[str, Any] = None):
        """Save all debug information to JSON"""
        if not self.debug_mode:
            logger.info("Debug mode not enabled, no debug data to save")
            return
            
        debug_data = {
            "debug_mode": True,
            "timestamp": datetime.now().isoformat(),
            "total_conversations": len(self.debug_conversations) if self.debug_conversations else 0,
            "conversations": self.debug_conversations or []
        }
        
        # Add additional data if provided
        if additional_data:
            debug_data.update(additional_data)
            
        # Analyze conversations
        if self.debug_conversations:
            debug_data["analysis"] = self._analyze_conversations()
            
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(debug_data, f, indent=2)
            
        logger.info(f"Debug data saved to {output_path}")
        
    def _analyze_conversations(self) -> Dict[str, Any]:
        """Analyze debug conversations for patterns"""
        analysis = {
            "total_fields": len(self.debug_conversations),
            "steps_distribution": {},
            "llm_calls": 0,
            "average_steps": 0
        }
        
        total_steps = 0
        
        for conv in self.debug_conversations:
            # Count steps
            steps = conv.get("steps", [])
            total_steps += len(steps)
            
            for step in steps:
                step_name = step.get("step", "unknown")
                analysis["steps_distribution"][step_name] = \
                    analysis["steps_distribution"].get(step_name, 0) + 1
                    
            # Count LLM calls
            llm_calls = conv.get("llm_calls", [])
            analysis["llm_calls"] += len(llm_calls)
            
        if self.debug_conversations:
            analysis["average_steps"] = total_steps / len(self.debug_conversations)
            
        return analysis
        
    def get_conversation_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all conversations"""
        if not self.debug_conversations:
            return []
            
        summaries = []
        
        for conv in self.debug_conversations:
            field = conv.get("field", {})
            steps = conv.get("steps", [])
            
            # Get final result
            final_result = None
            for step in reversed(steps):
                if step.get("step") == "Variable Selection":
                    final_result = step.get("result")
                    break
                    
            summary = {
                "label": field.get("label", ""),
                "type": field.get("type", ""),
                "steps_count": len(steps),
                "final_annotation": final_result,
                "timestamp": conv.get("timestamp")
            }
            
            summaries.append(summary)
            
        return summaries