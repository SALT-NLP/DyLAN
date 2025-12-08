"""
Memory Manager.

Uses LLM-driven decision mechanism to determine what operation to perform on memories (ADD/UPDATE/DELETE/NOOP).
"""
import json
import re
from typing import Optional

from .types import MemoryEntry, MemoryOperation, MemoryUpdateEvent
from .bank import MemoryBank
from .prompts import construct_memory_decision_prompt


class MemoryManager:
    """
    Memory Manager class.
    
    Responsible for deciding what operation to perform on new facts (ADD/UPDATE/DELETE/NOOP).
    Uses LLM for decision-making to ensure intelligent memory management.
    """
    
    def __init__(self, memory_bank: MemoryBank, model: str = "gpt-3.5-turbo"):
        """
        Initialize memory manager.
        
        Args:
            memory_bank: MemoryBank instance
            model: LLM model name for decision-making
        """
        self.memory_bank = memory_bank
        self.model = model
    
    def process_fact(
        self,
        fact_summary: str,
        owner: str = "system",
        context: Optional[dict] = None
    ) -> MemoryUpdateEvent:
        """
        Process a new fact and decide what operation to perform.
        
        Args:
            fact_summary: Summary text of the new fact
            owner: Memory owner (agent role name, e.g., "Mathematician")
            context: Optional context information (e.g., importance, is_correct, etc.)
        
        Returns:
            MemoryUpdateEvent recording this operation
        """
        # 1. Retrieve potentially relevant existing memories (only search memories with same owner)
        existing_memories = self.memory_bank.search(
            fact_summary,
            top_k=10,
            owner=owner  # Only search memories with same role
        )
        
        # 2. Build existing memory list (for prompt)
        memory_list = [(mem.id, mem.text) for mem in existing_memories]
        
        # 3. Call LLM to decide operation
        decision = self._decide_operation(fact_summary, memory_list)
        
        # 4. Execute corresponding operation
        if decision["operation"] == "ADD":
            # Create new memory entry
            entry = self.memory_bank.create_entry(
                text=fact_summary,
                owner=owner
            )
            event = self.memory_bank.add(entry)
            event.reason = decision.get("reason", "New memory added")
            return event
        
        elif decision["operation"] == "UPDATE":
            entry_id = decision.get("entry_id")
            if entry_id and entry_id in self.memory_bank.entries:
                event = self.memory_bank.update(
                    entry_id,
                    fact_summary,
                    reason=decision.get("reason", "Memory updated")
                )
                return event
            else:
                # entry_id invalid, fallback to ADD
                entry = self.memory_bank.create_entry(
                    text=fact_summary,
                    owner=owner
                )
                event = self.memory_bank.add(entry)
                event.reason = f"UPDATE failed (invalid entry_id), fallback to ADD: {decision.get('reason', '')}"
                return event
        
        elif decision["operation"] == "DELETE":
            entry_id = decision.get("entry_id")
            if entry_id and entry_id in self.memory_bank.entries:
                event = self.memory_bank.delete(
                    entry_id,
                    reason=decision.get("reason", "Memory deleted")
                )
                return event
            else:
                # entry_id invalid, fallback to NOOP
                return MemoryUpdateEvent(
                    entry_id=None,
                    operation=MemoryOperation.NOOP,
                    old_text=None,
                    new_text=None,
                    reason=f"DELETE failed (invalid entry_id): {decision.get('reason', '')}"
                )
        
        else:  # NOOP
            return MemoryUpdateEvent(
                entry_id=None,
                operation=MemoryOperation.NOOP,
                old_text=None,
                new_text=None,
                reason=decision.get("reason", "No operation needed")
            )
    
    def _decide_operation(
        self,
        fact_summary: str,
        existing_memories: list[tuple[str, str]]
    ) -> dict:
        """
        Use LLM to decide what operation to perform.
        
        Args:
            fact_summary: New fact summary
            existing_memories: Existing memory list [(id, text), ...]
        
        Returns:
            Decision dictionary {"operation": "ADD", "entry_id": "mem_123", "reason": "..."}
        """
        # Build prompt
        messages = construct_memory_decision_prompt(fact_summary, existing_memories)
        
        # Call LLM (using project's generate_answer function)
        try:
            # Try to import from MMLU directory (if memory_bank is under code directory)
            import sys
            import os
            # Add MMLU directory to path (if not in path)
            mmlu_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "MMLU")
            if mmlu_path not in sys.path:
                sys.path.insert(0, mmlu_path)
            from utils import generate_answer
            reply, _, _ = generate_answer(messages, self.model)
        except Exception as e:
            # If LLM call fails, fallback to ADD
            print(f"Warning: LLM decision failed, fallback to ADD: {e}")
            return {
                "operation": "ADD",
                "entry_id": None,
                "reason": f"LLM decision failed, fallback to ADD: {str(e)}"
            }
        
        # Parse JSON returned by LLM
        decision = self._parse_decision_json(reply)
        
        # Validate decision
        if decision is None:
            # Parsing failed, fallback to ADD
            return {
                "operation": "ADD",
                "entry_id": None,
                "reason": "Failed to parse LLM response, fallback to ADD"
            }
        
        # Validate operation value
        valid_operations = ["ADD", "UPDATE", "DELETE", "NOOP"]
        if decision.get("operation") not in valid_operations:
            decision["operation"] = "ADD"
            decision["reason"] = f"Invalid operation, fallback to ADD. Original: {decision.get('reason', '')}"
        
        return decision
    
    def _parse_decision_json(self, text: str) -> Optional[dict]:
        """
        Parse JSON decision from LLM reply.
        
        Args:
            text: LLM reply text
        
        Returns:
            Parsed decision dictionary, returns None if parsing fails
        """
        # Try to parse JSON directly
        try:
            # Find JSON object
            json_match = re.search(r'\{[^{}]*"operation"[^{}]*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except:
            pass
        
        # Try to extract operation field
        try:
            operation_match = re.search(r'"operation"\s*:\s*"([^"]+)"', text)
            entry_id_match = re.search(r'"entry_id"\s*:\s*"([^"]+)"', text)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
            
            if operation_match:
                decision = {
                    "operation": operation_match.group(1),
                    "entry_id": entry_id_match.group(1) if entry_id_match else None,
                    "reason": reason_match.group(1) if reason_match else ""
                }
                # Handle null
                if decision["entry_id"] == "null" or decision["entry_id"] is None:
                    decision["entry_id"] = None
                return decision
        except:
            pass
        
        return None

