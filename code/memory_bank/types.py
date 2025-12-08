"""
Data type definitions for the Memory Bank system.

Provides core data structures: MemoryEntry (memory entry), MemoryOperation (operation type),
MemoryUpdateEvent (update event).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MemoryOperation(Enum):
    """Memory operation type enumeration"""
    ADD = "ADD"           # Add new memory
    UPDATE = "UPDATE"     # Update existing memory
    DELETE = "DELETE"     # Delete memory
    NOOP = "NOOP"         # No operation


@dataclass
class MemoryEntry:
    """
    Memory entry data class.
    
    Stores a long-term memory, containing memory content and owner (role).
    
    Attributes:
        id: Unique identifier
        owner: Memory owner (agent role name, e.g., "Mathematician", "Doctor", or "system"/"user")
        text: Natural language memory content (high-level experience summary)
    """
    id: str
    owner: str
    text: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format (for serialization)"""
        return {
            "id": self.id,
            "owner": self.owner,
            "text": self.text
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """Create MemoryEntry from dictionary"""
        return cls(
            id=data["id"],
            owner=data["owner"],
            text=data["text"]
        )


@dataclass
class MemoryUpdateEvent:
    """
    Memory update event data class.
    
    Records detailed information about a memory operation (add/update/delete).
    
    Attributes:
        entry_id: Related memory entry ID (if ADD operation, this is the newly created ID)
        operation: Operation type
        old_text: Text before update (only for UPDATE and DELETE operations)
        new_text: Text after update (only for ADD and UPDATE operations)
        reason: Operation reason (from MemoryManager's decision rationale)
    """
    entry_id: Optional[str]
    operation: MemoryOperation
    old_text: Optional[str]
    new_text: Optional[str]
    reason: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "entry_id": self.entry_id,
            "operation": self.operation.value,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "reason": self.reason
        }

