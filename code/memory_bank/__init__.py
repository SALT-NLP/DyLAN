"""
Memory Bank Package

Provides inference-time memory management functionality, supports role-based specialized memories.
"""

from .types import MemoryEntry, MemoryOperation, MemoryUpdateEvent
from .bank import MemoryBank
from .manager import MemoryManager

__all__ = [
    "MemoryEntry",
    "MemoryOperation",
    "MemoryUpdateEvent",
    "MemoryBank",
    "MemoryManager",
]

