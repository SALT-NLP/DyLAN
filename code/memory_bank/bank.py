"""
Memory Bank core class.

Provides memory storage, retrieval, similarity search, and other functions.
Supports role-based filtering to ensure each agent can only access memories related to its role.
"""
import json
import os
import uuid
from typing import Callable, Optional
import numpy as np

from .types import MemoryEntry, MemoryUpdateEvent, MemoryOperation


class MemoryBank:
    """
    Memory Bank class.
    
    Responsible for storing all memory entries and providing similarity-based retrieval.
    Uses simple vector similarity search (via injected embedding function).
    Supports role-based filtering to ensure each agent only sees relevant memories.
    
    Attributes:
        entries: Dictionary of all memory entries {id: MemoryEntry}
        embeddings: Memory embedding vectors {id: np.ndarray}
        embed_fn: Embedding function that converts text to vectors
    """
    
    def __init__(self, embed_fn: Optional[Callable[[str], np.ndarray]] = None):
        """
        Initialize memory bank.
        
        Args:
            embed_fn: Optional embedding function. If None, uses simple word overlap similarity.
        """
        self.entries: dict[str, MemoryEntry] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        
        if embed_fn is None:
            # Default: use simple word overlap similarity (simple TF-IDF style implementation)
            self.embed_fn = self._simple_text_embedding
        else:
            self.embed_fn = embed_fn
    
    def _simple_text_embedding(self, text: str) -> np.ndarray:
        """
        Simple text embedding (based on word frequency).
        
        This is a fallback implementation. In practice, it's recommended to inject a better embedding function
        (e.g., sentence-transformers, OpenAI embeddings, etc.).
        """
        words = text.lower().split()
        # Simple word frequency vector (vocabulary size dimension, simplified to fixed dimension here)
        # In practice, should use a real embedding model
        vec = np.zeros(128)  # Fixed dimension
        for i, word in enumerate(words[:128]):
            # Simple hash-based embedding
            hash_val = hash(word) % 128
            vec[hash_val] += 1.0 / (i + 1)  # Position weighting
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    
    def add(self, entry: MemoryEntry) -> MemoryUpdateEvent:
        """
        Add a new memory entry.
        
        Args:
            entry: Memory entry to add
        
        Returns:
            MemoryUpdateEvent recording this operation
        """
        if entry.id in self.entries:
            raise ValueError(f"Memory entry {entry.id} already exists. Use update() instead.")
        
        self.entries[entry.id] = entry
        # Compute and store embedding
        self.embeddings[entry.id] = self.embed_fn(entry.text)
        
        return MemoryUpdateEvent(
            entry_id=entry.id,
            operation=MemoryOperation.ADD,
            old_text=None,
            new_text=entry.text,
            reason="New memory added"
        )
    
    def update(self, entry_id: str, new_text: str, reason: str = "") -> MemoryUpdateEvent:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: Memory ID to update
            new_text: New memory text
            reason: Update reason
        
        Returns:
            MemoryUpdateEvent recording this operation
        """
        if entry_id not in self.entries:
            raise ValueError(f"Memory entry {entry_id} not found.")
        
        old_entry = self.entries[entry_id]
        old_text = old_entry.text
        
        # Update entry
        old_entry.text = new_text
        old_entry.timestamp = datetime.now()
        # Update embedding
        self.embeddings[entry_id] = self.embed_fn(new_text)
        
        return MemoryUpdateEvent(
            entry_id=entry_id,
            operation=MemoryOperation.UPDATE,
            old_text=old_text,
            new_text=new_text,
            reason=reason or "Memory updated"
        )
    
    def delete(self, entry_id: str, reason: str = "") -> MemoryUpdateEvent:
        """
        Delete a memory entry.
        
        Args:
            entry_id: Memory ID to delete
            reason: Delete reason
        
        Returns:
            MemoryUpdateEvent recording this operation
        """
        if entry_id not in self.entries:
            raise ValueError(f"Memory entry {entry_id} not found.")
        
        old_entry = self.entries[entry_id]
        old_text = old_entry.text
        
        # Delete entry and embedding
        del self.entries[entry_id]
        if entry_id in self.embeddings:
            del self.embeddings[entry_id]
        
        return MemoryUpdateEvent(
            entry_id=entry_id,
            operation=MemoryOperation.DELETE,
            old_text=old_text,
            new_text=None,
            reason=reason or "Memory deleted"
        )
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get memory entry by ID.
        
        Args:
            entry_id: Memory ID
        
        Returns:
            MemoryEntry or None (if not found)
        """
        return self.entries.get(entry_id)
    
    def all_entries(self, owner: Optional[str] = None) -> list[MemoryEntry]:
        """
        Get all memory entries (optionally filtered by owner).
        
        Args:
            owner: Optional owner filter (agent role name)
        
        Returns:
            List of memory entries
        """
        entries = list(self.entries.values())
        if owner is not None:
            entries = [e for e in entries if e.owner == owner]
        return entries
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        owner: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> list[MemoryEntry]:
        """
        Search memory entries based on similarity (supports role-based filtering).
        
        Args:
            query: Query text
            top_k: Return top k most similar results
            owner: Optional owner filter (agent role name, e.g., "Mathematician")
                   If provided, only returns memories for that role
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of memory entries sorted by similarity (returns empty list [] if memory bank is empty)
        """
        # If memory bank is empty, return empty list directly (graceful handling)
        if not self.entries:
            return []
        
        # Compute query embedding
        query_embedding = self.embed_fn(query)
        
        # Compute similarity between all memories and query
        similarities = []
        for entry_id, entry in self.entries.items():
            # Filter by owner (role-based)
            if owner is not None and entry.owner != owner:
                continue
            
            if entry_id in self.embeddings:
                # Compute cosine similarity
                mem_embedding = self.embeddings[entry_id]
                similarity = np.dot(query_embedding, mem_embedding)
                
                # Filter by minimum similarity
                if similarity >= min_similarity:
                    similarities.append((similarity, entry))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in similarities[:top_k]]
    
    def save(self, filepath: str) -> None:
        """
        Save memory bank to JSON file in simplified format.
        
        Format: {
            "AgentRole": {
                "1": "memory_content",
                "2": "memory_content"
            }
        }
        
        Args:
            filepath: Save path
        """
        # Group memories by owner (agent role)
        data = {}
        for entry in self.entries.values():
            if entry.owner not in data:
                data[entry.owner] = {}
            
            # Use a simple counter for each agent's memories
            memory_count = len(data[entry.owner]) + 1
            data[entry.owner][str(memory_count)] = entry.text
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str) -> None:
        """
        Load memory bank from JSON file in simplified format.
        
        Expected format: {
            "AgentRole": {
                "1": "memory_content",
                "2": "memory_content"
            }
        }
        
        Note: Embeddings are recomputed after loading.
        
        Args:
            filepath: File path
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.entries = {}
        self.embeddings = {}
        
        # Handle both old and new formats for backward compatibility
        if "entries" in data:
            # Old format - convert to new structure
            for entry_dict in data["entries"]:
                entry = MemoryEntry.from_dict(entry_dict)
                self.entries[entry.id] = entry
                self.embeddings[entry.id] = self.embed_fn(entry.text)
        else:
            # New format - agent-based dictionary
            for owner, memories in data.items():
                for memory_id, memory_text in memories.items():
                    entry_id = f"mem_{owner}_{memory_id}"
                    entry = MemoryEntry(
                        id=entry_id,
                        owner=owner,
                        text=memory_text
                    )
                    self.entries[entry_id] = entry
                    self.embeddings[entry_id] = self.embed_fn(memory_text)
    
    def create_entry(
        self,
        text: str,
        owner: str = "system"
    ) -> MemoryEntry:
        """
        Create a new memory entry (not automatically added to bank).
        
        This is a convenience method for generating a MemoryEntry with a unique ID.
        
        Args:
            text: Memory text (high-level experience summary)
            owner: Owner (agent role name, e.g., "Mathematician")
        
        Returns:
            New MemoryEntry (not yet added to bank)
        """
        entry_id = f"mem_{uuid.uuid4().hex[:8]}"
        return MemoryEntry(
            id=entry_id,
            owner=owner,
            text=text
        )

