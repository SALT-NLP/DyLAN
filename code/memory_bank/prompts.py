"""
Prompt templates for the Memory Bank system.

Defines LLM prompts for MemoryManager.
"""
from typing import Optional


def construct_memory_decision_prompt(
    new_fact: str,
    existing_memories: list[tuple[str, str]],  # [(id, text), ...]
    max_memories: int = 10
) -> list[dict[str, str]]:
    """
    Build prompt for memory management decision.
    
    Let LLM judge what operation to perform on new fact (ADD/UPDATE/DELETE/NOOP).
    
    Args:
        new_fact: New fact summary (from current dialogue turn)
        existing_memories: Existing memory list, format: [(id, text), ...]
        max_memories: Maximum number of existing memories to show (avoid prompt being too long)
    
    Returns:
        Messages list, can be directly used with generate_answer
    """
    # Limit number of memories to show
    memories_to_show = existing_memories[:max_memories]
    
    system_prompt = (
        "You are a memory manager for an LLM agent system. "
        "Your task is to decide how to handle a new fact from the current dialogue turn. "
        "You can ADD a new memory, UPDATE an existing one, DELETE an obsolete one, or do NOOP.\n\n"
        "Guidelines:\n"
        "- ADD: if the fact is new and worth remembering (e.g., successful reasoning patterns, important insights)\n"
        "- UPDATE: if the fact contradicts or refines an existing memory\n"
        "- DELETE: if an existing memory is now obsolete or incorrect\n"
        "- NOOP: if the fact is trivial, temporary, or already well-covered\n\n"
        "Output format: JSON only, like {\"operation\": \"ADD\", \"entry_id\": null, \"reason\": \"...\"} "
        "or {\"operation\": \"UPDATE\", \"entry_id\": \"mem_123\", \"reason\": \"...\"} "
        "or {\"operation\": \"DELETE\", \"entry_id\": \"mem_456\", \"reason\": \"...\"} "
        "or {\"operation\": \"NOOP\", \"entry_id\": null, \"reason\": \"...\"}."
    )
    
    memories_text = ""
    if memories_to_show:
        memories_text = "\n\nExisting memories:\n"
        for mem_id, mem_text in memories_to_show:
            memories_text += f"- [{mem_id}] {mem_text}\n"
    else:
        memories_text = "\n\nNo existing memories."
    
    user_prompt = (
        f"New fact from current dialogue:\n{new_fact}\n"
        f"{memories_text}\n\n"
        "What operation should be performed? Output JSON only."
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

