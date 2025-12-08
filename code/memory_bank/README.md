# Memory Bank Usage Guide

## 1. Plug-and-Play Design

Memory Bank uses a **fully plug-and-play design**, you can choose whether to enable it:

### Enable

Set environment variables:
```bash
export USE_MEMORY_BANK=1
export MEMORY_IMPORTANCE_THRESHOLD=0.3  # Optional, default 0.3
```

### Disable

Do not set `USE_MEMORY_BANK` or set it to `0`:
```bash
# Do not set, or
export USE_MEMORY_BANK=0
```

**Key**: If disabled, code behavior remains exactly the same as before (backward compatible).

## 2. Workflow

### Forward Stage (Retrieve Memories)
- In `LLMNeuron.activate()`, if `memory_bank` is not None, it will retrieve relevant memories
- Each agent filters by role, only seeing memories for its own role
- If memory bank is empty, returns empty list, does not affect reasoning

### Backward Stage (Store Memories)
- In `llmlp_listwise_mmlu.py` main loop, after backward completes, extract memories
- Only store memories with importance >= threshold
- Memories are stored by agent's role, only agents of that role can see them

## 3. File Structure

```
memory_bank/
├── __init__.py          # Export main classes
├── types.py             # Data type definitions
├── bank.py              # MemoryBank (storage and retrieval)
├── manager.py           # MemoryManager (decision operations)
├── prompts.py           # LLM prompt templates
└── README.md           # This document
```

## 4. Usage Examples

### Basic Usage (Enable Memory Bank)

```bash
# Set environment variables
export USE_MEMORY_BANK=1
export MEMORY_IMPORTANCE_THRESHOLD=0.3

# Run script
python llmlp_listwise_mmlu.py \
    data/MMLU/test/abstract_algebra_test.csv \
    exp_name \
    gpt-3.5-turbo \
    output_dir \
    "['Mathematician', 'Economist', 'Doctor', 'Lawyer']"
```

### Disable Memory Bank

```bash
# Do not set USE_MEMORY_BANK, or set to 0
unset USE_MEMORY_BANK
# or
export USE_MEMORY_BANK=0

# Run script (behavior exactly the same as before)
python llmlp_listwise_mmlu.py ...
```

## 5. Memory Storage

- Memory file saved at: `{DIR_NAME}/memory_bank.json`
- Auto-save every 100 questions
- Final save when program ends

## 6. Role-based Specialization

- **Mathematician**: Can only see memories with owner="Mathematician"
- **Doctor**: Can only see memories with owner="Doctor"
- **Economist**: Can only see memories with owner="Economist"
- etc...

This ensures each agent focuses on memories in their own domain, avoiding interference from irrelevant information.

## 7. Configuration Options

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `USE_MEMORY_BANK` | Whether to enable Memory Bank | `0` (disabled) |
| `MEMORY_IMPORTANCE_THRESHOLD` | Importance threshold for memory extraction | `0.3` |

## 8. Notes

1. **First Run**: Memory bank is empty, system performs normal reasoning (progressive enhancement)
2. **Backward Compatible**: If `USE_MEMORY_BANK=0` or not set, behavior is exactly the same as before
3. **Error Handling**: If memory bank errors occur, normal reasoning is not affected (graceful degradation)
4. **Performance Impact**: Enabling memory bank adds a small number of LLM calls (for decision-making), but retrieval itself is fast

## 9. Troubleshooting

### Memory Bank Not Enabled
- Check if `USE_MEMORY_BANK` environment variable is set to `1`
- Check if you see "Memory bank disabled" message

### Import Error
- Ensure `memory_bank` directory is in Python path
- Check if `code/memory_bank/__init__.py` exists

### Memories Not Saved
- Check write permissions
- Check if `DIR_NAME` directory exists
