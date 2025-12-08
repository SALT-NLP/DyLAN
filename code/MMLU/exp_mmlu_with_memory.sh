#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Memory Bank Enabled MMLU Experiment Script
# Experiment script with Memory Bank enabled, can view results
# ------------------------------------------------------------

# ------------------------------------------------------------
# Optional: load API keys from repo-root .env (so Python sees them)
# ------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

# ------------------------------------------------------------
# Configurable knobs (can override via env)
# ------------------------------------------------------------
MODEL="${MODEL:-openai/gpt-oss-20b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Can choose to use small dataset for quick testing, or full dataset
# Small dataset: data/MMLU/small_team_selection or data/MMLU/one_percent_team_selection
# Full dataset: data/MMLU/test
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/MMLU/small_team_selection}"

exp_name="mmlu_with_memory"
ROLES="['Economist','Doctor','Lawyer','Mathematician','Psychologist','Programmer','Historian']"

# Memory Bank configuration
USE_MEMORY_BANK="${USE_MEMORY_BANK:-1}"  # Enabled by default
MEMORY_IMPORTANCE_THRESHOLD="${MEMORY_IMPORTANCE_THRESHOLD:-0.3}"

# Parallelism (can set to 1 for sequential execution, easier to view output)
MAX_PARALLEL="${MAX_PARALLEL:-1}"

# Output folder name
OUT_DIR="${exp_name}_$(echo "$ROLES" | tr -d "[]' " | tr ',' '_')"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Memory Bank MMLU Experiment"
echo "=========================================="
echo "Model: $MODEL"
echo "Data Dir: $DATA_DIR"
echo "Output Dir: $OUT_DIR"
echo "Memory Bank: $([ "$USE_MEMORY_BANK" = "1" ] && echo "ENABLED" || echo "DISABLED")"
echo "Importance Threshold: $MEMORY_IMPORTANCE_THRESHOLD"
echo "Max Parallel: $MAX_PARALLEL"
echo "=========================================="
echo ""

active_jobs() { jobs -rp | wc -l; }

# Wait for any background job to finish (compatible with bash 3.2+)
wait_any() {
  local pids=($(jobs -rp))
  if [[ ${#pids[@]} -gt 0 ]]; then
    wait "${pids[0]}" || true
  fi
}

# Statistics counters
total_files=0
completed_files=0
failed_files=0

shopt -s nullglob
for file in "$DATA_DIR"/*.csv; do
  filename="$(basename -- "$file")"
  filename="${filename%.*}"

  RES_NAME="$OUT_DIR/${filename}_73.txt"
  LOG_NAME="$OUT_DIR/${filename}_73.log"

  # Python writes 6 lines: accs, resp_cnts, importances, avg_importances, prompt_tokens, completion_tokens
  if [[ -f "$RES_NAME" ]] && [[ "$(wc -l < "$RES_NAME")" -ge 6 ]]; then
    echo "Skip $filename (already done)"
    continue
  fi

  total_files=$((total_files + 1))

  # backpressure on background jobs
  while (( $(active_jobs) >= MAX_PARALLEL )); do
    wait_any
  done

  echo "Running $filename -> $LOG_NAME"
  (
    echo "[START] $(date -Iseconds) file=$file model=$MODEL USE_MEMORY_BANK=$USE_MEMORY_BANK"
    SECONDS=0
    
    # Set Memory Bank environment variables
    export USE_MEMORY_BANK="$USE_MEMORY_BANK"
    export MEMORY_IMPORTANCE_THRESHOLD="$MEMORY_IMPORTANCE_THRESHOLD"
    
    # IMPORTANT: redirect stdout first, then stderr
    if python "$SCRIPT_DIR/llmlp_listwise_mmlu.py" \
           "$file" "$filename" "$MODEL" "$exp_name" "$ROLES" \
           > "$LOG_NAME" 2>&1; then
      status=0
      completed_files=$((completed_files + 1))
    else
      status=1
      failed_files=$((failed_files + 1))
    fi
    
    elapsed=$SECONDS
    echo "[END] $(date -Iseconds) file=$file status=$status elapsed_sec=$elapsed" >> "$LOG_NAME"
    
    # Display brief results
    if [[ $status -eq 0 ]] && [[ -f "$RES_NAME" ]]; then
      # Extract accuracy (last number in first line)
      accuracy=$(head -n 1 "$RES_NAME" | awk '{print $NF}')
      echo "$filename completed (accuracy: $accuracy, elapsed: ${elapsed}s)"
      
      # If Memory Bank is enabled, show memory count
      if [[ "$USE_MEMORY_BANK" = "1" ]]; then
        memory_file="$OUT_DIR/memory_bank.json"
        if [[ -f "$memory_file" ]]; then
          # Use Python to quickly count memories (by role)
          memory_count=$(python3 -c "
import json
try:
    with open('$memory_file', 'r') as f:
        data = json.load(f)
    entries = data.get('entries', [])
    print(len(entries))
except:
    print('0')
" 2>/dev/null || echo "0")
          if [[ "$memory_count" != "0" ]]; then
            echo "   Memory Bank: $memory_count entries"
          fi
        fi
      fi
    else
      echo "$filename failed (check $LOG_NAME)"
    fi
    
    exit $status
  ) &
done

# Wait for any remaining jobs; keep the script alive even if some fail
echo ""
echo "Waiting for all jobs to complete..."
for pid in $(jobs -rp); do
  if ! wait "$pid"; then
    failed_files=$((failed_files + 1))
  fi
done

echo ""
echo "=========================================="
echo "Experiment Summary"
echo "=========================================="
echo "Total files: $total_files"
echo "Completed: $completed_files"
echo "Failed: $failed_files"
echo "Skipped: $((total_files - completed_files - failed_files))"
echo ""

# Display overall results
if [[ $completed_files -gt 0 ]]; then
  echo "Calculating overall results..."
  
  # Calculate average accuracy
  total_acc=0
  count=0
  for res_file in "$OUT_DIR"/*_73.txt; do
    if [[ -f "$res_file" ]]; then
      acc=$(head -n 1 "$res_file" | awk '{print $NF}')
      if [[ -n "$acc" ]] && [[ "$acc" != "[]" ]]; then
        total_acc=$(echo "$total_acc + $acc" | bc -l 2>/dev/null || echo "$total_acc")
        count=$((count + 1))
      fi
    fi
  done
  
  if [[ $count -gt 0 ]]; then
    avg_acc=$(echo "scale=4; $total_acc / $count" | bc -l 2>/dev/null || echo "N/A")
    echo "Average Accuracy: $avg_acc"
  fi
  
  # Display Memory Bank statistics
  if [[ "$USE_MEMORY_BANK" = "1" ]]; then
    memory_file="$OUT_DIR/memory_bank.json"
    if [[ -f "$memory_file" ]]; then
      echo ""
      echo "Memory Bank Statistics:"
      python3 -c "
import json
from collections import Counter

try:
    with open('$memory_file', 'r') as f:
        data = json.load(f)
    entries = data.get('entries', [])
    
    if entries:
        owners = [e.get('owner', 'unknown') for e in entries]
        owner_counts = Counter(owners)
        
        print(f'  Total memories: {len(entries)}')
        print('  By role:')
        for owner, count in sorted(owner_counts.items()):
            print(f'    {owner}: {count}')
    else:
        print('  No memories stored yet')
except Exception as e:
    print(f'  Error reading memory bank: {e}')
" 2>/dev/null || echo "  Could not read memory bank statistics"
    fi
  fi
fi

echo ""
echo "Results saved in: $OUT_DIR"
echo "Logs saved in: $OUT_DIR/*.log"
if [[ "$USE_MEMORY_BANK" = "1" ]]; then
  echo "Memory Bank: $OUT_DIR/memory_bank.json"
fi
echo ""

# Run post-processing from the MMLU folder so relative paths match
if [[ -f "$SCRIPT_DIR/anal_imp.sh" ]]; then
  echo "Running post-processing..."
  ( cd "$SCRIPT_DIR" && bash anal_imp.sh ) || echo "Post-processing failed (non-critical)"
fi

echo "Experiment completed!"

