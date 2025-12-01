#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Baseline experiment runner - single LLM call per question
# No multi-agent debate, no importance scoring
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
MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct-Turbo-Free}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dir="${DATA_DIR:-$REPO_ROOT/data/MMLU/evaluation}"
exp_name="baseline"
MAX_PARALLEL="${MAX_PARALLEL:-10}"

# Output folder name
OUT_DIR="${exp_name}_$(basename "$MODEL" | tr '/' '_' | tr '.' '_')"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Running Baseline MMLU Evaluation"
echo "Model: $MODEL"
echo "Data directory: $dir"
echo "Output directory: $OUT_DIR"
echo "Max parallel jobs: $MAX_PARALLEL"
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

shopt -s nullglob
for file in "$dir"/*.csv; do
  filename="$(basename -- "$file")"
  filename="${filename%.*}"

  RES_NAME="$OUT_DIR/${filename}_baseline.txt"
  LOG_NAME="$OUT_DIR/${filename}_baseline.log"

  # Python writes 6 lines: accs, resp_cnts, empty importances, prompt_tokens, completion_tokens
  if [[ -f "$RES_NAME" ]] && [[ "$(wc -l < "$RES_NAME")" -ge 6 ]]; then
    echo "Skip $filename (already done)"
    continue
  fi

  # backpressure on background jobs
  while (( $(active_jobs) >= MAX_PARALLEL )); do
    wait_any
  done

  echo "Running $filename → $LOG_NAME"
  (
    echo "[START] $(date -Iseconds) file=$file model=$MODEL"
    SECONDS=0
    # IMPORTANT: redirect stdout first, then stderr
    python "$SCRIPT_DIR/baseline_mmlu.py" \
           "$file" "$filename" "$MODEL" "$OUT_DIR" \
           > "$LOG_NAME" 2>&1
    status=$?
    echo "[END] $(date -Iseconds) file=$file status=$status elapsed_sec=$SECONDS" >> "$LOG_NAME"
    exit $status
  ) &
done

# Wait for any remaining jobs; keep the script alive even if some fail
fails=0
for pid in $(jobs -rp); do
  if ! wait "$pid"; then fails=$((fails+1)); fi
done

echo ""
echo "=========================================="
echo "All baseline jobs finished (failures: $fails)."
echo "=========================================="

# Calculate average accuracy across all test files
if command -v python &> /dev/null; then
  echo ""
  echo "Computing aggregate statistics..."
  python << 'EOF'
import os
import sys
import glob

out_dir = sys.argv[1] if len(sys.argv) > 1 else "baseline_*"
txt_files = glob.glob(f"{out_dir}/*_baseline.txt")

if not txt_files:
    print("No results found.")
    sys.exit(0)

accuracies = []
total_prompt_tokens = 0
total_completion_tokens = 0

for txt_file in txt_files:
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 6:
            # Line 1 has format: [True, False, ...] accuracy
            acc_line = lines[0].strip()
            if ' ' in acc_line:
                acc = float(acc_line.split()[-1])
                accuracies.append(acc)
            # Lines 5 and 6 have token counts
            total_prompt_tokens += int(lines[4].strip())
            total_completion_tokens += int(lines[5].strip())

if accuracies:
    avg_acc = sum(accuracies) / len(accuracies)
    print(f"\nAggregate Results:")
    print(f"  Files evaluated: {len(accuracies)}")
    print(f"  Average accuracy: {avg_acc:.4f}")
    print(f"  Total prompt tokens: {total_prompt_tokens:,}")
    print(f"  Total completion tokens: {total_completion_tokens:,}")
    print(f"  Total tokens: {total_prompt_tokens + total_completion_tokens:,}")
else:
    print("No valid results to aggregate.")
EOF
fi

# ------------------------------------------------------------
# Optional: Compute bootstrap CI metrics
# ------------------------------------------------------------
if [[ "${COMPUTE_METRICS:-1}" == "1" ]]; then
  echo ""
  echo "=========================================="
  echo "Computing bootstrap confidence intervals..."
  echo "=========================================="
  python "$SCRIPT_DIR/compute_baseline_metrics.py" \
    "$OUT_DIR" \
    "$MODEL" \
    "${N_BOOT:-1000}"
fi
