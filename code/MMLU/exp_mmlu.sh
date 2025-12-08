#!/usr/bin/env bash
set -euo pipefail

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
# meta-llama/Llama-3.2-3B-Instruct-Turbo -> change the model here
# ------------------------------------------------------------
MODEL="${MODEL:-openai/gpt-oss-20b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dir="$REPO_ROOT/data/MMLU/small_team_selection"
exp_name="mmlu_downsampled"
ROLES="['Economist','Doctor','Lawyer','Mathematician','Psychologist','Programmer','Historian']"
MAX_PARALLEL="${MAX_PARALLEL:-24}"

# Output folder name must match what anal_imp.sh expects
OUT_DIR="${exp_name}_$(echo "$ROLES" | tr -d "[]' " | tr ',' '_')"
mkdir -p "$OUT_DIR"

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

  RES_NAME="$OUT_DIR/${filename}_73.txt"
  LOG_NAME="$OUT_DIR/${filename}_73.log"

  # Python writes 6 lines: accs, resp_cnts, importances, avg_importances, prompt_tokens, completion_tokens
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
    python "$SCRIPT_DIR/llmlp_listwise_mmlu.py" \
           "$file" "$filename" "$MODEL" "$exp_name" "$ROLES" \
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
echo "All python jobs finished (failures: $fails)."

# Run post-processing from the MMLU folder so relative paths match
( cd "$SCRIPT_DIR" && bash anal_imp.sh )
