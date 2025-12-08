# #!/usr/bin/env bash
# set -euo pipefail

# # ------------------------------------------------------------
# # Optional: load API keys from repo-root .env (so Python sees them)
# # ------------------------------------------------------------
# REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
# if [[ -f "$REPO_ROOT/.env" ]]; then
#   set -a
#   # shellcheck disable=SC1090
#   source "$REPO_ROOT/.env"
#   set +a
# fi

# # ------------------------------------------------------------
# # Configurable knobs (can override via env)
# # meta-llama/Llama-3.2-3B-Instruct-Turbo -> change the model here
# # ------------------------------------------------------------
# MODEL="${MODEL:-openai/gpt-oss-20b}"
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# dir="$REPO_ROOT/data/math/math_json"
# exp_name="math_downsampled"
# ROLES="['Economist','Doctor','Lawyer','Mathematician','Psychologist','Programmer','Historian']"
# MAX_PARALLEL="${MAX_PARALLEL:-4}"

# # Output folder name must match what anal_imp.sh expects
# OUT_DIR="${exp_name}_$(echo "$ROLES" | tr -d "[]' " | tr ',' '_')"
# mkdir -p "$OUT_DIR"

# active_jobs() { jobs -rp | wc -l; }

# # Wait for any background job to finish (compatible with bash 3.2+)
# wait_any() {
#   local pids=($(jobs -rp))
#   if [[ ${#pids[@]} -gt 0 ]]; then
#     wait "${pids[0]}" || true
#   fi
# }

# shopt -s nullglob
# for file in "$dir"/*.csv; do
#   filename="$(basename -- "$file")"
#   filename="${filename%.*}"

#   RES_NAME="$OUT_DIR/${filename}_73.txt"
#   LOG_NAME="$OUT_DIR/${filename}_73.log"

#   # Python writes 6 lines: accs, resp_cnts, importances, avg_importances, prompt_tokens, completion_tokens
#   if [[ -f "$RES_NAME" ]] && [[ "$(wc -l < "$RES_NAME")" -ge 6 ]]; then
#     echo "Skip $filename (already done)"
#     continue
#   fi

#   # backpressure on background jobs
#   while (( $(active_jobs) >= MAX_PARALLEL )); do
#     wait_any
#   done

#   echo "Running $filename → $LOG_NAME"
#   (
#     echo "[START] $(date -Iseconds) file=$file model=$MODEL"
#     SECONDS=0
#     # IMPORTANT: redirect stdout first, then stderr
#     python "$SCRIPT_DIR/llmlp_listwise_math.py" \
#            "$file" "$filename" "$MODEL" "$exp_name" "$ROLES" \
#            > "$LOG_NAME" 2>&1
#     status=$?
#     echo "[END] $(date -Iseconds) file=$file status=$status elapsed_sec=$SECONDS" >> "$LOG_NAME"
#     exit $status
#   ) &
# done

# # Wait for any remaining jobs; keep the script alive even if some fail
# fails=0
# for pid in $(jobs -rp); do
#   if ! wait "$pid"; then fails=$((fails+1)); fi
# done
# echo "All python jobs finished (failures: $fails)."

# # Run post-processing from the MMLU folder so relative paths match
# ( cd "$SCRIPT_DIR" && bash anal_imp.sh )

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

export OPENAI_API_KEY="${OPENAI_API_KEY:-$TOGETHER_API_KEY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-$TOGETHER_BASE_URL}"

MODEL="${MODEL:-openai/gpt-oss-20b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dir="${DIR_OVERRIDE:-$REPO_ROOT/data/math_json/small_team_selection}"

exp_name="math_downsampled"

ROLES="['Mathematician','AlgebraExpert','CountingProbabilitySpecialist','GeometryWizard','IntermediateAlgebraMaestro','NumberTheoryScholar','PrecalculusGuru']"

MAX_PARALLEL="${MAX_PARALLEL:-4}"

OUT_DIR="${exp_name}_$(echo "$ROLES" | tr -d "[]' " | tr ',' '_')"
mkdir -p "$OUT_DIR"

active_jobs(){ jobs -rp | wc -l; }
wait_any(){
  local pids=($(jobs -rp))
  if [[ ${#pids[@]} -gt 0 ]]; then
    wait "${pids[0]}" || true
  fi
}

# --- main loop ---
for subdir in "$dir"/*; do
  [[ -d "$subdir" ]] || continue
  echo "Processing $subdir"

  json_files=()
  while IFS= read -r line; do
    json_files+=("$line")
  done < <(find "$subdir" -type f -name "*.json" -exec basename {} .json \; | LC_ALL=C sort -n)

  total=${#json_files[@]}
  (( total == 0 )) && { echo "No JSON in $subdir, skip"; continue; }

  loops=$(( (total + 99) / 100 ))
  base=$(basename "$subdir")

  for ((i=0;i<loops;i++)); do
    start=$((i*100))
    end=$(((i+1)*100-1))
    (( end >= total )) && end=$((total-1))

    minf=${json_files[$start]}
    maxf=${json_files[$end]}

    batch_name="${base}_${minf}_${maxf}"
    RES_NAME="$OUT_DIR/${batch_name}_73.txt"
    LOG_NAME="$OUT_DIR/${batch_name}_73.log"

    if [[ -f "$RES_NAME" ]] && [[ "$(wc -l < "$RES_NAME")" -ge 6 ]]; then
      echo "Skip $batch_name (already done)"
      continue
    fi

    echo "Launch: $subdir  $minf → $maxf"
    while (( $(active_jobs) >= MAX_PARALLEL )); do wait_any; done

    (
      echo "[START] $(date -Iseconds) subdir=$subdir minf=$minf maxf=$maxf model=$MODEL"
      SECONDS=0
      python "$SCRIPT_DIR/llmlp_listwise_math.py" \
             "$subdir" "$minf" "$maxf" "$MODEL" "$exp_name" "$ROLES" \
             > "$LOG_NAME" 2>&1
      status=$?
      echo "[END] $(date -Iseconds) subdir=$subdir minf=$minf maxf=$maxf status=$status elapsed_sec=$SECONDS" >> "$LOG_NAME"
      exit $status
    ) &
  done

  echo "Queued batches for $subdir"
done

fails=0
for pid in $(jobs -rp); do
  if ! wait "$pid"; then fails=$((fails+1)); fi
done
echo "All python jobs finished (failures: $fails)."

( cd "$SCRIPT_DIR" && bash anal_imp_math.sh )
