#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Single-LLM MATH Evaluation + CI Script
# ------------------------------------------------------------
# - One model call per math problem
# - Uses data/math_json/<split>/<subject>/*.json
# - Uses utils.get_math_qa_pairs and utils.is_equiv for grading
# - Reports accuracy, API calls, tokens-in/out with 95% bootstrap CIs
# - Breaks out metrics overall and by math subcategory
# - Optional comparison to pre-selection (7 roles) via IMPORTANCE_CSV
# ------------------------------------------------------------

MODEL="${MODEL:-openai/gpt-oss-20b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Root directory for math evaluation data (each subject is a subdirectory here)
MATH_EVAL_ROOT="${MATH_EVAL_ROOT:-$SCRIPT_DIR/../../data/math_json/evaluation}"

OUTPUT_DIR="${OUTPUT_DIR:-single_llm_math_results}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
N_BOOT="${N_BOOT:-1000}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"

# Optional: CSV from a 7-role importance run (for comparison)
IMPORTANCE_CSV="${IMPORTANCE_CSV:-}"   # e.g. importance_math_1to7.csv

usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Run a single-LLM MATH baseline (one model call per problem) and report:
accuracy, API calls, tokens-in, tokens-out with 95% bootstrap CIs.

OPTIONS:
  -m, --model MODEL              Chat model id (default: ${MODEL})
  -d, --dataset DIR              Path to math eval root (default: ${MATH_EVAL_ROOT})
  -o, --output DIR               Output directory (default: ${OUTPUT_DIR})
  -p, --max-parallel NUM         Max parallel subjects (default: ${MAX_PARALLEL})
  --n-boot NUM                   Bootstrap replicates for CI (default: ${N_BOOT})
  -t, --temperature FLOAT        Temperature (default: ${TEMPERATURE})
  --top-p FLOAT                  top_p (default: ${TOP_P})
  -i, --importance-csv FILE      (optional) importance_math_1to7.csv for 7-role comparison
  -h, --help                     Show help

Examples:
  $0
  $0 --model "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
  N_BOOT=2000 $0 --dataset "../../data/math_json/medium_team_selection" --max-parallel 8
  IMPORTANCE_CSV=importance_math_1to7.csv $0
EOF
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

# -------- arg parsing ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="$2"; shift 2;;
    -d|--dataset) MATH_EVAL_ROOT="$2"; shift 2;;
    -o|--output) OUTPUT_DIR="$2"; shift 2;;
    -p|--max-parallel) MAX_PARALLEL="$2"; shift 2;;
    --n-boot) N_BOOT="$2"; shift 2;;
    -t|--temperature) TEMPERATURE="$2"; shift 2;;
    --top-p) TOP_P="$2"; shift 2;;
    -i|--importance-csv) IMPORTANCE_CSV="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

if [[ ! -d "$MATH_EVAL_ROOT" ]]; then
  log "ERROR: Math evaluation root not found: $MATH_EVAL_ROOT"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

log "Single-LLM MATH Evaluation"
log "Model          : $MODEL"
log "Math eval root : $MATH_EVAL_ROOT"
log "Output         : $OUTPUT_DIR"
log "Max parallel   : $MAX_PARALLEL"
log "Bootstrap reps : $N_BOOT"
log "Temp / top-p   : $TEMPERATURE / $TOP_P"
[[ -n "$IMPORTANCE_CSV" ]] && log "Pre (7-role) comparison CSV: $IMPORTANCE_CSV"

# Make temperature/top_p visible to Python
export TEMPERATURE TOP_P

# ------------------------------------------------------------
# Write the Python evaluator
# ------------------------------------------------------------
cat > "$OUTPUT_DIR/single_llm_math_eval.py" << 'PY'
#!/usr/bin/env python3
import os, sys, json, math, time
from pathlib import Path
from typing import Dict, List
import concurrent.futures

import numpy as np
import pandas as pd

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Import MATH helpers (get_math_qa_pairs, is_equiv) from code/MMLU/utils.py
SCRIPT_DIR = Path(__file__).resolve().parent          # e.g. single_llm_math_results
MMLU_DIR   = SCRIPT_DIR.parent                        # code/MMLU
if str(MMLU_DIR) not in sys.path:
    sys.path.append(str(MMLU_DIR))

from utils import get_math_qa_pairs, is_equiv

# OpenAI / Together-compatible client
import openai
openai.api_base = os.getenv(
    "TOGETHER_BASE_URL",
    os.getenv("OPENAI_BASE_URL", "https://api.together.xyz/v1")
)
openai.api_key  = os.getenv("TOGETHER_API_KEY", os.getenv("OPENAI_API_KEY"))

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
TOP_P       = float(os.getenv("TOP_P", "1.0"))

# ---------------------------
# Math subjects and meta categories
# ---------------------------
MATH_SUBJECTS = [
    "algebra_test",
    "counting_and_probability_test",
    "geometry_test",
    "intermediate_algebra_test",
    "number_theory_test",
    "prealgebra_test",
    "precalculus_test",
]

SUBJECT_META = {
    "algebra_test": "algebra",
    "counting_and_probability_test": "counting_probability",
    "geometry_test": "geometry",
    "intermediate_algebra_test": "intermediate_algebra",
    "number_theory_test": "number_theory",
    "prealgebra_test": "prealgebra",
    "precalculus_test": "precalculus",
}

def meta_for_subject(subject: str) -> str:
    return SUBJECT_META.get(subject, "math")

# ---------------------------
# Optional: pre-selection (7 roles) metrics
# ---------------------------
def subject_from_filename(fname: str) -> str:
    for sub in sorted(MATH_SUBJECTS, key=len, reverse=True):
        if fname.startswith(sub):
            return sub
    return fname

def collect_pre_metrics(importance_csv: str) -> pd.DataFrame:
    df = pd.read_csv(importance_csv)
    if "subject" in df.columns:
        df["subject"] = df["subject"].astype(str)
    else:
        df["subject"] = df["filename"].astype(str).apply(subject_from_filename)

    rows = []
    has_tin  = "prompt_tokens" in df.columns
    has_tout = "completion_tokens" in df.columns

    for _, r in df.iterrows():
        subj = str(r["subject"])
        meta = meta_for_subject(subj)
        q_cnt = int(r["q_cnt"])
        acc = float(r["acc"])
        correct_float = acc * q_cnt
        responses = int(r["resp"])
        row = {
            "subject": subj,
            "meta": meta,
            "questions": q_cnt,
            "correct": correct_float,
            "responses": responses,
        }
        if has_tin:
            row["prompt_tokens"] = int(r["prompt_tokens"])
        if has_tout:
            row["completion_tokens"] = int(r["completion_tokens"])
        rows.append(row)

    d = pd.DataFrame(rows)
    if "prompt_tokens" not in d.columns:
        d["prompt_tokens"] = np.nan
    if "completion_tokens" not in d.columns:
        d["completion_tokens"] = np.nan
    return d[["subject","meta","questions","correct","responses","prompt_tokens","completion_tokens"]]

# ---------------------------
# Single-LLM math call helpers
# ---------------------------
def extract_final_answer(text: str) -> str:
    """
    Extract the final answer string from the model output:
    - Take the first non-empty line
    - Strip common prefixes like 'Answer:' or 'Final answer:'
    """
    if not text:
        return ""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        prefixes = [
            "Answer:", "answer:",
            "Final answer:", "final answer:",
        ]
        for prefix in prefixes:
            if s.lower().startswith(prefix.lower()):
                s = s[len(prefix):].strip()
                break
        return s
    return ""

def chat_single_math(model: str, question: str, max_tokens: int = 256):
    sys_msg = (
        "You are a strong competition mathematician. "
        "Solve the following math problem. "
        "On the FIRST line, output ONLY the final answer (no words, just the expression or number). "
        "Then, on subsequent lines, briefly explain your reasoning."
    )
    user_msg = f"Problem:\n{question}"
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_tokens,
    )
    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    ptok = int(usage.get("prompt_tokens", 0))
    ctok = int(usage.get("completion_tokens", 0))
    ans = extract_final_answer(content)
    return ans, ptok, ctok

def evaluate_subject(subject: str, root_dir: str, model: str,
                     retries: int = 3, backoff: float = 1.0) -> Dict:
    """
    Evaluate one math subject (e.g., algebra_test) with a single LLM call per problem.
    - Reads all problems under <root_dir>/<subject>/*.json via get_math_qa_pairs
    - For each (question, answer), calls chat_single_math once
    - Uses is_equiv(gt, pred) for correctness
    """
    subdir = os.path.join(root_dir, subject)
    if not os.path.isdir(subdir):
        raise FileNotFoundError(f"Subject dir not found: {subdir}")

    json_files = [f for f in os.listdir(subdir) if f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No json files in {subdir}")

    basenames = sorted(os.path.splitext(f)[0] for f in json_files)
    min_int = int(basenames[0])
    max_int = int(basenames[-1])

    qa_pairs = get_math_qa_pairs(subdir, min_int, max_int)

    qn = 0
    correct = 0
    calls = 0
    ptok = 0
    ctok = 0

    last_err = None

    for que, ans in qa_pairs:
        qn += 1
        question = str(que)

        for attempt in range(retries):
            try:
                pred, p, c = chat_single_math(model, question)
                calls += 1
                ptok += p
                ctok += c
                try:
                    if is_equiv(ans, pred):
                        correct += 1
                except Exception:
                    # If equivalence check fails, treat as incorrect
                    pass
                break
            except Exception as e:
                last_err = e
                time.sleep(backoff * (2 ** attempt))
        else:
            sys.stderr.write(f"[WARN] Skipped one problem in {subject} after retries: {last_err}\n")

    meta = meta_for_subject(subject)
    return {
        "subject": subject,
        "meta": meta,
        "questions": qn,
        "correct": correct,
        "responses": calls,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
    }

# ---------------------------
# Bootstrap & summaries
# ---------------------------
def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0):
    if df.empty:
        return {}
    rng = np.random.default_rng(seed)
    n = len(df)

    def agg(d: pd.DataFrame):
        q = d["questions"].sum()
        acc = float(d["correct"].sum()) / q if q > 0 else float("nan")
        api = d["responses"].sum()
        tin = d["prompt_tokens"].sum(skipna=True)
        tout = d["completion_tokens"].sum(skipna=True)
        return acc, api, tin, tout

    point = agg(df)
    acc_s, api_s, tin_s, tout_s = [], [], [], []
    have_tin  = df["prompt_tokens"].notna().any()
    have_tout = df["completion_tokens"].notna().any()

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = df.iloc[idx]
        a, b, c, d = agg(s)
        acc_s.append(a)
        api_s.append(b)
        tin_s.append(c if have_tin else np.nan)
        tout_s.append(d if have_tout else np.nan)

    def pct_ci(arr):
        arr = np.array(arr, dtype=float)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        return [float(lo), float(hi)]

    out = {
        "accuracy": {"point": float(point[0]), "ci95": pct_ci(acc_s)},
        "api_calls": {
            "point": int(point[1]),
            "ci95": [
                int(np.nanpercentile(api_s, 2.5)),
                int(np.nanpercentile(api_s, 97.5)),
            ],
        },
    }
    out["tokens_in"] = (
        {"point": float(point[2]), "ci95": pct_ci(tin_s)}
        if have_tin else {"point": float("nan"), "ci95": [float("nan"), float("nan")]}
    )
    out["tokens_out"] = (
        {"point": float(point[3]), "ci95": pct_ci(tout_s)}
        if have_tout else {"point": float("nan"), "ci95": [float("nan"), float("nan")]}
    )
    return out

def print_block(title: str, res: dict):
    print(f"\n{title}")
    print("-" * len(title))
    if not res:
        print("No data.")
        return
    a = res["accuracy"]
    k = res["api_calls"]
    print(f"Accuracy   : {a['point']:.4f}  [95% CI {a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]")
    print(f"API calls  : {k['point']}  [95% CI {k['ci95'][0]}, {k['ci95'][1]}]")
    ti = res["tokens_in"]
    to = res["tokens_out"]
    if not math.isnan(ti["point"]):
        print(f"Tokens in  : {ti['point']:.0f}  [95% CI {ti['ci95'][0]:.0f}, {ti['ci95'][1]:.0f}]")
    else:
        print("Tokens in  : N/A")
    if not math.isnan(to["point"]):
        print(f"Tokens out : {to['point']:.0f}  [95% CI {to['ci95'][0]:.0f}, {to['ci95'][1]:.0f}]")
    else:
        print("Tokens out : N/A")

# ---------------------------
# main
# ---------------------------
def main():
    if len(sys.argv) not in (7, 8):
        print("Usage: single_llm_math_eval.py <math_eval_root> <model> <output_dir> <max_parallel> <n_boot> <rounds_dummy> [importance_csv]")
        print("Note: rounds_dummy is unused (for interface symmetry). Pass 0.")
        sys.exit(1)

    math_root      = sys.argv[1]
    model          = sys.argv[2]
    out_dir        = sys.argv[3]
    max_parallel   = int(sys.argv[4])
    n_boot         = int(sys.argv[5])
    _rounds_dummy  = int(sys.argv[6])   # kept for interface symmetry with two-agent script
    importance_csv = sys.argv[7] if len(sys.argv) == 8 else ""

    subjects = [s for s in MATH_SUBJECTS if os.path.isdir(os.path.join(math_root, s))]
    subjects.sort()

    print(f"Found {len(subjects)} math subjects: {subjects}")
    print(f"Model={model}, temp={TEMPERATURE}, top_p={TOP_P}")
    print("Schedule per problem: single model call; final answer = first line of output.")

    results = []

    def run_one(subj: str):
        try:
            return evaluate_subject(subj, math_root, model)
        except Exception as e:
            sys.stderr.write(f"✗ {subj}: ERROR {e}\n")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
        fut = {ex.submit(run_one, s): s for s in subjects}
        for f in concurrent.futures.as_completed(fut):
            r = f.result()
            if r:
                results.append(r)

    df = pd.DataFrame(results)
    os.makedirs(out_dir, exist_ok=True)

    by_subj_path = os.path.join(out_dir, "single_llm_math_by_subject.json")
    with open(by_subj_path, "w") as f:
        json.dump({r["subject"]: r for r in results}, f, indent=2)

    print("\n" + "=" * 60)
    print("SINGLE-LLM — MATH RESULTS")
    print("=" * 60)

    if df.empty:
        print("\nNo data.")
        sys.exit(2)

    single_overall = bootstrap_ci(df, n_boot=n_boot, seed=0)
    print_block("OVERALL (Single-LLM baseline on MATH)", single_overall)

    summary = {
        "notes": {
            "mode": "single_llm_math",
            "n_boot": n_boot,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
        "overall": {
            "single_llm": single_overall,
        },
        "by_meta": {},
    }

    metas = sorted(df["meta"].unique())
    for meta in metas:
        sub = df[df["meta"] == meta]
        res = bootstrap_ci(sub, n_boot=n_boot, seed=hash("math_"+meta) % (2**32))
        print_block(f"{meta} — Single-LLM", res)
        summary["by_meta"][meta] = {"single_llm": res}

    # Optional: compare against pre-selection (7 roles)
    if importance_csv and os.path.exists(importance_csv):
        pre_df = collect_pre_metrics(importance_csv)
        pre_overall = bootstrap_ci(pre_df, n_boot=n_boot, seed=1)
        print_block("OVERALL (Pre-selection / 7 roles)", pre_overall)
        summary["overall"]["pre_7roles"] = pre_overall

        pre_metas = sorted(pre_df["meta"].unique())
        for meta in pre_metas:
            pre_cat = pre_df[pre_df["meta"] == meta]
            res = bootstrap_ci(pre_cat, n_boot=n_boot, seed=hash("pre_"+meta) % (2**32))
            print_block(f"{meta} — Pre-selection (7 roles)", res)
            if meta not in summary["by_meta"]:
                summary["by_meta"][meta] = {}
            summary["by_meta"][meta]["pre_7roles"] = res

    out_json = os.path.join(out_dir, "metrics_summary_single_llm_math.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed JSON saved to: {out_json}")
    print(f"Per-subject JSON saved to: {by_subj_path}")

if __name__ == "__main__":
    main()
PY

chmod +x "$OUTPUT_DIR/single_llm_math_eval.py"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
# The 6th argument is a dummy "rounds" to keep interface similar to two-agent script (pass 0).
python "$OUTPUT_DIR/single_llm_math_eval.py" \
  "$MATH_EVAL_ROOT" \
  "$MODEL" \
  "$OUTPUT_DIR" \
  "$MAX_PARALLEL" \
  "$N_BOOT" \
  0 \
  ${IMPORTANCE_CSV:+$IMPORTANCE_CSV}

log "Single-LLM MATH evaluation completed!"
log "Results saved in: $OUTPUT_DIR"
