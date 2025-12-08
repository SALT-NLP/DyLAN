#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Two-Agent Debate MATH Evaluation + CI
# ------------------------------------------------------------
# - Data: data/math_json/<split>/<subject>/*.json
# - Subjects: 7 standard MATH subjects
# - For each question:
#     Proposer_0 → (Critic_i, Proposer_i) * ROUNDS
#   Final answer = last Proposer answer.
# - Metrics: accuracy, API calls, tokens-in/out + 95% bootstrap CI
# - Optional: compare against pre-selection (7-role) via importance_math_1to7.csv
# ------------------------------------------------------------

MODEL="${MODEL:-openai/gpt-oss-20b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Root directory for math evaluation data (each subject is a subdirectory here)
MATH_EVAL_ROOT="${MATH_EVAL_ROOT:-$SCRIPT_DIR/../../data/math_json/evaluation}"

OUTPUT_DIR="${OUTPUT_DIR:-two_agent_math_results}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
N_BOOT="${N_BOOT:-1000}"

# Debate hyperparameters
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
ROUNDS="${ROUNDS:-3}"                # Number of (Critic→Proposer) pairs after the initial Proposer

# Optional: pre-selection comparison (7-role run)
IMPORTANCE_CSV="${IMPORTANCE_CSV:-}" # e.g. importance_math_1to7.csv

usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Run a 2-agent (Proposer–Critic) debate baseline on the MATH dataset and report:
  - accuracy
  - API calls
  - tokens-in / tokens-out
with 95% bootstrap confidence intervals (overall and per math subcategory).

OPTIONS:
  -m, --model MODEL              Chat model id (default: ${MODEL})
  -d, --dataset DIR              Path to math eval root (default: ${MATH_EVAL_ROOT})
  -o, --output DIR               Output directory (default: ${OUTPUT_DIR})
  -p, --max-parallel NUM         Max parallel subjects (default: ${MAX_PARALLEL})
  --n-boot NUM                   Bootstrap replicates for CI (default: ${N_BOOT})
  -t, --temperature FLOAT        Temperature (default: ${TEMPERATURE})
  --top-p FLOAT                  top_p (default: ${TOP_P})
  -r, --rounds NUM               Number of (Critic→Proposer) pairs (default: ${ROUNDS})
  -i, --importance-csv FILE      (optional) importance_math_1to7.csv for pre-selection comparison
  -h, --help                     Show this help

Examples:
  $0
  $0 --rounds 2 --model "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
  N_BOOT=2000 $0 --dataset "../../data/math_json/medium_team_selection" --max-parallel 8
  IMPORTANCE_CSV=importance_math_1to7.csv $0
EOF
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

# -------- argument parsing ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="$2"; shift 2;;
    -d|--dataset) MATH_EVAL_ROOT="$2"; shift 2;;
    -o|--output) OUTPUT_DIR="$2"; shift 2;;
    -p|--max-parallel) MAX_PARALLEL="$2"; shift 2;;
    --n-boot) N_BOOT="$2"; shift 2;;
    -t|--temperature) TEMPERATURE="$2"; shift 2;;
    --top-p) TOP_P="$2"; shift 2;;
    -r|--rounds) ROUNDS="$2"; shift 2;;
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

log "Two-Agent Debate MATH Evaluation"
log "Model          : $MODEL"
log "Math eval root : $MATH_EVAL_ROOT"
log "Output         : $OUTPUT_DIR"
log "Max parallel   : $MAX_PARALLEL"
log "Bootstrap reps : $N_BOOT"
log "Temp / top-p   : $TEMPERATURE / $TOP_P"
log "Rounds (C→P)   : $ROUNDS  |  Calls per question (no early stop): $((1 + 2 * ROUNDS))"
[[ -n "$IMPORTANCE_CSV" ]] && log "Pre-selection CSV: $IMPORTANCE_CSV"

# Propagate temperature / top_p into Python
export TEMPERATURE TOP_P

# ------------------------------------------------------------
# Write Python evaluator: two_agent_debate_math_eval.py
# ------------------------------------------------------------
cat > "$OUTPUT_DIR/two_agent_debate_math_eval.py" << 'PY'
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

# Import math utilities (get_math_qa_pairs, is_equiv) from code/MMLU/utils.py
SCRIPT_DIR = Path(__file__).resolve().parent          # e.g. two_agent_math_results
MMLU_DIR   = SCRIPT_DIR.parent                        # code/MMLU
if str(MMLU_DIR) not in sys.path:
    sys.path.append(str(MMLU_DIR))

from utils import get_math_qa_pairs, is_equiv

# OpenAI / Together-style client
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
# Pre-selection metrics (7 roles) — optional
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
# Debate helpers (open-ended math)
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

def chat(model: str, messages: List[Dict], max_tokens: int = 256):
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_tokens,
    )
    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    ptok = int(usage.get("prompt_tokens", 0))
    ctok = int(usage.get("completion_tokens", 0))
    return content, ptok, ctok

def proposer_initial(model: str, question: str):
    sys_msg = (
        "You are a strong competition mathematician. "
        "Solve the following math problem. "
        "On the FIRST line, output ONLY the final answer (no words, just the expression or number). "
        "Then, on subsequent lines, you may briefly explain your reasoning."
    )
    user_msg = f"Problem:\n{question}"
    content, p, c = chat(model, [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": user_msg},
    ], max_tokens=256)
    ans = extract_final_answer(content)
    return ans, content, p, c

def critic_feedback(model: str, question: str, proposer_content: str):
    sys_msg = (
        "You are the Critic. "
        "Check the Proposer's solution carefully. "
        "If you think the final answer is wrong, provide a corrected final answer. "
        "On the FIRST line, output ONLY your final answer (expression or number). "
        "Then briefly justify your judgement."
    )
    user_msg = f"""Problem:
{question}

Proposer's solution:
{proposer_content}"""
    content, p, c = chat(model, [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": user_msg},
    ], max_tokens=256)
    ans = extract_final_answer(content)
    return ans, content, p, c

def proposer_revise(model: str, question: str, critic_content: str, prev_proposer_content: str):
    sys_msg = (
        "You are the Proposer revising your answer after receiving a critique. "
        "Consider the Critic's arguments and decide the best final answer. "
        "On the FIRST line, output ONLY the final answer (expression or number). "
        "Then briefly justify."
    )
    user_msg = f"""Problem:
{question}

Your previous solution:
{prev_proposer_content}

Critic's feedback:
{critic_content}"""
    content, p, c = chat(model, [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": user_msg},
    ], max_tokens=256)
    ans = extract_final_answer(content)
    return ans, content, p, c

def evaluate_subject(subject: str, root_dir: str, model: str, rounds: int,
                     retries: int = 3, backoff: float = 1.0) -> Dict:
    """
    Evaluate one math subject (e.g., algebra_test) with two-agent debate:
      - Read all problems under <root_dir>/<subject>/*.json
      - Per question: P0 → (Critic, Proposer)*rounds
      - Final answer = last Proposer's answer
      - Correctness checked via is_equiv(gt, pred)
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

        # Initial Proposer
        last_p_answer = ""
        last_p_content = ""
        init_p_answer = ""
        init_p_content = ""

        for attempt in range(retries):
            try:
                lp, pc, p_p, p_c = proposer_initial(model, question)
                calls += 1
                ptok += p_p
                ctok += p_c
                init_p_answer, init_p_content = lp, pc
                last_p_answer, last_p_content = lp, pc
                break
            except Exception as e:
                last_err = e
                time.sleep(backoff * (2 ** attempt))
        else:
            # Proposer repeatedly failed; skip this question
            continue

        # R rounds of (Critic → Proposer)
        for _ in range(rounds):
            # Critic step
            for attempt in range(retries):
                try:
                    cl, cc, c_p, c_c = critic_feedback(model, question, last_p_content)
                    calls += 1
                    ptok += c_p
                    ctok += c_c
                    critic_ans, critic_content = cl, cc
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(backoff * (2 ** attempt))
            else:
                # Critic repeatedly failed; stop further interaction on this question
                break

            # Proposer revision
            for attempt in range(retries):
                try:
                    lp, pc, p_p, p_c = proposer_revise(model, question, critic_content, last_p_content)
                    calls += 1
                    ptok += p_p
                    ctok += p_c
                    last_p_answer, last_p_content = lp, pc
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(backoff * (2 ** attempt))
            else:
                # Proposer revision repeatedly failed; stop this question
                break

        final_pred = last_p_answer or init_p_answer or ""
        try:
            if is_equiv(ans, final_pred):
                correct += 1
        except Exception:
            # If is_equiv fails, treat as incorrect
            pass

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
# Bootstrap + summaries
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
        print("Usage: two_agent_debate_math_eval.py <math_eval_root> <model> <output_dir> <max_parallel> <n_boot> <rounds> [importance_csv]")
        sys.exit(1)

    math_root      = sys.argv[1]
    model          = sys.argv[2]
    out_dir        = sys.argv[3]
    max_parallel   = int(sys.argv[4])
    n_boot         = int(sys.argv[5])
    rounds         = int(sys.argv[6])
    importance_csv = sys.argv[7] if len(sys.argv) == 8 else ""

    subjects = [s for s in MATH_SUBJECTS if os.path.isdir(os.path.join(math_root, s))]
    subjects.sort()

    print(f"Found {len(subjects)} math subjects: {subjects}")
    print(f"Model={model}, temp={TEMPERATURE}, top_p={TOP_P}, rounds={rounds}")
    print("Schedule per question: P0 → (Critic, Proposer)*rounds; final = last Proposer answer.")

    results = []

    def run_one(subj: str):
        try:
            return evaluate_subject(subj, math_root, model, rounds)
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

    by_subj_path = os.path.join(out_dir, "two_agent_math_by_subject.json")
    with open(by_subj_path, "w") as f:
        json.dump({r["subject"]: r for r in results}, f, indent=2)

    print("\n" + "=" * 60)
    print("TWO-AGENT DEBATE — MATH RESULTS")
    print("=" * 60)

    if df.empty:
        print("\nNo data.")
        sys.exit(2)

    # Overall debate metrics
    debate_overall = bootstrap_ci(df, n_boot=n_boot, seed=0)
    print_block("OVERALL (Two-agent debate on MATH)", debate_overall)

    summary = {
        "notes": {
            "mode": "two_agent_debate_math",
            "n_boot": n_boot,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "rounds": rounds,
        },
        "overall": {
            "two_agent_debate": debate_overall,
        },
        "by_meta": {},
    }

    metas = sorted(df["meta"].unique())
    for meta in metas:
        sub = df[df["meta"] == meta]
        res = bootstrap_ci(sub, n_boot=n_boot, seed=hash("math_"+meta) % (2**32))
        print_block(f"{meta} — Two-agent debate", res)
        summary["by_meta"][meta] = {"two_agent_debate": res}

    # Optional: pre-selection (7 roles) comparison
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

    out_json = os.path.join(out_dir, "metrics_summary_two_agent_math.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed JSON saved to: {out_json}")
    print(f"Per-subject JSON saved to: {by_subj_path}")

if __name__ == "__main__":
    main()
PY

chmod +x "$OUTPUT_DIR/two_agent_debate_math_eval.py"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
python "$OUTPUT_DIR/two_agent_debate_math_eval.py" \
  "$MATH_EVAL_ROOT" \
  "$MODEL" \
  "$OUTPUT_DIR" \
  "$MAX_PARALLEL" \
  "$N_BOOT" \
  "$ROUNDS" \
  ${IMPORTANCE_CSV:+$IMPORTANCE_CSV}

log "Two-agent MATH debate evaluation completed!"
log "Results saved in: $OUTPUT_DIR"
