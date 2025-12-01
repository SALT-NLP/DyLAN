#!/usr/bin/env bash
set -euo pipefail

# Summarize an EXISTING MMLU run (no re-evaluation).
# Reports accuracy, API calls, tokens-in, tokens-out with 95% bootstrap CIs
# overall and by meta-category, for pre- (7 roles) and post-selection (reduced roles).
#
# Example:
#   ./evaluate_existing_mmlu_run.sh \
#     --run-dir code/MMLU/runs/baseline_20251101-1402 \
#     --n-boot 1000
#
# Expected contents of RUN_DIR (names are flexible; defaults are guessed):
#   RUN_DIR/importance_1to7*.csv                 # pre-selection summary per test
#   RUN_DIR/evaluation_results*                  # post-selection eval outputs (folders with *_eval.txt)
#
# You can override auto-detected paths with --importance-csv and --eval-dir.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_DIR="${RUN_DIR:-}"
IMPORTANCE_CSV="${IMPORTANCE_CSV:-}"
EVAL_DIR="${EVAL_DIR:-}"
N_BOOT="${N_BOOT:-1000}"

usage() {
  cat <<EOF
Usage: $0 --run-dir RUN_DIR [--importance-csv FILE] [--eval-dir DIR] [--n-boot NUM]

Summarize an existing MMLU run (pre vs post, overall + meta-categories)
with 95% bootstrap CIs for accuracy, API calls, tokens-in, tokens-out.

Options:
  --run-dir DIR           Root directory of the run (required)
  --importance-csv FILE   Path to importance_1to7*.csv (default: auto-detect in RUN_DIR)
  --eval-dir DIR          Directory with *_eval.txt files (default: auto-detect in RUN_DIR)
  --n-boot NUM            Bootstrap replicates (default: 1000)

Examples:
  $0 --run-dir code/MMLU/runs/baseline_20251101-1402
  $0 --run-dir ... --n-boot 2000
EOF
}

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

# --------------------
# Parse arguments
# --------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2;;
    --importance-csv) IMPORTANCE_CSV="$2"; shift 2;;
    --eval-dir) EVAL_DIR="$2"; shift 2;;
    --n-boot) N_BOOT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: --run-dir is required."
  usage
  exit 1
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: RUN_DIR not found: ${RUN_DIR}"
  exit 1
fi

# Try to auto-detect inputs if not provided
if [[ -z "${IMPORTANCE_CSV}" ]]; then
  found_csv="$(ls -1 "${RUN_DIR}"/importance_1to7*.csv 2>/dev/null | head -1 || true)"
  if [[ -n "${found_csv}" ]]; then
    IMPORTANCE_CSV="${found_csv}"
  fi
fi
if [[ -z "${EVAL_DIR}" ]]; then
  # prefer a directory named like evaluation_results*
  found_eval="$(ls -1d "${RUN_DIR}"/evaluation_results* 2>/dev/null | head -1 || true)"
  if [[ -n "${found_eval}" ]]; then
    EVAL_DIR="${found_eval}"
  fi
fi

if [[ -z "${IMPORTANCE_CSV}" || ! -f "${IMPORTANCE_CSV}" ]]; then
  echo "ERROR: Could not find importance_1to7 CSV. Use --importance-csv to specify."
  exit 1
fi
if [[ -z "${EVAL_DIR}" || ! -d "${EVAL_DIR}" ]]; then
  echo "ERROR: Could not find evaluation results directory. Use --eval-dir to specify."
  exit 1
fi

log "Summarizing existing run"
log "RUN_DIR        : ${RUN_DIR}"
log "IMPORTANCE_CSV : ${IMPORTANCE_CSV}"
log "EVAL_DIR       : ${EVAL_DIR}"
log "N_BOOT         : ${N_BOOT}"

# --------------------
# Python driver
# --------------------
python - "$IMPORTANCE_CSV" "$EVAL_DIR" "$N_BOOT" "$RUN_DIR" << 'PYCODE'
import os, sys, re, json, ast, math
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

importance_csv = sys.argv[1]
eval_dir       = sys.argv[2]
n_boot         = int(sys.argv[3])
run_dir        = sys.argv[4]

# ------------- Category mapping -------------
SUBCATEGORY = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

def subject_key_from_name(name: str) -> str:
    base = Path(name).stem
    base = re.sub(r'_eval$', '', base)
    base = re.sub(r'_(test|val)(_\d+)?$', '', base)
    return base

def meta_for_subject(subject: str) -> str:
    subs = SUBCATEGORY.get(subject, ["other"])
    sub = subs[0] if subs else "other"
    for meta, sublist in CATEGORIES.items():
        if sub in sublist:
            return meta
    return "other (business, health, misc.)"

# ------------- Parse post-selection *_eval.txt -------------
def parse_eval_txt(fp: str):
    # Expected:
    # 0: [True, False, ...] 0.5
    # 1: <total_responses> <avg_responses>
    # 2: [[...]] (ignored)
    # 3: [...]  (ignored)
    # 4: <prompt_tokens>
    # 5: <completion_tokens>
    with open(fp, 'r') as f:
        lines = f.readlines()
    accs_parts = lines[0].strip().rsplit(' ', 1)
    accs = ast.literal_eval(accs_parts[0])
    resp_parts = lines[1].strip().split(' ', 1)
    total_resp = int(resp_parts[0])
    prompt_tokens = int(lines[4].strip())
    completion_tokens = int(lines[5].strip())
    q = len(accs)
    c = sum(1 for a in accs if a)
    return q, c, total_resp, prompt_tokens, completion_tokens

post_rows = []
for root, _, files in os.walk(eval_dir):
    for fn in files:
        if fn.endswith("_eval.txt"):
            fp = os.path.join(root, fn)
            test_name = Path(fn).stem.replace('_eval','')
            try:
                q, c, responses, tin, tout = parse_eval_txt(fp)
            except Exception:
                # Skip malformed outputs
                continue
            subj = subject_key_from_name(test_name)
            meta = meta_for_subject(subj)
            post_rows.append({
                "test_name": test_name,
                "subject": subj,
                "meta": meta,
                "questions": q,
                "correct": c,
                "responses": responses,
                "prompt_tokens": float(tin),
                "completion_tokens": float(tout),
            })

post_df = pd.DataFrame(post_rows)
if post_df.empty:
    print("ERROR: No *_eval.txt files found under:", eval_dir)
    sys.exit(1)

# ------------- Load pre-selection importance_1to7 -------------
df_imp = pd.read_csv(importance_csv)
pre_rows = []
has_tin = 'prompt_tokens' in df_imp.columns
has_tout = 'completion_tokens' in df_imp.columns
for _, r in df_imp.iterrows():
    fname = str(r['filename'])
    subj  = subject_key_from_name(fname)
    meta  = meta_for_subject(subj)
    q_cnt = int(r['q_cnt'])
    acc   = float(r['acc'])
    # keep fractional "correct" for aggregation
    correct = acc * q_cnt
    pre_rows.append({
        "test_name": Path(fname).stem,
        "subject"  : subj,
        "meta"     : meta,
        "questions": q_cnt,
        "correct"  : correct,
        "responses": int(r['resp']),
        "prompt_tokens": float(r['prompt_tokens']) if has_tin and not pd.isna(r['prompt_tokens']) else np.nan,
        "completion_tokens": float(r['completion_tokens']) if has_tout and not pd.isna(r['completion_tokens']) else np.nan,
    })
pre_df = pd.DataFrame(pre_rows)

# ------------- If pre tokens missing, estimate per-test from post using response ratio -------------
total_resp_pre  = pre_df['responses'].sum()
total_resp_post = post_df['responses'].sum()
R = (float(total_resp_pre) / float(total_resp_post)) if total_resp_post > 0 else np.nan

if not (pre_df['prompt_tokens'].notna().any() and pre_df['completion_tokens'].notna().any()):
    # Join post tokens by subject (1:1 in MMLU); estimate pre tokens per test as R × post tokens per test
    post_tok_by_subj = post_df[['subject','prompt_tokens','completion_tokens']]
    pre_df = pre_df.merge(post_tok_by_subj, on='subject', how='left', suffixes=('', '_post'))
    pre_df['prompt_tokens'] = pre_df['prompt_tokens'].where(pre_df['prompt_tokens'].notna(),
                                                            pre_df['prompt_tokens_post'] * R)
    pre_df['completion_tokens'] = pre_df['completion_tokens'].where(pre_df['completion_tokens'].notna(),
                                                                    pre_df['completion_tokens_post'] * R)
    pre_df.drop(columns=[c for c in pre_df.columns if c.endswith('_post')], inplace=True)
    pre_tokens_estimated = True
else:
    pre_tokens_estimated = False

# ------------- Bootstrap -------------
def bootstrap_ci(df: pd.DataFrame, n_boot=1000, seed=0):
    if df.empty:
        return {}
    rng = np.random.default_rng(seed)
    n = len(df)
    # point estimates
    q_sum = df['questions'].sum()
    acc_point = float(df['correct'].sum()) / q_sum if q_sum > 0 else float('nan')
    api_point = float(df['responses'].sum())
    tin_point = float(df['prompt_tokens'].sum(skipna=True))
    tout_point= float(df['completion_tokens'].sum(skipna=True))

    acc_s, api_s, tin_s, tout_s = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = df.iloc[idx]
        q   = s['questions'].sum()
        acc = float(s['correct'].sum()) / q if q > 0 else float('nan')
        api = float(s['responses'].sum())
        tin = float(s['prompt_tokens'].sum(skipna=True))
        tout= float(s['completion_tokens'].sum(skipna=True))
        acc_s.append(acc); api_s.append(api); tin_s.append(tin); tout_s.append(tout)

    def pct_ci(vals):
        a = np.array(vals, dtype=float)
        lo, hi = np.nanpercentile(a, [2.5, 97.5])
        return [float(lo), float(hi)]

    out = {
        "accuracy":   {"point": acc_point, "ci95": pct_ci(acc_s)},
        "api_calls":  {"point": api_point, "ci95": [float(np.nanpercentile(api_s,2.5)), float(np.nanpercentile(api_s,97.5))]},
        "tokens_in":  {"point": tin_point, "ci95": pct_ci(tin_s)},
        "tokens_out": {"point": tout_point, "ci95": pct_ci(tout_s)},
    }
    return out

def print_block(title: str, res: dict, mark_est=False):
    print(f"\n{title}")
    print("-" * len(title))
    acc = res['accuracy']; api = res['api_calls']; tin = res['tokens_in']; tout = res['tokens_out']
    print(f"Accuracy     : {acc['point']:.4f}  [95% CI {acc['ci95'][0]:.4f}, {acc['ci95'][1]:.4f}]")
    print(f"API calls    : {int(api['point'])}  [95% CI {int(api['ci95'][0])}, {int(api['ci95'][1])}]")
    tag = " (est.)" if mark_est else ""
    print(f"Tokens in    : {tin['point']:.0f}{tag}  [95% CI {tin['ci95'][0]:.0f}, {tin['ci95'][1]:.0f}]")
    print(f"Tokens out   : {tout['point']:.0f}{tag}  [95% CI {tout['ci95'][0]:.0f}, {tout['ci95'][1]:.0f}]")

# Overall
post_overall = bootstrap_ci(post_df, n_boot=n_boot, seed=0)
pre_overall  = bootstrap_ci(pre_df,  n_boot=n_boot, seed=1)

print("="*60)
print("EXISTING RUN SUMMARY (pre vs post)")
print("="*60)
print_block("OVERALL — Post-selection (reduced roles)", post_overall, mark_est=False)
print_block("OVERALL — Pre-selection (7 roles)",       pre_overall,  mark_est=pre_tokens_estimated)

# By meta-category
by_meta = {}
for meta in CATEGORIES.keys():
    post_m = bootstrap_ci(post_df[post_df['meta']==meta], n_boot=n_boot, seed=(hash("post"+meta)%(2**32)))
    pre_m  = bootstrap_ci(pre_df [pre_df ['meta']==meta], n_boot=n_boot, seed=(hash("pre"+meta) %(2**32)))
    print_block(f"{meta} — Post-selection", post_m, mark_est=False)
    print_block(f"{meta} — Pre-selection",  pre_m,  mark_est=pre_tokens_estimated)
    by_meta[meta] = {"post": post_m, "pre": pre_m}

# Notes & JSON
summary = {
    "notes": {
        "api_calls_definition": "Sum of model responses over all questions (proxy for #API calls).",
        "pre_tokens_estimated": pre_tokens_estimated,
    },
    "ratios": {
        "response_ratio_R_pre_over_post": (float(R) if not np.isnan(R) else None),
    },
    "overall": {"post": post_overall, "pre": pre_overall},
    "by_meta": by_meta,
}

out_json = os.path.join(run_dir, "metrics_summary_existing_run.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nDetailed JSON saved to: {out_json}")
PYCODE
