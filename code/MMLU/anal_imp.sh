#!/usr/bin/env bash
set -euo pipefail

# Inputs must match exp_mmlu.sh
EXP_NAME="mmlu_downsampled"
ROLES="['Economist','Doctor','Lawyer','Mathematician','Psychologist','Programmer','Historian']"

# Derive
TOTAL_AGENTS="$(python - <<'PY'
import ast
roles = ast.literal_eval("['Economist','Doctor','Lawyer','Mathematician','Psychologist','Programmer','Historian']")
print(len(roles))
PY
)"
DIR_NAME="${EXP_NAME}_$(echo "$ROLES" | tr -d "[]' " | tr ',' '_')"
TARGET_CSV="${TARGET_CSV:-importance_1to7.csv}"

# Post-processing
python proc_lists.py $TOTAL_AGENTS "$DIR_NAME" "$TARGET_CSV" "[0]" "[1]" "[2]" "[3]" "[4]" "[5]" "[6]"
python build_csv.py $TOTAL_AGENTS "$DIR_NAME" "$TARGET_CSV" "[0]" "[1]" "[2]" "[3]" "[4]" "[5]" "[6]" Economist Doctor Lawyer Mathematician Psychologist Programmer Historian
python calc_ave_acc.py "$TARGET_CSV"
