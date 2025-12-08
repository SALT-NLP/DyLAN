#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="math_downsampled"
ROLES="['Mathematician','AlgebraExpert','CountingProbabilitySpecialist','GeometryWizard','IntermediateAlgebraMaestro','NumberTheoryScholar','PrecalculusGuru']"

TOTAL_AGENTS="$(python - <<'PY'
import ast
roles = ast.literal_eval("['Mathematician','AlgebraExpert','CountingProbabilitySpecialist','GeometryWizard','IntermediateAlgebraMaestro','NumberTheoryScholar','PrecalculusGuru']")
print(len(roles))
PY
)"

DIR_NAME="${EXP_NAME}_$(echo "$ROLES" | tr -d "[]' " | tr ',' '_')"
TARGET_CSV="${TARGET_CSV:-importance_math_1to7.csv}"

echo "TOTAL_AGENTS = $TOTAL_AGENTS"
echo "DIR_NAME     = $DIR_NAME"
echo "TARGET_CSV   = $TARGET_CSV"

# 注意这里用的是 *_math.py
python proc_lists_math.py \
  "$TOTAL_AGENTS" \
  "$DIR_NAME" \
  "$TARGET_CSV" \
  "[0]" "[1]" "[2]" "[3]" "[4]" "[5]" "[6]"

python build_csv_math.py \
  "$TOTAL_AGENTS" \
  "$DIR_NAME" \
  "$TARGET_CSV" \
  "[0]" "[1]" "[2]" "[3]" "[4]" "[5]" "[6]" \
  Mathematician AlgebraExpert CountingProbabilitySpecialist GeometryWizard IntermediateAlgebraMaestro NumberTheoryScholar PrecalculusGuru

python calc_ave_acc.py "$TARGET_CSV"
