#!/bin/bash

# This code might cause zombies. If you run it, you might have to kill the processes manually.

MODEL=gpt-3.5-turbo
# MODEL=gpt-4


# specify your directory
DIR_NAME=code_human_eval_35_0301_all_tests_t0.8

ROLES="['PythonAssistant', 'AlgorithmDeveloper', 'ComputerScientist', 'Programmer']"
JUDGES="['Passer', 'Tester', 'Reflector', 'Ranker']"

for part in {0..3}
do
    EXP_NAME="llmlp_human_eval_${i}_${part}"
    # run python script in background
    python llmlp_listwise_human_eval.py "$part" "$EXP_NAME" "$MODEL" "$DIR_NAME" "$ROLES" "$JUDGES" &
done

wait
echo "All done"


DIR_NAME=code_human_eval_35_0301_all_tests_t0.8_PythonAssistant_AlgorithmDeveloper_ComputerScientist_Programmer
for i in {0..0}
do

    for part in {0..3}
    do
    EXP_NAME="llmlp_human_eval_${i}_${part}"
    cat ${DIR_NAME}/${EXP_NAME}_443.jsonl >> ${DIR_NAME}/llmlp_human_eval_${i}.jsonl
    done
    evaluate_functional_correctness ${DIR_NAME}/llmlp_human_eval_${i}.jsonl > ${DIR_NAME}/llmlp_human_eval_${i}.txt

done
