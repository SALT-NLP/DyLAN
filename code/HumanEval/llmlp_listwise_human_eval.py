import ast
import json
import os
import openai
import random
import sys
from CoLLMLP import CoLLMLP
from utils import *

# openai.api_key =
# openai.api_base =
# openai.api_type =
# openai.api_version =

PART = int(sys.argv[1])
EXP_NAME = sys.argv[2]
MODEL = sys.argv[3]

ACTIVATION = "listwise"
TYPE = "code_completion"
# ROLES = ["Assistant", "Mathematician", "Mathematician", "Assistant"]
DIR_NAME = sys.argv[4]
ROLES = ast.literal_eval(sys.argv[5])
JUDGES = ast.literal_eval(sys.argv[6])
DIR_NAME = DIR_NAME + '_' + '_'.join(ROLES)

SUBSET = 50

def set_rd_seed(seed):
    random.seed(seed)

def main():
    set_rd_seed(0)
    assert len(ROLES) > 0
    assert len(JUDGES) > 0
    os.makedirs(DIR_NAME, exist_ok=True)

    llmlp = CoLLMLP(MODEL, len(ROLES), ROLES, len(JUDGES), JUDGES, 3, ACTIVATION, TYPE, MODEL)
    qa_pairs = get_human_eval_qa_pairs()

    with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+str(len(JUDGES))+'3.json', 'w') as f:
        f.write("")
    with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+str(len(JUDGES))+'3.tests', 'w') as f:
        f.write("")

    results, resp_cnts, importances = [], 0, []
    completion_list = []
    tests_list = []
    total_prompt_tokens, total_completion_tokens = 0, 0

    for task_id, que, entry_point in qa_pairs:
        qid = int(task_id.split("/")[-1])
        if qid < PART*SUBSET or qid >= (PART+1)*SUBSET:
            continue

        llmlp.zero_grad()
        res, resp_cnt, completions, prompt_tokens, completion_tokens, tests = llmlp.forward(que, entry_point)
        imp_score = llmlp.backward(res, que, entry_point)

        completion_list.append(completions)
        results.append({"task_id": task_id, "completion": res})
        resp_cnts += resp_cnt
        importances.append(imp_score)
        tests_list.append(tests)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+str(len(JUDGES))+'3.json', 'a') as f:
            f.write(json.dumps(completions) + '\n')
        with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+str(len(JUDGES))+'3.tests', 'a') as f:
            f.write(json.dumps(tests) + '\n')

    print(results)
    print(resp_cnts)
    print(importances)
    print(total_prompt_tokens, total_completion_tokens)

    with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+str(len(JUDGES))+'3.txt', 'w') as f:
        f.write(str(resp_cnts) + " " + str(resp_cnts/len(qa_pairs)) + '\n')
        f.write(json.dumps(importances) + '\n')
        f.write(json.dumps([sum(pos)/len(qa_pairs) for pos in zip(*importances)]) + '\n')
        f.write(str(total_prompt_tokens) + " " + str(total_completion_tokens) + '\n')
    
    write_jsonl(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+str(len(JUDGES))+'3.jsonl', results)

if __name__ == "__main__":
    main()
