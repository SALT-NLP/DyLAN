import ast
import json
import os
import openai
import random
import sys
from LLMLP import LLMLP
from utils import *

# openai.api_key =
# openai.api_base =
# openai.api_type =
# openai.api_version =

SUB_DIR = sys.argv[1]
MIN_FILENAME = int(sys.argv[2])
MAX_FILENAME = int(sys.argv[3])
EXP_NAME = sys.argv[4]
EXP_NAME = EXP_NAME + '_' + str(MIN_FILENAME) + '_' + str(MAX_FILENAME)
MODEL = sys.argv[5]

ACTIVATION = "listwise"
TYPE = "math_exp"
# ROLES = ["Assistant", "Mathematician", "Mathematician", "Assistant"]
DIR_NAME = sys.argv[6]
ROLES = ast.literal_eval(sys.argv[7])
DIR_NAME = DIR_NAME + '_' + '_'.join(ROLES)


def set_rd_seed(seed):
    random.seed(seed)

def main():
    set_rd_seed(0)
    assert len(ROLES) > 0
    os.makedirs(DIR_NAME, exist_ok=True)

    llmlp = LLMLP(MODEL, len(ROLES), ROLES, 3, ACTIVATION, TYPE, MODEL)
    qa_pairs = get_math_qa_pairs(SUB_DIR, MIN_FILENAME, MAX_FILENAME)

    with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+'3.json', 'w') as f:
        f.write("")

    accs, resp_cnts, importances = [], 0, []
    completion_list = []
    total_prompt_tokens, total_completion_tokens = 0, 0

    for que, ans in qa_pairs:
        llmlp.zero_grad()
        res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(que)
        imp_score = llmlp.backward(res)

        completion_list.append(completions)
        accs.append(is_equiv(ans, res))
        resp_cnts += resp_cnt
        importances.append(imp_score)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+'3.json', 'a') as f:
            f.write(json.dumps(completions) + '\n')

    print(accs)
    print(resp_cnts)
    print(importances)

    with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+'3.txt', 'w') as f:
        f.write(str(accs) + ' ' + str(sum(accs)/len(qa_pairs)) + '\n')
        f.write(str(resp_cnts) + " " + str(resp_cnts/len(qa_pairs)) + '\n')
        f.write(json.dumps(importances) + '\n')
        f.write(json.dumps([sum(pos)/len(qa_pairs) for pos in zip(*importances)]) + '\n')
        f.write(str(total_prompt_tokens) + ' ' + str(total_completion_tokens) + '\n')

if __name__ == "__main__":
    main()
