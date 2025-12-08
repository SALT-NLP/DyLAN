# import ast
# import json
# import os
# import openai
# import random
# import sys
# from LLMLP import LLMLP
# from utils import *

# # openai.api_key =
# # openai.api_base =
# # openai.api_type =
# # openai.api_version =

# SUB_DIR = sys.argv[1]
# MIN_FILENAME = int(sys.argv[2])
# MAX_FILENAME = int(sys.argv[3])
# EXP_NAME = sys.argv[4]
# EXP_NAME = EXP_NAME + '_' + str(MIN_FILENAME) + '_' + str(MAX_FILENAME)
# MODEL = sys.argv[5]

# ACTIVATION = "listwise"
# TYPE = "math_exp"
# # ROLES = ["Assistant", "Mathematician", "Mathematician", "Assistant"]
# DIR_NAME = sys.argv[6]
# ROLES = ast.literal_eval(sys.argv[7])
# DIR_NAME = DIR_NAME + '_' + '_'.join(ROLES)


# def set_rd_seed(seed):
#     random.seed(seed)

# def main():
#     set_rd_seed(0)
#     assert len(ROLES) > 0
#     os.makedirs(DIR_NAME, exist_ok=True)

#     #Print configuration status
#     if os.getenv("RATIONALE", "0") == "1":
#         print("=" * 10)
#         print("RATIONALE MODE ENABLED: Agents will provide rationales for their scores.")
#         print("=" * 10) 

#     llmlp = LLMLP(MODEL, len(ROLES), ROLES, 3, ACTIVATION, TYPE, MODEL)
#     qa_pairs = get_math_qa_pairs(SUB_DIR, MIN_FILENAME, MAX_FILENAME)

#     with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+'3.json', 'w') as f:
#         f.write("")

#     accs, resp_cnts, importances = [], 0, []
#     completion_list = []
#     total_prompt_tokens, total_completion_tokens = 0, 0

#     for que, ans in qa_pairs:
#         llmlp.zero_grad()
#         res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(que)
#         imp_score = llmlp.backward(res)

#         completion_list.append(completions)
#         accs.append(is_equiv(ans, res))
#         resp_cnts += resp_cnt
#         importances.append(imp_score)
#         total_prompt_tokens += prompt_tokens
#         total_completion_tokens += completion_tokens

#         with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+'3.json', 'a') as f:
#             f.write(json.dumps(completions) + '\n')

#     print(accs)
#     print(resp_cnts)
#     print(importances)

#     with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+'3.txt', 'w') as f:
#         f.write(str(accs) + ' ' + str(sum(accs)/len(qa_pairs)) + '\n')
#         f.write(str(resp_cnts) + " " + str(resp_cnts/len(qa_pairs)) + '\n')
#         f.write(json.dumps(importances) + '\n')
#         f.write(json.dumps([sum(pos)/len(qa_pairs) for pos in zip(*importances)]) + '\n')
#         f.write(str(total_prompt_tokens) + ' ' + str(total_completion_tokens) + '\n')

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
import ast
import json
import os
import random
import sys

from dotenv import load_dotenv
from LLMLP import LLMLP

# ---- 根据你的仓库结构，把 MATH 那边的 util 加到 path 里 ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)        # 上一层是 code
MATH_DIR = os.path.join(REPO_ROOT, "MATH")
if MATH_DIR not in sys.path:
    sys.path.append(MATH_DIR)

# 这里假设 get_math_qa_pairs / is_equiv 在 code/MATH/util.py 里
from utils import get_math_qa_pairs, is_equiv

load_dotenv()  # 让 .env 中的 key 生效

# 参数：
# sys.argv[1] = SUB_DIR       题目所在子目录，比如 data/math_json/small_team_selection/algebra_test
# sys.argv[2] = MIN_BASENAME  起始文件名（不含 .json），例如 "0001"
# sys.argv[3] = MAX_BASENAME  结束文件名，例如 "0012"
# sys.argv[4] = MODEL         模型名，比如 openai/gpt-oss-20b
# sys.argv[5] = EXP_NAME_BASE 实验名前缀，比如 "math_downsampled"
# sys.argv[6] = ROLES_STR     角色列表字符串，例如 "['Economist','Doctor',...]"

SUB_DIR = sys.argv[1]
MIN_BASENAME = sys.argv[2]
MAX_BASENAME = sys.argv[3]
MODEL = sys.argv[4]
EXP_NAME_BASE = sys.argv[5]
ROLES = ast.literal_eval(sys.argv[6])

MIN_FILENAME = int(MIN_BASENAME)
MAX_FILENAME = int(MAX_BASENAME)

ACTIVATION = "listwise"
TYPE = "math_exp"  # ✅ math 用 math_exp，会用数学 prompt + is_equiv

# 统一结果目录名：例如 math_downsampled_Economist_Doctor_...
DIR_NAME = EXP_NAME_BASE + '_' + '_'.join(ROLES)

# 单个 batch 的名字：例如 algebra_test_0001_0012
SUBDIR_BASE = os.path.basename(SUB_DIR.rstrip("/"))
EXP_NAME = f"{SUBDIR_BASE}_{MIN_BASENAME}_{MAX_BASENAME}"


def set_rd_seed(seed: int):
    random.seed(seed)


def main():
    set_rd_seed(0)
    assert len(ROLES) > 0
    os.makedirs(DIR_NAME, exist_ok=True)

    # 打印 rationale 配置
    if os.getenv("RATIONALE", "0") == "1":
        print("=" * 10)
        print("RATIONALE MODE ENABLED: Agents will provide rationales for their scores.")
        print("=" * 10)

    # 初始化 multi-agent 网络
    llmlp = LLMLP(MODEL, len(ROLES), ROLES, 3, ACTIVATION, TYPE, MODEL)

    # 读取这个 batch 的题目
    qa_pairs = get_math_qa_pairs(SUB_DIR, MIN_FILENAME, MAX_FILENAME)

    json_path = os.path.join(DIR_NAME, f"{EXP_NAME}_{len(ROLES)}3.json")
    txt_path = os.path.join(DIR_NAME, f"{EXP_NAME}_{len(ROLES)}3.txt")

    # 清空 json 文件
    with open(json_path, "w") as f:
        f.write("")

    accs = []
    resp_cnts = 0
    importances = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for que, ans in qa_pairs:
        llmlp.zero_grad()
        res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(que)

        # ✅ 传 question 进去，让 judge 有上下文
        imp_score = llmlp.backward(res, que)

        # 数学答案用 is_equiv 判断
        accs.append(is_equiv(ans, res))
        resp_cnts += resp_cnt
        importances.append(imp_score)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        with open(json_path, "a") as f:
            f.write(json.dumps(completions) + "\n")

    print(accs)
    print(resp_cnts)
    print(importances)

    # 写统计 txt：和 MMLU 格式完全一致（6 行）
    with open(txt_path, "w") as f:
        f.write(str(accs) + " " + str(sum(accs) / len(qa_pairs)) + "\n")
        f.write(str(resp_cnts) + " " + str(resp_cnts / len(qa_pairs)) + "\n")
        f.write(json.dumps(importances) + "\n")
        f.write(json.dumps([sum(pos) / len(qa_pairs) for pos in zip(*importances)]) + "\n")
        f.write(str(total_prompt_tokens) + "\n")
        f.write(str(total_completion_tokens) + "\n")


if __name__ == "__main__":
    main()
