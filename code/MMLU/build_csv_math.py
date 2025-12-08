import csv
import sys
import ast
import os

TOTAL_AGENTS = int(sys.argv[1])
RES_DIR = sys.argv[2]
TARGET_CSV = sys.argv[3]

def process_lists(*lists):
    data_dict = {}

    for filename in os.listdir(RES_DIR):
        if filename.endswith('.txt'):
            with open(os.path.join(RES_DIR, filename), 'r') as f:
                lines = f.readlines()
                if len(lines) >= 4:
                    acc_score = float(lines[0].strip().rsplit(' ', 1)[-1])
                    resp = int(lines[1].strip().split(' ', 1)[0])
                    # 用第一行的布尔列表长度来算题目数
                    bool_part = lines[0].strip().rsplit(' ', 1)[0]
                    q_cnt = len(bool_part.strip('[]').split(','))

                    fourth_line = ast.literal_eval(lines[3].strip())
                    if isinstance(fourth_line, list):
                        fourth_line = [fourth_line[i:i+TOTAL_AGENTS]
                                       for i in range(0, len(fourth_line), TOTAL_AGENTS)]
                        sums = [sum(pos) for pos in zip(*fourth_line)]
                        scores = [
                            (sum([sums[idx] for idx in list_[0]]) / len(list_[0]), list_[1])
                            for list_ in lists
                        ]

                        base = filename[:-7]  # 去掉 "_73.txt"
                        # 所有 math 结果都要
                        data_dict[base] = {score[1] + "_imp": score[0] for score in scores}
                        data_dict[base]['acc'] = acc_score
                        data_dict[base]['resp'] = resp
                        data_dict[base]['q_cnt'] = q_cnt

    # 写 CSV
    with open(TARGET_CSV, mode='w', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=['filename'] + [score[1] + "_imp" for score in scores] + ['acc', 'resp', 'q_cnt']
        )
        writer.writeheader()
        for filename, scores in data_dict.items():
            row_dict = {'filename': filename}
            row_dict.update(scores)
            writer.writerow(row_dict)

if __name__ == "__main__":
    lists = []
    idx_list = sys.argv[4:]
    idx_list, name_list = idx_list[:len(idx_list)//2], idx_list[len(idx_list)//2:]
    for idx, arg in enumerate(idx_list):
        try:
            list_ = ast.literal_eval(arg)
            if isinstance(list_, list):
                lists.append((list_, name_list[idx]))
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            print(f"Invalid list argument: {arg}")
            sys.exit(1)
    process_lists(*lists)
