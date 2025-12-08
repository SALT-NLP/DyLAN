import sys
import ast
import os
from numpy import argmax

TOTAL_AGENTS = int(sys.argv[1])
RES_DIR = sys.argv[2]
TARGET_CSV = sys.argv[3]

def process_lists(*lists):
    imp_tests = [[] for _ in lists]

    for filename in os.listdir(RES_DIR):
        if filename.endswith('.txt'):
            with open(os.path.join(RES_DIR, filename), 'r') as f:
                lines = f.readlines()
                if len(lines) >= 4:
                    fourth_line = ast.literal_eval(lines[3].strip())
                    if isinstance(fourth_line, list):
                        fourth_line = [fourth_line[i:i+TOTAL_AGENTS] 
                                       for i in range(0, len(fourth_line), TOTAL_AGENTS)]
                        sums = [sum(pos) for pos in zip(*fourth_line)]
                        scores = [sum([sums[idx] for idx in list_]) / len(list_) for list_ in lists]

                        norms = [(scores[tid] - sum(scores)/len(scores)) / (sum(scores)/len(scores))
                                 for tid in range(len(imp_tests))]
                        norms = list(enumerate(norms))
                        norms.sort(key=lambda x: x[1], reverse=True)
                        for tid, norm in norms[:2]:
                            imp_tests[tid].append((filename[:-4], norm))

    for idx, list_ in enumerate(lists, 1):
        print(f"List {idx}: {list_}")
        # 只按 score 排序，不再过滤 test/merged
        imp_tests[idx-1].sort(key=lambda x: x[1], reverse=True)

        print("best tests:", imp_tests[idx-1])
        print()
        print("best 10 tests:")
        for test in imp_tests[idx-1][:10]:
            print(test[0], test[1])
        print()

if __name__ == "__main__":
    lists = []
    for arg in sys.argv[4:]:
        try:
            list_ = ast.literal_eval(arg)
            if isinstance(list_, list):
                lists.append(list_)
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            print(f"Invalid list argument: {arg}")
            sys.exit(1)
    process_lists(*lists)
