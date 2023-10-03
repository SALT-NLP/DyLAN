import json
import os
import openai
import numpy as np
import time
import re
import sys

RES_JSON_DIR = sys.argv[1]
FILTER = sys.argv[2]

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

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def parse_yes_no(string):
    """
    Parses a string containing "yes" or "no" and returns a boolean value.

    Args:
        string (str): The string to parse.

    Returns:
        bool: True if the string contains "yes", False if the string contains "no".

    Raises:
        ValueError: If the input string does not contain "yes" or "no".
    """
    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None


def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def parse_answer(input_str):
    pattern = r'\(([ABCDabcd])\)'
    matches = re.findall(pattern, input_str)

    solution = None
    # print("predicted solution")
    # print(input_str)
    # print("matches")
    # print(matches)

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    if solution is None:
        alter_pattern = r'([ABCDabcd])\)'
        alter_matches = re.findall(alter_pattern, input_str)
        for match_str in alter_matches[::-1]:
            solution = match_str.upper()
            if solution:
                break

    return solution


def compute_accuracy(gt, pred_solutions):
    if type(pred_solutions) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            if pred_answer is None:
                pred_answer = solve_math_problems(pred_solution)
                print(pred_solution)

            if pred_answer is not None:
                pred_answers.append(pred_answer)

        # filter except ABCD
        pred_answers = [answer for answer in pred_answers if answer in ["A", "B", "C", "D"]]

        if len(pred_answers) == 0:
            print("No answer found")
            return 0
        pred_answer = most_frequent(pred_answers)
        # pred_answer = pred_answers[0]
    else:
        pred_answer = parse_answer(pred_solutions)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solutions)

    if gt == pred_answer:
        return 1
    else:
        return 0


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

def parse_resp(file):
    with open(file, "r") as f:
        lines = f.readlines()
    
    return sum([int(line.strip()) for line in lines])

if __name__ == "__main__":
    accuracies = []
    resp_cnt = 0

    sub_accs = {}
    for key in CATEGORIES:
        sub_accs[key] = []

    for file in os.listdir(RES_JSON_DIR):
        if FILTER != 'None' and FILTER not in file:
            continue

        if file.endswith(".txt"):
            resp_cnt = parse_resp(os.path.join(RES_JSON_DIR, file))

        if not file.endswith(".json"):
            continue

        response_dict = json.load(open(os.path.join(RES_JSON_DIR, file), "r"))
        questions = list(response_dict.keys())



        for question in questions:
            responses, gt = response_dict[question]

            pred_solutions = []
            max_len = max([len(response) for response in responses])
            for response in responses:
                if len(response) < max_len:
                    continue
                pred_solution = response[-1]['content']

                pred_solutions.append(pred_solution)
                # break

            # pred_solutions = pred_solutions[:1]

            accurate = compute_accuracy(gt, pred_solutions)


            if accurate is not None:
                accuracies.append(float(accurate))
                sub_cat = SUBCATEGORY[file.split("_test")[0]][0]
                for key in sub_accs:
                    if sub_cat in CATEGORIES[key]:
                        sub_accs[key].append(float(accurate))
                        break
            else:
                import pdb
                pdb.set_trace()
                print(gt)

    print("total:", len(accuracies))
    if resp_cnt != 0:
        print("resp:", resp_cnt/len(accuracies))
    print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))
    for key in sub_accs:
        print(key, np.mean(sub_accs[key]), np.std(sub_accs[key]) / (len(sub_accs[key]) ** 0.5), len(sub_accs[key]))
    # print(accuracies)

