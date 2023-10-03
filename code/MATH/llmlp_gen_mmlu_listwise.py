import math
import os
import re
import pandas as pd
import json
import time
import random
import openai
import sys

QUERY_CSV = sys.argv[1]
EXP_NAME = sys.argv[2]
MODEL = sys.argv[3]
ENGINE = sys.argv[4]
DIR_NAME = "llmlp_mmlu_" + MODEL
RESPONSES_TOTAL = DIR_NAME+"/responses_total.txt"
SYSTEM_PROMPT = "It's a debate. Explain your reasons at each round thoroughly.\nAll questions are single choice."

# openai.api_key =
# openai.api_base =
# openai.api_type =
# openai.api_version =

def construct_message(agents, question):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), (D)."}

    prefix_string = "Here is the question: " + question + "\n\nThese are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[-1]["content"]
        response = "\n\nOne agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and that other agents step by step. Notice that their answers might be all wrong. Put your answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), (D)."""
    return {"role": "user", "content": prefix_string}


def construct_ranking_message(agents, question):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), (D)."}

    prefix_string = "Here is the question: " + question +  "\n\nThese are the solutions to the problem from other agents: "

    for aid, agent in enumerate(agents, 1):
        agent_response = agent[-1]["content"]
        response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\nPlease choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def generate_answer(answer_context):
    try:
        completion = openai.ChatCompletion.create(
                #   model=MODEL,
                  engine=ENGINE,
                  messages=answer_context,
                #   temperature=0.2,
                  n=1)
        if "content" not in completion["choices"][0]["message"]:
            print("\nno content in completion")
            print(completion)
            print(answer_context)
            print("\n")
        assert "content" in completion["choices"][0]["message"]
    except:
        print("retrying due to an error......")
        time.sleep(20)
        return generate_answer(answer_context)

    return completion


def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} Explain your answer, putting the answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), (D).".format(question, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer

def parse_ranks(completion):
    content = completion["choices"][0]["message"]["content"]
    pattern = r'\[([1234]),\s*([1234])\]'
    matches = re.findall(pattern, content)

    try:
        match = matches[-1]
        tops = [int(match[0])-1, int(match[1])-1]
        def clip(x):
            if x < 0:
                return 0
            if x > 3:
                return 3
            return x
        tops = [clip(x) for x in tops]
    except:
        print("error in parsing ranks")
        tops = [0, 1]

    return tops

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

def check_reach_consensus(agent_contexts):
    pred_solutions = [context[-1]["content"] for context in agent_contexts]
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
        return False
    
    def most_frequent(List):
        counter = 0
        num = List[0]

        for i in List:
            current_frequency = List.count(i)
            if current_frequency > counter:
                counter = current_frequency
                num = i

        return num, counter
    
    consensus_answer, counter = most_frequent(pred_answers)
    if counter > math.floor(2/3 * len(agent_contexts)):
        print("Consensus answer: {}".format(consensus_answer))
        return True


if __name__ == "__main__":
    agents = 4
    rounds = 3

    random.seed(0)
    response_dict = {}

    df = pd.read_csv(QUERY_CSV, header=None)
    ix = len(df)
    total_responses = 0

    for idx in range(ix):
        question, answer = parse_question_answer(df, idx)

        agent_contexts = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}] for _ in range(agents)]
        store_contexts = [[{"role": "system", "content": SYSTEM_PROMPT}] for _ in range(agents)]

        consensus = False
        for i, agent_context in enumerate(agent_contexts):
            print(idx, 0, i, agent_context, "\n")
            completion = generate_answer(agent_context)

            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)
            store_contexts[i].extend(agent_context[1:])
            print(completion, "\n")
            total_responses += 1

            if i >= math.floor(2/3 * len(agent_contexts)) and check_reach_consensus(agent_contexts[:i+1]):
                response_dict[question] = (store_contexts[:i+1], answer)
                consensus = True
                break

        if consensus:
            continue

        consensus = False
        message = construct_message(agent_contexts, question)
        for i, agent_context in enumerate(agent_contexts):
            agent_context.pop()
            agent_context.pop()
            agent_context.append(message)
            print(idx, 1, i, agent_context, "\n")
            completion = generate_answer(agent_context)

            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)
            store_contexts[i].extend(agent_context[1:])
            print(completion, "\n")
            total_responses += 1

            if i >= math.floor(2/3 * len(agent_contexts)) and check_reach_consensus(agent_contexts[:i+1]):
                response_dict[question] = (store_contexts, answer)
                consensus = True
                break

        if consensus:
            continue

        # TODO: PageRanker
        message = construct_ranking_message(agent_contexts, question)
        completion = generate_answer([message])
        total_responses += 1
        print(completion, "\n")
        tops = parse_ranks(completion)
        agent_contexts = [agent_contexts[top] for top in tops]

        if check_reach_consensus(agent_contexts):
            response_dict[question] = (agent_contexts, answer)
            continue

        message = construct_message(agent_contexts, question)
        for i, agent_context in enumerate(agent_contexts):
            agent_context.pop()
            agent_context.pop()
            agent_context.append(message)
            print(idx, 2, i, agent_context, "\n")
            completion = generate_answer(agent_context)
            total_responses += 1

            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)
            print(completion, "\n")
            store_contexts[i].extend(agent_context[1:])

        response_dict[question] = (store_contexts, answer)

    # create a directory if not exists
    try:
        os.mkdir(DIR_NAME)
    except:
        pass

    json.dump(response_dict, open(DIR_NAME+"/{}_{}_{}.json".format(EXP_NAME, agents, rounds), "w"))
    print("average responses per question: {}".format(total_responses/ix))
    with open(RESPONSES_TOTAL, "a") as f:
        f.write("{}\n".format(total_responses))
