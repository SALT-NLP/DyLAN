import math
import re
import pandas as pd
import json
import time
import random
import openai
import sys
import os
from util import _strip_string, extract_math_answer, is_equiv
import backoff
from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError, Timeout
from util import OutOfQuotaException, AccessTerminatedException


SUB_DIR = sys.argv[1]
MIN_FILENAME = int(sys.argv[2])
MAX_FILENAME = int(sys.argv[3])
MODEL = sys.argv[4]
ENGINE = sys.argv[5]
DIR_NAME = "llmlp_math_" + MODEL
RESPONSES_TOTAL = DIR_NAME+"/responses_total.txt"
SYSTEM_PROMPT = "It's a debate. Explain your reasons at each round thoroughly.\n Follow the given examples and answer the mathematics problem."
EXAMPLES = """Problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6 trees that were planted. The answer is 6.
###
Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot. The answer is 5.
###
Problem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total. The answer is 39.
###
Problem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8.
###
Problem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 * 2 = 4 more toys. Now he has 5 + 4 = 9 toys. The answer is 9.
###
Problem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer: There were originally 9 computers. For each day from monday to thursday, 5 more computers were installed. So 4 * 5 = 20 computers were added. Now 9 + 20 = 29 computers are now in the server room. The answer is 29.
###
Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Michael started with 58 golf balls. He lost 23 on Tuesday, and lost 2 more on wednesday. So he had 58 - 23 = 35 at the end of Tuesday, and 35 - 2 = 33 at the end of wednesday. The answer is 33.
###
Problem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? 
Answer: Olivia had 23 dollars. She bought 5 bagels for 3 dollars each. So she spent 5 * 3 = 15 dollars. Now she has 23 - 15 = 8 dollars left. The answer is 8."""

# openai.api_key =
# openai.api_base =
# openai.api_type =
# openai.api_version =

def construct_message(agents, question):
    if len(agents) == 0:
        # unused
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), (D)."}

    prefix_string = "Follow the given examples and answer the mathematics problem.\n\n" + question +  "\n\nThese are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[-1]["content"]
        response = "\n\nOne agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and that other agents step by step. Notice that the former answers might be all wrong.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_ranking_message(agents, question):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), (D)."}

    prefix_string = "Follow the given examples and answer the mathematics problem.\n\n" + question +  "\n\nThese are the solutions to the problem from other agents: "

    for aid, agent in enumerate(agents, 1):
        agent_response = agent[-1]["content"]
        response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\nPlease choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response.".format(question)
    return {"role": "user", "content": prefix_string} #TODO: add role as judge


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


@backoff.on_exception(backoff.expo, (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError, Timeout), max_tries=20)
def generate_answer(answer_context):
    try:
        completion = openai.ChatCompletion.create(
                #   model=MODEL,
                  engine=ENGINE,
                  messages=answer_context,
                  temperature=0.2,
                  max_tokens=2048,
                  n=1)
    except RateLimitError as e:
        if "You exceeded your current quota, please check your plan and billing details" in e.user_message:
            raise OutOfQuotaException(openai.api_key)
        elif "Your access was terminated due to violation of our policies" in e.user_message:
            raise AccessTerminatedException(openai.api_key)
        else:
            raise e

    return completion


def parse_question_answer(subdir, file):
    
    def find_math_answer(s):
        assert('boxed' in s)
        # s = s.replace(",", "")
        ans = s.split('boxed')[-1]
        if(ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if(c == '{'):
                    stack += 1
                    a += c
                elif(c == '}'):
                    stack -= 1
                    if(stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a=_strip_string(a)
        return a

    with open(os.path.join(subdir, file), 'r') as fp:
        try:
            problem_data = json.load(fp)
        except Exception as e:
            print(f"Error loading JSON from {file}", e)
            raise e
        prob_content = problem_data["problem"]
        question = EXAMPLES + "\n\nPlease solve the problem below.\nProblem: " + prob_content + "\nAnswer:"
        prob_level = problem_data["level"]
        prob_type = problem_data["type"]
        try:
            prob_level = int(prob_level.split("Level ")[1])
        except:
            prob_level = None

        # answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))
        answer = find_math_answer(problem_data['solution'])

        return question, prob_level, prob_type, answer

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

def check_reach_consensus(agent_contexts):
    pred_solutions = [context[-1]["content"] for context in agent_contexts]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = extract_math_answer(pred_solution)
        if pred_answer:
            pred_answers.append(pred_answer)

    if len(pred_answers) == 0:
        print("No answer found")
        return False
    
    def most_frequent(List):
        counter = 0
        num = List[0]

        for i in List:
            current_frequency = sum(is_equiv(i, item) for item in List)
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
    idx = 0
    total_responses = 0

    for subdir, dirs, files in os.walk(SUB_DIR):
        for file in files:
            file_num = int(os.path.splitext(file)[0])  # Get the filename without extension and convert to int
            if MIN_FILENAME <= file_num <= MAX_FILENAME:
                question, prob_level, prob_type, answer = parse_question_answer(subdir, file)
            else:
                continue

            agent_contexts = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}] for _ in range(agents)]
            store_conetxts = [[{"role": "system", "content": SYSTEM_PROMPT}] for _ in range(agents)]

            consensus = False
            for i, agent_context in enumerate(agent_contexts):
                print(idx, 0, i, agent_context, "\n")
                completion = generate_answer(agent_context)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                store_conetxts[i].extend(agent_context[1:])
                print(completion, "\n")
                total_responses += 1

                if i >= math.floor(2/3 * len(agent_contexts)) and check_reach_consensus(agent_contexts[:i+1]):
                    response_dict[question] = (store_conetxts[:i+1], answer, prob_level, prob_type)
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
                store_conetxts[i].extend(agent_context[1:])
                print(completion, "\n")
                total_responses += 1

                if i >= math.floor(2/3 * len(agent_contexts)) and check_reach_consensus(agent_contexts[:i+1]):
                    response_dict[question] = (store_conetxts, answer, prob_level, prob_type)
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
                response_dict[question] = (agent_contexts, answer, prob_level, prob_type)
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
                store_conetxts[i].extend(agent_context[1:])
                print(completion, "\n")

            response_dict[question] = (store_conetxts, answer, prob_level, prob_type)
            idx += 1
        
    # create a directory if not exists
    try:
        os.mkdir(DIR_NAME)
    except:
        pass

    json.dump(response_dict, open(DIR_NAME+"/{}_{}_{}_{}_{}.json".format(os.path.basename(os.path.normpath(SUB_DIR)), MIN_FILENAME, MAX_FILENAME, agents, rounds), "w"))
    with open(RESPONSES_TOTAL, "a") as f:
        f.write("{}\n".format(total_responses))
