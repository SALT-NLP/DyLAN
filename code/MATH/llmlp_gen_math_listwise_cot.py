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
DIR_NAME = "llmlp_math_cot_" + MODEL
RESPONSES_TOTAL = DIR_NAME+"/responses_total.txt"
TOKENS_TOTAL = DIR_NAME+"/tokens_total.txt"
SYSTEM_PROMPT = "It's a debate. Explain your reasons at each round thoroughly.\nFollow the given examples and answer the mathematics problem."
EXAMPLES = r"""Problem: Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.
Answer: Let's think step by step
Kevin hops $1/3$ of the remaining distance with every hop.
His first hop takes $1/3$ closer.
For his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.
For his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.
In general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.
We want to find how far he has hopped after five hops.
This is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.
Thus, Kevin has hopped $\frac{\frac{1}{3}\left(1-\left(\frac{2}{3}\right)^5\right)}{1-\frac{2}{3}} = \boxed{\frac{211}{243}}$.
The answer is \frac{211}{243}}

Problem: What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?
Answer: Let's think step by step
We rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,
resulting in  $(x+7)^2-49 + (y-2)^2-4=10$,
or $(x+7)^2+(y-2)^2=63$.
This is the equation of a circle with center $(-7, 2)$ and radius $\sqrt{63},$
so the area of this region is $\pi r^2 = \boxed{63\pi}$.
The answer is 63\pi

Problem: If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?
Answer: Let's think step by step
If $(x,y)$ lies on the circle,
so does $(x,-y),$ $(-x,-y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$),
so we can assume that $x \ge 0$ and $y \ge 0.$
Then $|x| + |y| = x + y.$  Squaring, we get
\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\]
Note that $(x - y)^2 \ge 0.$
Expanding, we get $x^2 - 2xy + y^2 \ge 0,$ so $2xy \le x^2 + y^2 = 1.$
Hence,\[1 + 2xy \le 2,\]which means $x + y \le \sqrt{2}.$
Equality occurs when $x = y = \frac{1}{\sqrt{2}},$
so the maximum value of $|x| + |y|$ is $\boxed{\sqrt{2}}.$
The answer is \sqrt{2}

Problem: If $f(x)=\frac{ax+b}{cx+d}, abcd\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?
Answer: Let's think step by step
The condition $f(f(x))$ means that $f$ is the inverse of itself,
so its graph is symmetrical about the line $y = x$.
With a rational function of this form, we will have two asymptotes:
a vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,
and a horizontal one at $y=a/c$,
if we take the limit of $f(x)$ as $x$ goes to $\pm\infty$.
In order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$
so that it and its asymptotes reflect onto themselves.
This means that $-d/c=a/c$,
and therefore $-d=a$ and $a+d=\boxed{0}$.
The answer is 0

Problem: A math teacher requires Noelle to do one homework assignment for each of the first five homework points she wants to earn; for each of the next five homework points, she needs to do two homework assignments; and so on, so that to earn the $n^{\text{th}}$ homework point, she has to do $n\div5$ (rounded up) homework assignments. For example, when she has 11 points, it will take $12\div5=2.4\rightarrow3$ homework assignments to earn her $12^{\text{th}}$ point. What is the smallest number of homework assignments necessary to earn a total of 25 homework points?
Answer: Let's think step by step
Noelle only has to do 1 homework assignment to earn her first point,
and the same is true for each of her first five points.
She must then do 2 homework assignments to earn her sixth point, seventh point, and so on, up to her tenth point.
Continuing, we see that Noelle must do a total of \[1+1+1+1+1+2+2+2+2+2+\dots+5+5+5+5+5\] homework assignments to earn 25 points.
This sum may be rewritten as $5(1+2+3+4+5)=5(15)=\boxed{75}$.
The answer is 75

Problem: The quadratic equation $x^2+mx+n=0$ has roots that are twice those of $x^2+px+m=0,$ and none of $m,$ $n,$ and $p$ is zero. What is the value of $n/p?$
Answer: Let's think step by step
Let $r_1$ and $r_2$ be the roots of $x^2+px+m=0.$
Since the roots of $x^2+mx+n=0$ are $2r_1$ and $2r_2,$ we have the following relationships: \[
m=r_1 r_2,\quad n=4r_1 r_2,\quad p=-(r_1+r_2), \quad\text{and}\quad
m=-2(r_1+r_2).
\] So \[
n = 4m, \quad p = \frac{1}{2}m,
\quad\text{and}\quad
\frac{n}{p}=\frac{4m}{\frac{1}{2}m}=\boxed{8}.
\]
Alternatively, the roots of \[
\left(\frac{x}{2}\right)^2 + p\left(\frac{x}{2}\right) + m = 0
\] are twice those of $x^2 + px + m = 0.$
Since the first equation is equivalent to $x^2 + 2px + 4m = 0,$
we have \[m = 2p \quad\text{and}\quad n = 4m, \quad\text{so}\quad \frac{n}{p} = \boxed{8}.\]
The answer is 8

Problem: Expand $(2z^2 + 5z - 6)(3z^3 - 2z + 1)$.
Answer: Let's think step by step
$$\begin{array}{crrrrrrr}
& & & 3z^3 & & -2z & + 1 & \\
\times & & & & 2z^2 & +5z & -6 \\
\cline{1-7}\rule{0pt}{0.17in}
& & & -18z^3 & & +12z & -6 & \\
& & +15z^4 & & -10z^2 & +5z & & \\
+ & 6z^5 & & -4z^3 & +2z^2 & & & \\
\cline{1-7}\rule{0pt}{0.17in}
& 6z^5 & +15z^4 & -22z^3 & - 8z^2 &+17z & -6 &
\end{array}$$
The answer is 6z^5+15z^4-22z^3-8z^2+17z-6}.

Problem: Find the mean of all solutions for $x$ when $x^3 + 3x^2 - 10x = 0$.
Answer: Let's think step by step
First, we factor the equation as $x(x^2 +3x - 10) = 0$.
So, one solution is $x=0$ and the other two solutions are the solutions to $x^2 + 3x-10=0$.
We could either factor the quadratic, or note that the sum of the solutions to this quadratic is $-(3/1)=-3$,
so the mean of the three solutions to the original equation is $-3/3=\boxed{-1}$.
The answer is -1"""

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
        response = "\n\nOne agent's solution: ```{}```".format(agent_response)

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

    return completion, completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"]


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
    total_prompt_tokens, total_completion_tokens = 0, 0

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
                completion, prompt_tokens, completion_tokens = generate_answer(agent_context)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                store_conetxts[i].extend(agent_context[1:])
                print(completion, "\n")
                total_responses += 1
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

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
                completion, prompt_tokens, completion_tokens = generate_answer(agent_context)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                store_conetxts[i].extend(agent_context[1:])
                print(completion, "\n")
                total_responses += 1
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

                if i >= math.floor(2/3 * len(agent_contexts)) and check_reach_consensus(agent_contexts[:i+1]):
                    response_dict[question] = (store_conetxts, answer, prob_level, prob_type)
                    consensus = True
                    break

            if consensus:
                continue

            # TODO: PageRanker
            message = construct_ranking_message(agent_contexts, question)
            completion, prompt_tokens, completion_tokens = generate_answer([message])
            total_responses += 1
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
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
                completion, prompt_tokens, completion_tokens = generate_answer(agent_context)
                total_responses += 1
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

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
    with open(TOKENS_TOTAL, "a") as f:
        f.write("Prompt tokens: {}, Completion tokens: {}\n".format(total_prompt_tokens, total_completion_tokens))
