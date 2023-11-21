import ast
import random
import astunparse
import re
import time
import pandas as pd
from prompt_lib import MMLU_QUESTION, TEMPERATURE, MAX_TOKENS, construct_ranking_message
import openai
from human_eval.data import read_problems, write_jsonl
from human_eval.execution import TimeoutException, create_tempdir, reliability_guard, swallow_io, time_limit
import multiprocessing
from threading import Thread
from typing import Dict, List


def parse_ranks(completion, max_num=4):
    if not isinstance(completion, str):
        content = completion["choices"][0]["message"]["content"]
    else:
        content = completion
    pattern = r'\[([1234]),\s*([1234])\]'
    matches = re.findall(pattern, content)

    try:
        match = matches[-1]
        tops = [int(match[0])-1, int(match[1])-1]
        def clip(x):
            if x < 0:
                return 0
            if x > max_num-1:
                return max_num-1
            return x
        tops = [clip(x) for x in tops]
    except:
        print("error in parsing ranks")
        tops = random.sample(list(range(max_num)), 2)

    return tops

def listwise_ranker_2(responses, question, qtype, model="chatgpt0301"):
    if model == "gpt-3.5-turbo":
        model = "chatgpt0301"
    elif model == "gpt-4":
        model = "gpt4"
    else:
        raise NotImplementedError("Error init model type")
    assert 2 < len(responses) <= 4
    message = construct_ranking_message(responses, question, qtype)
    completion, prompt_tokens, completion_tokens = generate_answer([message], model)
    return parse_ranks(completion, max_num=len(responses)), prompt_tokens, completion_tokens

def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = MMLU_QUESTION.format(question, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer

def get_mmlu_qa_pairs(csv_name):
    df = pd.read_csv(csv_name, header=None)
    ix = len(df)
    return [parse_question_answer(df, idx) for idx in range(ix)]

def get_human_eval_qa_pairs():
    problems = read_problems()
    problems = [(k, v["prompt"], v["entry_point"]) for k, v in problems.items()]
    return problems

def check_function_result(python_code: str, timeout: float = 5.0) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    def unsafe_execute():

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            # Disable functionalities that can make destructive changes to the test.
            reliability_guard()

            # Construct the check program and run it.
            check_program = python_code + "\n"

            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                        exec(check_program, exec_globals)
                result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        passed=result[0] == "passed",
        result=result[0],
    )

def generate_answer(answer_context, model):
    print("question context: ")
    print(answer_context)
    try:
        completion = openai.ChatCompletion.create(
                #   model=model,
                  engine=model,
                  temperature=TEMPERATURE,
                  messages=answer_context,
                  max_tokens=MAX_TOKENS,
                  n=1)
    except Exception as e:
        print(e)
        print("retrying due to an error......")
        time.sleep(10)
        return generate_answer(answer_context, model)

    return completion["choices"][0]["message"]["content"], completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"]

def parse_single_choice(reply):
    pattern = r'\(([ABCDabcd])\)'
    matches = re.findall(pattern, reply)

    solution = None
    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    if solution is None:
        alter_pattern = r'([ABCDabcd])\)'
        alter_matches = re.findall(alter_pattern, reply)
        for match_str in alter_matches[::-1]:
            solution = match_str.upper()
            if solution:
                break

    return solution

def extract_last_python_code_block(text):
    # The regular expression pattern for Python code blocks
    pattern = r"```[pP]ython(.*?)```"

    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # If there are matches, return the last one
    if matches:
        return matches[-1].strip()
    else:
        return None

def parse_code_completion(agent_response, question):
    python_code = extract_last_python_code_block(agent_response)
    if python_code is None:
        if agent_response.count("impl]") == 0:
            python_code = agent_response
        else:
            python_code_lines = agent_response.split("\n")
            python_code = ""
            in_func = False
            for line in python_code_lines:
                if in_func:
                    python_code += line + "\n"
                if "impl]" in line:
                    in_func = True
    if python_code.count("def") == 0:
        python_code = question + python_code
    return python_code

def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False
    
def get_unit_test_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    def get_call_str(assert_statement: str) -> str:
        ast_parsed = ast.parse(assert_statement)
        try:
            call_str = ast_parsed.body[0].test.left # type: ignore
        except:
            call_str = ast_parsed.body[0].test # type: ignore

        return astunparse.unparse(call_str).strip()
    def function_with_timeout(func, args, timeout):
        result_container = []

        def wrapper():
            result_container.append(func(*args))

        thread = PropagatingThread(target=wrapper)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise TimeoutError()
        else:
            return result_container[0]
    try:
        exec(f"from typing import *\n{func}", globals()) # not safe
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return str(output)
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)

def parse_judge_attitude(agent_response, question, role, former_results):
    if role == "Passer":
        attitude = dict()
        for res_code in former_results:
            result = check_function_result(res_code)
            if result["passed"]:
                attitude[res_code] = "The code doesn't have syntax error."
            else:
                attitude[res_code] = "The code has syntax error: " + result["result"]
        return attitude
    elif role == "Tester":
        attitude = dict()
        def parse_tests(tests: str) -> List[str]:
            candidates = [test.strip() for test in tests.splitlines() if "assert" in test]
            candidates = [test for test in candidates if py_is_syntax_valid(test)]
            candidates = list(set(candidates))
            if len(candidates) > 10:
                candidates = candidates[:10]
            return candidates
        tests = parse_tests(agent_response)
        if len(tests) == 0:
            for res_code in former_results:
                attitude[res_code] = ""
            return (attitude, [])
        
        for res_code in former_results:
            passed_tests = []
            failed_tests = []
            for test in tests:
                result = check_function_result(res_code + "\n" + test)
                if result["passed"]:
                    passed_tests.append(test)
                else:
                    output = get_unit_test_output(res_code, test)
                    failed_tests.append(test + " # output: " + output)
                attitude[res_code] = ""
                if len(passed_tests) != 0:
                    attitude[res_code] += "Passed tests:\n"
                    for test in passed_tests:
                        attitude[res_code] += test + "\n"
                if len(failed_tests) != 0:
                    if len(passed_tests) != 0:
                        attitude[res_code] += "\n"
                    attitude[res_code] += "Failed tests:\n"
                    for test in failed_tests:
                        attitude[res_code] += test + "\n"
        return (attitude, tests)
    elif role == "Reflector":
        def parse_reflection(response):
            reflections = []

            # Extract content between [reflection n] and [reflection n+1] or [reflection n] and EOF
            matches = re.findall(r'\[reflection \d+\]:\n(.*?)(?=\[reflection \d+\]:|$)', response, re.DOTALL)

            for match in matches:
                # Strip each extracted reflection content
                reflections.append(match.strip())
            return reflections
        reflections = parse_reflection(agent_response)
        if len(reflections) != len(former_results):
            reflections = ["No reflection"] * len(former_results)
        attitude = dict()
        for res_code, reflection in zip(former_results, reflections):
            attitude[res_code] = reflection
        return attitude
    elif role == "Debugger":
        def parse_reflection(response):
            reflections = []

            # Extract content between [reflection n] and [reflection n+1] or [reflection n] and EOF
            matches = re.findall(r'\[bug fix \d+\]:\n(.*?)(?=\[bug fix \d+\]:|$)', response, re.DOTALL)

            for match in matches:
                # Strip each extracted reflection content
                reflections.append(match.strip())
            return reflections
        reflections = parse_reflection(agent_response)
        if len(reflections) != len(former_results):
            reflections = ["No Bugs"] * len(former_results)
        attitude = dict()
        for res_code, reflection in zip(former_results, reflections):
            attitude[res_code] = reflection
        return attitude
    elif role == "QualityManager":
        def parse_reflection(response):
            reflections = []

            # Extract content between [reflection n] and [reflection n+1] or [reflection n] and EOF
            matches = re.findall(r'\[code review \d+\]:\n(.*?)(?=\[code review \d+\]:|$)', response, re.DOTALL)

            for match in matches:
                # Strip each extracted reflection content
                reflections.append(match.strip())
            return reflections
        reflections = parse_reflection(agent_response)
        if len(reflections) != len(former_results):
            reflections = ["No code review"] * len(former_results)
        attitude = dict()
        for res_code, reflection in zip(former_results, reflections):
            attitude[res_code] = reflection
        return attitude
    elif role == "Ranker":
        attitude = dict()
        tops = parse_ranks(agent_response, max_num=len(former_results)) if 2 < len(former_results) <= 4 else [0, 1]
        for res_id, res_code in enumerate(former_results):
            if res_id in tops:
                attitude[res_code] = True
            else:
                attitude[res_code] = False
        return attitude
    else:
        raise NotImplementedError

def most_frequent(clist, cmp_func):
    counter = 0
    num = clist[0]

    for i in clist:
        current_frequency = sum(cmp_func(i, item) for item in clist)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num, counter

class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret
