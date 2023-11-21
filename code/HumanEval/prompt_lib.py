
TEMPERATURE = 0.8
MAX_TOKENS = 1024

MMLU_QUESTION = "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} "

ROLE_MAP = {
    "PythonAssistant": "You are a Python writing assistant, an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature).", # from https://github.com/composable-models/llm_multiagent_debate.git
    "AlgorithmDeveloper": "You are an algorithm developer. You are good at developing and utilizing algorithms to solve problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
    "ComputerScientist": "You are a computer scientist. You are good at writing high performance code and recognizing corner cases while solve real problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
    "Programmer": "You are an intelligent programmer. You must complete the python function given to you by the user. And you must follow the format they present when giving your answer! You can only respond with comments and actual code, no free-flowing text (unless in a comment).", # from https://github.com/getcursor/eval.git
    "CodingArtist": "You are a coding artist. You write Python code that is not only functional but also aesthetically pleasing and creative. Your goal is to make the code an art form while maintaining its utility. You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
    "SoftwareArchitect": "You are a software architect, skilled in designing and structuring code for scalability, maintainability, and robustness. Your responses should focus on best practices in software design. You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature)."
}

ROLE_MAP_INIT = {
    "PythonAssistant": "You are a Python writing assistant, an AI that only responds with python code, NOT ENGLISH. You will be given a series of previous implementations of the same function signature and docstring. You use the previous implementations as a hint and your goal is to write your full implementation again (restate the function signature). Here're some examples.", # from https://github.com/composable-models/llm_multiagent_debate.git
    "AlgorithmDeveloper": "You are an algorithm developer. You are good at developing and utilizing algorithms to solve problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Remember to follow the format (restate the function signature). Here're some examples.",
    "ComputerScientist": "You are a computer scientist. You are good at writing high performance code and recognizing corner cases while solve real problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Remember to follow the format (restate the function signature). Here're some examples.",
    "Programmer": "You are an intelligent programmer. You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. And you must follow the format they present when giving your answer! You can only respond with comments and actual code, no free-flowing text (unless in a comment). Here're some examples.", # from https://github.com/getcursor/eval.git
    "CodingArtist": "You are a coding artist. You write Python code that is not only functional but also aesthetically pleasing and creative. Your goal is to make the code an art form while maintaining its utility. You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Remember to follow the format (restate the function signature). Here're some examples.",
    "SoftwareArchitect": "You are a software architect, skilled in designing and structuring code for scalability, maintainability, and robustness. Your responses should focus on best practices in software design. You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Remember to follow the format (restate the function signature). Here're some examples."
}

TOOL_LIST = ['Passer']

EXAMPLE_TESTER = """Example:
function signature:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"

unit tests:
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False"""

EXAMPLE_REFLECTOR = """Example:
[previous impl 1]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[previous impl 2]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[reflection 1]:
The implementation will fail when no subarray fulfills the condition. The issue in the implementation is due to the use of >= instead of > in the condition to update the result. Because of this, it returns a subarray even when the sum is greater than the target, as it still updates the result when the current subarray length is equal to the previous longest subarray length. To overcome this error, we should change the condition to only update the result when the current subarray length is strictly greater than the previous longest subarray length. This can be done by replacing >= with > in the condition.

[reflection 2]:
The implementation has an issue stemming from the while loop `while current_sum + nums[right] <= target:`, which directly accesses `nums[right]` without checking if right is within the bounds of the list. This results in a runtime error when right goes beyond the list length. To overcome this error, we need to add a bounds check for the right variable in the mentioned while loop. We can modify the loop condition to `while right < len(nums) and current_sum + nums[right] <= target:`. This change will ensure that we only access elements within the bounds of the list, thus avoiding the IndexError."""

EXAMPLE_DEBUGGER = """Example:
[previous impl 1]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[previous impl 2]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[bug fix 1]:
Expression `right - left + 1 >= max_length` is buggy. I suggest to fix it as `right - left + 1 > max_length`.

[bug fix 2]:
The while loop `while current_sum + nums[right] <= target:` is a bug. It's better to use `while right < len(nums) and current_sum + nums[right] <= target:`."""


EXAMPLE_QUALITY_MANAGER = """Example:
[previous impl 1]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[previous impl 2]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[code review 1]:
- **Efficiency**: Efficient use of the sliding window technique.
- **Correctness**: Correct but updates `result` even for subarrays of equal max length, which could be optimized.
- **Readability**: Clear variable names; adding comments would improve understanding.
- **Edge Cases**: Consider adding checks for empty lists or negative targets.

[code review 2]:
- **Correctness**: Potential `IndexError` due to the initial loop exceeding list bounds.
- **Efficiency**: Redundant initial loop, could be simplified.
- **Readability**: Good variable naming, but initial loop adds unnecessary complexity.
- **Consistency**: Inconsistent update condition for `max_length` compared to the first implementation.
- **Edge Cases**: Similar to the first, needs checks for edge cases and error handling."""


JUDGE_MAP = {
    "Passer": "",
    "Tester": f"""You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring. Here's the example.

{EXAMPLE_TESTER}""",
    "Reflector": f"You are a Python writing assistant. You will be given a series of function implementations of the same function signature. Write a few sentences to explain whether and why the implementations are wrong. These comments will be used as a hint and your goal is to write your thoughts on the n-th previous implementation after [reflection n]. Here's the example.\n{EXAMPLE_REFLECTOR}",
    "Ranker": "You are a Python writing assistant. You will be given a series of function implementations of the same function signature. You need to choose the best 2 implementations in consideration of correctness, efficiency, and possible corner cases.",
    "Debugger": f"You are a debugger, specialized in finding and fixing bugs in Python code. You will be given a function implementation with a bug in it. Your goal is to identify the bug and provide a corrected implementation. Include comments to explain what was wrong and how it was fixed. Here's the example.\n{EXAMPLE_DEBUGGER}",
    "QualityManager": f"You are a quality manager, ensuring that the code meets high standards in terms of readability, efficiency, and accuracy. You will be given a function implementation and you need to provide a code review. Comment on its correctness, efficiency, and readability, and suggest improvements if needed. Here's the example.\n{EXAMPLE_QUALITY_MANAGER}"
}

JUDGE_PREFIX = {
    "Passer": "[syntax check {}]:\n",
    "Tester": "[unit test results {}]:\n",
    "Reflector": "[reflection {}]:\n",
    "Debugger": "[bug fix {}]:\n",
    "QualityManager": "[code review {}]:\n",
}

JUDGE_NAMES = {
    "Passer": "Syntax Checker",
    "Tester": "Unit Tests",
    "Reflector": "Reflector",
    "Debugger": "Debugger",
    "QualityManager": "Quality Manager",
}


SYSTEM_PROMPT_MMLU = "Here's a debate. Explain your reasons at each round thoroughly.\nAll questions are single choice."
SYSTEM_PROMPT_HUMAN_EVAL_INIT = ""
'''Example 1:
[function impl]:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a + b
```

Example 2:
[function impl]:
```python
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    if not words:
        return []

    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
```
END EXAMPLE'''
SYSTEM_PROMPT_HUMAN_EVAL = ""
'''Example 1:
[previous impl 1]:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a - b
```

[syntax check 1]:
The code doesn't have syntax error.

[unit test results 1]:
Passed tests:
assert add(0, 0) == 0

Failed tests:
assert add(1, 2) == 3 # output: -1

[reflection 1]:
From the docstring, we should put the operator `+` in the return statement. This will ensure that the function returns the correct output for the given input.

[improved impl]:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    # the operator should be `+` instead of `-`
    return a + b
```

Example 2:
[previous impl 1]:
```python
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
```

[syntax check 1]:
The code doesn't have syntax error.

[unit test results 1]:
Failed tests:
assert fullJustify([], 10) == [] # output: ['          ']
assert fullJustify([], 0) == [] # output: ['']

[reflection 1]:
From the docstring, the formatting of each line is done in a way that distributes the spaces between words as evenly as possible. If the number of spaces cannot be evenly distributed, the extra spaces are added from left to right.
For a line with only one word, the remaining spaces are added to the end of the line. The last line is left-justified, and no extra space is inserted between words. Any remaining spaces are added to the end of the line.
The code should handle the case where there are no words to process. We should add a condition at the beginning of the function to check if the input list is empty, and return an empty list if it is. This will ensure that the function returns the correct output for empty input lists.

[improved impl]:
```python
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    # check if the input list is empty, and return an empty list if it is
    if not words:
        return []

    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
```
END EXAMPLE'''

def construct_ranking_message(responses, question, qtype):
    if qtype == "single_choice":
        prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nPlease choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."
        return {"role": "user", "content": prefix_string}

    elif qtype == "code_completion":
        prefix_string = "Here are some implementations of the same function and the thoughts of them. The function has a signature and a docstring explaining its functionality.\n\n"
        for aid, agent_response in enumerate(responses, start=1):
            response = "[function impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + """[function signature]:\n```python\n{}\n```\n\nTake correctness, efficiency, and possible corner cases into consideration, choose top 2 solutions that match the function's docstring best. Think it step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response.""".format(question)
        return {"role": "user", "content": prefix_string}
    else:
        raise NotImplementedError

def construct_message(responses, question, qtype):
    if qtype == "single_choice":
        if len(responses) == 0:
            prefix_string = "Here is the question:\n" + question + "\n\nPut your answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), or (D)."
            return {"role": "user", "content": prefix_string}

        prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and that other agents step by step. Notice that their answers might be all wrong. Put your answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), or (D). Along with the answer, give a score ranged from 1 to 5 to the solutions of other agents. Put all {} scores in the form like [[1, 5, 2, ...]].".format(len(responses))
        return {"role": "user", "content": prefix_string}

    elif qtype == "code_completion":
        if len(responses) == 0:
            qtemplate = "```python\n{}\n```"
            qtemplate = '''You must complete the python function I give you.
Be sure to use the same indentation I specified. Furthermore, you may only write your response in code/comments.
[function impl]:
{}\nOnce more, please follow the template by repeating the original function, then writing the completion.'''.format(qtemplate.format(question))
            return {"role": "user", "content": qtemplate}
        else:
            flag_ranker = False
            ranker = None
            for agent in responses:
                if agent.role == "Ranker":
                    flag_ranker = True
                    ranker = agent
                    break
            pre_impls = list(responses[0].get_answer().keys())
            if flag_ranker:
                ranks = ranker.get_answer()
                pre_impls = [pre_impl for pre_impl in pre_impls if ranks[pre_impl]]
            
            prefix_string = ""
            for impl_id, pre_impl in enumerate(pre_impls, start=1):
                prefix_string += "[previous impl {}]:\n```python\n{}\n```\n\n".format(impl_id, pre_impl)
                for agent in responses:
                    if agent.role == "Ranker":
                        continue
                    reply = agent.get_answer()[pre_impl]
                    prefix_string += JUDGE_PREFIX[agent.role].format(impl_id) + reply + "\n\n"
            
            def connect_judges(judges):
                # connect the name by ',' and "and"
                judges = [judge for judge in judges if judge.role != "Ranker"]
                if len(judges) == 0:
                    raise ValueError("No judges")
                if len(judges) == 1:
                    return JUDGE_NAMES[judges[0].role]
                if len(judges) == 2:
                    return JUDGE_NAMES[judges[0].role] + " and " + JUDGE_NAMES[judges[1].role]
                return ", ".join([JUDGE_NAMES[judge.role] for judge in judges[:-1]]) + " and " + JUDGE_NAMES[judges[-1].role]

            prefix_string = prefix_string + "You must complete the python function I give you by rectifying previous implementations. Use the other information as a hint.\nBe sure to use the same indentation I specified. Furthermore, you may only write your response in code/comments.\n[improved impl]:\n```python\n{}\n```\n\nPlease follow the template by repeating the function signature and complete the new implementation in [improved impl]. If no changes are needed, simply rewrite the implementation in the Python code block.\nAlong with the new implementation, give a score ranged from 1 to 5 to {} in terms of helpfulness. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of your response.".format(question, connect_judges(responses), len(responses)-1)

            return {"role": "user", "content": prefix_string}
    else:
        raise NotImplementedError

def construct_judge_message(responses, question, qtype, role):
    if qtype == "code_completion":
        if role == "Tester":
            prefix_string = "function signature:\n```python\n{}\n```\n\nunit tests:\n".format(question)
            return {"role": "user", "content": prefix_string}
        elif role == "Reflector":
            prefix_string = "Here are previous implementations of the same function.The function has a signature and a docstring explaining its functionality.\n\n"
            for aid, agent_response in enumerate(responses, start=1):
                response = "[previous impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)

                prefix_string = prefix_string + response
            prefix_string = prefix_string + "Write your reflection on these implementations in consideration of correctness, efficiency, and possible corner cases. Put your reflection of the n-th implementation after [reflection n].\nAlong with the reflections, give a score ranged from 1 to 5 to each previous implementation. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of your response.".format(len(responses))
            return {"role": "user", "content": prefix_string}
        elif role == "Debugger":
            prefix_string = "Here are previous implementations of the same function.The function has a signature and a docstring explaining its functionality.\n\n"
            for aid, agent_response in enumerate(responses, start=1):
                response = "[previous impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)

                prefix_string = prefix_string + response
            prefix_string = prefix_string + "Debug this version of implementation and write your feedback as a debugger. Put your reflection of the n-th implementation after [reflection n].\nAlong with the reflections, give a score ranged from 1 to 5 to each previous implementation. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of your response.".format(len(responses))
            return {"role": "user", "content": prefix_string}
        elif role == "QualityManager":
            prefix_string = "Here are previous implementations of the same function.The function has a signature and a docstring explaining its functionality.\n\n"
            for aid, agent_response in enumerate(responses, start=1):
                response = "[previous impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)

                prefix_string = prefix_string + response
            prefix_string = prefix_string + "Write your code review on these implementations in multiple aspects. Put your reflection of the n-th implementation after [reflection n].\nAlong with the reflections, give a score ranged from 1 to 5 to each previous implementation. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of your response.".format(len(responses))
            return {"role": "user", "content": prefix_string}
        elif role == "Ranker":
            prefix_string = "Here are some implementations of the same function. The function has a signature and a docstring explaining its functionality.\n\n"
            for aid, agent_response in enumerate(responses, start=1):
                response = "[function impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)

                prefix_string = prefix_string + response

            prefix_string = prefix_string + "[function signature]:\n```python\n{}\n```\n\nTake correctness, efficiency, and possible corner cases into consideration, choose top 2 solutions that match the function's docstring best. Think it step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response.".format(question)
            return {"role": "user", "content": prefix_string}
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
