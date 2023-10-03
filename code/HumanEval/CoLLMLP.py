import ast
import math
import random
from LLM_Neuron import LLMNeuron, JudgeNeuron, LLMEdge
from utils import check_function_result, parse_single_choice, parse_code_completion, most_frequent, parse_judge_attitude, listwise_ranker_2
from sacrebleu import sentence_bleu


ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1} # TODO: only 0 is implemented
CODE_THRESHOLD = 0.9

class CoLLMLP:
    
    def __init__(self, default_model_name, agents=4, agent_roles=[], judges=4, judge_roles=[],
                 rounds=2, activation="listwise", qtype="single_choice", mtype="gpt-3.5-turbo"):
        self.default_model_name = default_model_name
        self.agents = agents
        self.judges = judges
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype
        
        assert len(agent_roles) == agents and agents > 0
        assert len(judge_roles) == judges and judges > 0
        self.agent_roles = agent_roles
        self.judge_roles = judge_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "code_completion":
            self.cmp_res = lambda x, y: sentence_bleu(x, [y], lowercase=True).score >= CODE_THRESHOLD * 100
            self.ans_parser = parse_code_completion
        else:
            raise NotImplementedError("Error init qtype")
    
        self.init_nn(self.activation, self.agent_roles, self.judge_roles)

    def init_nn(self, activation, agent_roles, judge_roles):
        self.nodes, self.edges = [], []
        for idx in range(self.agents):
            self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
        
        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
            for idx in range(self.judges):
                self.nodes.append(JudgeNeuron(judge_roles[idx], self.mtype, parse_judge_attitude, self.qtype))
                for a1 in agents_last_round:
                    self.edges.append(LLMEdge(a1, self.nodes[-1]))
            agents_last_round = self.nodes[-self.judges:]

            for idx in range(self.agents):
                self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
                # print(len(agents_last_round)) !!!
                for a1 in agents_last_round:
                    self.edges.append(LLMEdge(a1, self.nodes[-1]))
            agents_last_round = self.nodes[-self.agents:]

        if activation == 0:
            self.activation = listwise_ranker_2
            self.activation_cost = 1
        else:
            raise NotImplementedError("Error init activation func")
    
    def zero_grad(self):
        for edge in self.edges:
            edge.zero_weight()

    def cut_def_question(self, func_code, question, entry_point):
        def parse_imports(src_code):
            res = []
            for line in src_code.split("\n"):
                if "import" in line:
                    res.append(line)
            res = ["    " + line.strip() for line in res]
            return res
        import_lines = parse_imports(func_code)

        def extract_functions_with_body(source_code):
            # Parse the source code to an AST
            tree = ast.parse(source_code)

            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if the function is nested inside another function
                    # We can determine this by checking the ancestors of the node
                    parents = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    nesting_level = sum(1 for parent in parents if
                                        parent.lineno <= node.lineno and parent.end_lineno >= node.end_lineno)
                    
                    if nesting_level == 1:  # Only top-level functions
                        start_line = node.lineno - 1
                        end_line = node.end_lineno
                        function_body = source_code.splitlines()[start_line:end_line]
                        functions.append("\n".join(function_body))
                    
            return functions
        try:
            funcs = extract_functions_with_body(func_code)
        except:
            funcs = [func_code]

        def extract_func_def(src_code):
            for line in src_code.split("\n"):
                if "def" in line and entry_point in line:
                    return line
            return ""
        que_func = extract_func_def(question)

        for fiid, func_ins_code in enumerate(funcs):
            if question in func_ins_code:
                func_ins_code = func_ins_code.split(question)[-1]
            elif question.strip() in func_ins_code:
                func_ins_code = func_ins_code.split(question.strip())[-1]
            elif que_func in func_ins_code:
                # remove the line before def
                res_lines = func_ins_code.split("\n")
                func_ins_code = ""
                in_func = False
                for line in res_lines:
                    if in_func:
                        func_ins_code += line + "\n"
                    if "def" in line:
                        in_func = True
            else:
                continue

            other_funcs = []
            for other_func in funcs[:fiid] + funcs[fiid+1:]:
                other_func = other_func.split("\n")
                other_func = other_func[:1] + import_lines + other_func[1:]
                other_func = "\n".join(other_func)
                other_funcs.append(other_func)
                        
            return "\n".join(import_lines) + "\n" + func_ins_code + "\n" + "\n".join(other_funcs)
        
        res_lines = func_code.split("\n")
        func_code = ""
        in_func = False
        for line in res_lines:
            if in_func:
                func_code += line + "\n"
            if "def" in line:
                in_func = True
        
        return "\n".join(import_lines) + "\n" + func_code

    def check_consensus(self, idxs, idx_mask, question, entry_point):
        # check consensus based on idxs (range) and idx_mask (actual members, might exceed the range)
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        python_codes = []
        backup = []
        for cand in candidates:
            result = check_function_result(cand)
            if result["passed"]:
                python_codes.append(cand)
            else:
                backup.append(cand)
        
        if len(python_codes) == 0:
            return False, None

        pred_answers = []
        for python_code in python_codes:
            python_code = self.cut_def_question(python_code, question, entry_point)
            pred_answers.append(python_code)

        consensus_answer, ca_cnt = most_frequent(pred_answers, self.cmp_res)
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            print("Consensus answer: {}".format(consensus_answer))
            return True, consensus_answer
        return False, None

    def get_final_result(self, idxs, question, entry_point):
        # check consensus based on idxs (range) and idx_mask (actual members, might exceed the range)
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        python_codes = []
        backup = []
        for cand in candidates:
            result = check_function_result(cand)
            if result["passed"]:
                python_codes.append(cand)
            else:
                backup.append(cand)
        
        if len(python_codes) == 0:
            for rid in range(self.rounds-2, -1, -1):
                for idx in range(self.agents):
                    if self.nodes[rid*(self.agents+self.judges)+idx].active:
                        result = check_function_result(self.nodes[rid*(self.agents+self.judges)+idx].get_answer())
                        if result["passed"]:
                            python_codes.append(self.nodes[rid*(self.agents+self.judges)+idx].get_answer())
                
                if len(python_codes) > 0:
                    break
        
        if len(python_codes) == 0:
            python_codes = backup
        
        final_res = random.choice(python_codes)
        final_res = self.cut_def_question(final_res, question, entry_point)

        print("Final answer: {}".format(final_res))
        return final_res

    def all_tests_and_get_final_result(self, question, unit_tests, entry_point):
        candidates = [self.nodes[idx].get_answer() for idx in range(self.agents) if isinstance(self.nodes[idx], LLMNeuron) and self.nodes[idx].active]
        candidates = [self.cut_def_question(cand, question, entry_point) for cand in candidates]
        python_codes = []
        for cand in candidates:
            passed_tests = []
            failed_tests = []
            for test in unit_tests:
                result = check_function_result(question + "\n" + cand + "\n" + test)
                if result["passed"]:
                    passed_tests.append(test)
                else:
                    failed_tests.append(test)
            python_codes.append((cand, len(passed_tests), passed_tests, failed_tests))
        
        # Sort the codes based on the number of passed tests in descending order
        sorted_codes = sorted(python_codes, key=lambda x: x[1], reverse=True)
        
        # Get the maximum number of passed tests
        max_passed_tests = sorted_codes[0][1]
        
        # Filter the codes that have the maximum number of passed tests
        top_codes = [code for code in sorted_codes if code[1] == max_passed_tests]
        if len(top_codes) < 5:
            top_codes = [code for code in sorted_codes if code[1] == max_passed_tests or code[1] == max_passed_tests - 1]
        
        # Randomly select one of the top codes
        selected_code = random.choice(top_codes)
        
        return selected_code[0]  # Return the code part of the tuple

    def set_allnodes_deactivated(self):
        for node in self.nodes:
            node.deactivate()

    def forward(self, question, entry_point):
        def get_completions():
            # get completions
            completions = [[] for _ in range(self.agents+self.judges)]
            for rid in range(self.rounds):
                for idx in range((self.agents+self.judges)*rid, (self.agents+self.judges)*(rid+1)):
                    if idx < (self.agents+self.judges)*rid + self.agents and self.nodes[idx].active:
                        completions[idx % (self.agents+self.judges)].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % (self.agents+self.judges)].append(None)
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        unit_tests = []
        self.set_allnodes_deactivated()
        assert self.rounds > 2
        # question = format_question(question, self.qtype)
        
        # shuffle the order of agents
        loop_indices = list(range(self.agents))
        random.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            print(0, idx)
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        
            # if idx >= math.floor(2/3 * self.agents):
            #     reached, reply = self.check_consensus(activated_indices, list(range(self.agents)), question)
            #     if reached:
            #         return reply, resp_cnt, get_completions()

        for idx, node_idx in enumerate(range(self.agents, self.agents+self.judges)):
            print(0.5, idx)
            self.nodes[node_idx].activate(question)
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            resp_cnt += self.nodes[node_idx].resp_cost

            if self.nodes[node_idx].role == "Tester":
                unit_tests.extend(self.nodes[node_idx].get_unit_tests())

        loop_indices = list(range(self.agents+self.judges, self.agents*2+self.judges))
        random.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            print(1, idx)
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)), question, entry_point)
                if reached:
                    # return reply, resp_cnt, get_completions(), unit_tests
                    return self.all_tests_and_get_final_result(question, unit_tests, entry_point), resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens, unit_tests

        idx_mask = list(range(self.agents))
        idxs = list(range(self.agents+self.judges, self.agents*2+self.judges))
        for rid in range(2, self.rounds):
            if self.agents > 2:
                replies = [self.nodes[idx].get_answer() for idx in idxs]
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]
            
                tops, prompt_tokens, completion_tokens = self.activation(shuffled_replies, question, self.qtype, self.mtype)
                idx_mask = list(map(lambda x: idxs[indices[x]] % (self.agents+self.judges), tops))
                resp_cnt += self.activation_cost
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

            for idx, node_idx in enumerate(range((self.agents+self.judges)*rid-self.judges, (self.agents+self.judges)*rid)):
                print(rid-0.5, idx)
                self.nodes[node_idx].activate(question)
                total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                total_completion_tokens += self.nodes[node_idx].completion_tokens
                resp_cnt += self.nodes[node_idx].resp_cost

                if self.nodes[node_idx].role == "Tester":
                    unit_tests.extend(self.nodes[node_idx].get_unit_tests())

            loop_indices = list(range((self.agents+self.judges)*rid, (self.agents+self.judges)*(rid)+self.agents))
            random.shuffle(loop_indices)
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                if idx in idx_mask:
                    print(rid, idx)
                    self.nodes[node_idx].activate(question)
                    resp_cnt += 1
                    total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                    total_completion_tokens += self.nodes[node_idx].completion_tokens
                    idxs.append(node_idx)
                    if len(idxs) > math.floor(2/3 * len(idx_mask)):
                        reached, reply = self.check_consensus(idxs, idx_mask, question, entry_point)
                        if reached:
                            # return reply, resp_cnt, get_completions(), unit_tests
                            return self.all_tests_and_get_final_result(question, unit_tests, entry_point), resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens, unit_tests
        
        completions = get_completions()
        # return self.get_final_result(idxs, question), resp_cnt, completions, total_prompt_tokens, total_completion_tokens, unit_tests
        return self.all_tests_and_get_final_result(question, unit_tests, entry_point), resp_cnt, completions, total_prompt_tokens, total_completion_tokens, unit_tests


    def backward(self, result, question, entry_point):
        flag_last = False
        for rid in range(self.rounds-1, -1, -1):
            if not flag_last:
                if len([idx for idx in range((self.agents+self.judges)*rid, (self.agents+self.judges)*rid+self.agents) if self.nodes[idx].active]) > 0:
                    flag_last = True
                else:
                    continue

                # if rid == self.rounds-1:
                    # candidates = [idx for idx in range((self.agents+self.judges)*rid, (self.agents+self.judges)*rid+self.agents) if self.nodes[idx].active and check_function_result(self.nodes[idx].get_answer())]
                # else:
                    # candidates = [idx for idx in range((self.agents+self.judges)*rid, (self.agents+self.judges)*rid+self.agents) if self.nodes[idx].active and self.cmp_res(self.cut_def_question(self.nodes[idx].get_answer(), question), result)]
                candidates = [idx for idx in range((self.agents+self.judges)*rid, (self.agents+self.judges)*rid+self.agents) if self.nodes[idx].active and check_function_result(question + "\n" + self.cut_def_question(self.nodes[idx].get_answer(), question, entry_point))]
                ave_w = 1 / len(candidates)
                for idx in range((self.agents+self.judges)*rid, (self.agents+self.judges)*rid+self.agents):
                    if idx in candidates:
                        self.nodes[idx].importance = ave_w
                    else:
                        self.nodes[idx].importance = 0
            else:
                for idx in range((self.agents+self.judges)*rid+self.agents, (self.agents+self.judges)*(rid+1)):
                    self.nodes[idx].importance = 0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            self.nodes[idx].importance += edge.weight * edge.a2.importance
                for idx in range((self.agents+self.judges)*rid, (self.agents+self.judges)*rid+self.agents):
                    self.nodes[idx].importance = 0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            self.nodes[idx].importance += edge.weight * edge.a2.importance

        return [node.importance for node in self.nodes]
