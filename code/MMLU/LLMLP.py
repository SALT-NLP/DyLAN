import os
import math
import random
from LLM_Neuron import LLMNeuron, LLMEdge, listwise_ranker_2
from utils import parse_single_choice, most_frequent, is_equiv, extract_math_answer



ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1} # TODO: only 0 is implemented

class LLMLP:
    
    def __init__(self, default_model_name, agents=4, agent_roles=[],
                 rounds=2, activation="listwise", qtype="single_choice", mtype="gpt-3.5-turbo"):
        self.default_model_name = default_model_name
        self.agents = agents
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype
        
        assert len(agent_roles) == agents and agents > 0
        self.agent_roles = agent_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "math_exp":
            self.cmp_res = is_equiv
            self.ans_parser = extract_math_answer

        # cache the last question so backward() can access it for the judge
        self._last_question = None

        self.init_nn(self.activation, self.agent_roles)

    def init_nn(self, activation, agent_roles):
        self.nodes, self.edges = [], []
        for idx in range(self.agents):
            self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
        
        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
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

    def check_consensus(self, idxs, idx_mask):
        # check consensus based on idxs (range) and idx_mask (actual members, might exceed the range)
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        consensus_answer, ca_cnt = most_frequent(candidates, self.cmp_res)
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            print("Consensus answer: {}".format(consensus_answer))
            return True, consensus_answer
        return False, None

    def set_allnodes_deactivated(self):
        for node in self.nodes:
            node.deactivate()

    def forward(self, question):
        def get_completions():
            # get completions
            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    if self.nodes[idx].active:
                        completions[idx % self.agents].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % self.agents].append(None)
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()
        # Remember the raw question text for the backward (tie-break) step.
        assert self.rounds > 2
        self._last_question = question
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
        
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        loop_indices = list(range(self.agents, self.agents*2))
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
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        idx_mask = list(range(self.agents))
        idxs = list(range(self.agents, self.agents*2))
        for rid in range(2, self.rounds):
            # TODO: compatible with 1/2 agents
            if self.agents > 3:
                replies = [self.nodes[idx].get_reply() for idx in idxs]
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]
            
                tops, prompt_tokens, completion_tokens = self.activation(shuffled_replies, question, self.qtype, self.mtype)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                idx_mask = list(map(lambda x: idxs[indices[x]] % self.agents, tops))
                resp_cnt += self.activation_cost

            loop_indices = list(range(self.agents*rid, self.agents*(rid+1)))
            random.shuffle(loop_indices)
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                # TODO: report bug # if idx in idx_mask:
                if node_idx % self.agents in idx_mask:
                    print(rid, idx)
                    self.nodes[node_idx].activate(question)
                    resp_cnt += 1
                    total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                    total_completion_tokens += self.nodes[node_idx].completion_tokens
                    idxs.append(node_idx)
                    if len(idxs) > math.floor(2/3 * len(idx_mask)):
                        reached, reply = self.check_consensus(idxs, idx_mask)
                        if reached:
                            return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        completions = get_completions()
        return most_frequent([self.nodes[idx].get_answer() for idx in idxs], self.cmp_res)[0], resp_cnt, completions, total_prompt_tokens, total_completion_tokens

    def backward(self, result, question=None):
        """
        Compute Agent Importance via backward aggregation.
        If AIP_JUDGE_WEIGHTS=1 and there are >=2 last-layer survivors that match `result`,
        ask an LLM judge for a JSON weight vector instead of uniform splitting.
        """
        use_judge = os.getenv("AIP_JUDGE_WEIGHTS", "0") == "1"
        flag_last = False

        for rid in range(self.rounds - 1, -1, -1):
            layer_indices = list(range(self.agents * rid, self.agents * (rid + 1)))
            actives = [idx for idx in layer_indices if self.nodes[idx].active]

            if not flag_last:
                # Find the last active layer
                if len(actives) == 0:
                    continue
                flag_last = True

                # Among actives, only agents whose answer equals final `result` receive positive credit
                correct_idxs = [idx for idx in actives if self.cmp_res(self.nodes[idx].get_answer(), result)]

                if len(correct_idxs) == 0:
                    # No one matched the final result -> all zero (degenerate case)
                    for idx in layer_indices:
                        self.nodes[idx].importance = 0.0
                    continue

                if use_judge and len(correct_idxs) >= 2:
                    # Gather full replies for reasoning quality judging
                    replies = [self.nodes[idx].get_reply() for idx in correct_idxs]
                    try:
                        from utils import judge_importance_weights
                        weights, _, _ = judge_importance_weights(
                            replies,
                            question if question is not None else "",
                            self.qtype,
                            self.mtype
                        )
                    except Exception:
                        # Robust fallback
                        weights = [1.0 / len(correct_idxs)] * len(correct_idxs)

                    # Assign judged weights to correct survivors; others get 0
                    for idx in layer_indices:
                        self.nodes[idx].importance = 0.0
                    for w, idx in zip(weights, correct_idxs):
                        self.nodes[idx].importance = float(w)
                else:
                    # Default: split equally among correct survivors
                    ave_w = 1.0 / float(len(correct_idxs))
                    for idx in layer_indices:
                        if idx in correct_idxs:
                            self.nodes[idx].importance = ave_w
                        else:
                            self.nodes[idx].importance = 0.0

            else:
                # Standard backward aggregation through peer rating edges
                for idx in layer_indices:
                    self.nodes[idx].importance = 0.0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            # edge.weight is peer rating (1-5); here we do a simple weighted sum
                            self.nodes[idx].importance += edge.weight * edge.a2.importance

        return [node.importance for node in self.nodes]


