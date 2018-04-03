import numpy as np
import torch
from torch.autograd import Variable
import torch.functional as F

class alphazero:
    def __init__(self, net, actions, evaluate, transition, turns_until_tau0=10, cuda=False):
        self.num_sims = 20
        self.net = net
        self.actions = actions
        self.evaluate = evaluate
        self.transition = transition
        self.turns_until_tau0 = turns_until_tau0
        self.memories = []
        self.curr_memories = []
        if cuda:
            self.net = self.net.cuda

        self.cuda = cuda

    def mcts(self, root_state):
        game_over = False

        root_node = {
            "state": np.array(root_state)
        }

        root_node = self.expand(root_node)

        turn = 1
        while not game_over:
            root_node = self.add_dirichlet_noise(root_node)
            root_node = self.mcts_step(root_node, turn)
            turn += 1

            reward, game_over = self.evaluate((root_node["state"]))

        for memory in self.curr_memories:
            memory["reward"] = reward
        self.memories.extend(self.curr_memories)

        del self.curr_memories

    def mcts_step(self, root_node, turn):
        for _ in range(self.num_sims):
            leaf_node = self.select(root_node)

            expanded_node, value = self.expand(leaf_node)

            root_node = self.backup(expanded_node, value)

        action, search_probas = self.choose_real_move(root_node, turn)

        real_next_node = root_node["children"][action]
        real_next_node["state"] = self.transition(root_node["state"], action)
        del real_next_node["parent"]

        memory = {
            "search_probas": search_probas, "state": np.array(root_node["state"])
        }

        self.curr_memories.extend([memory])

        return real_next_node

    def choose_real_move(self, node, turn):
        child_visits_probas = node["child_visits"]/node["child_visits"].sum()

        if turn > self.turns_until_tau0:
            action = np.argmax(node["child_visits"])
        else:
            action = np.random.choice(
                node["child_visits"], p=child_visits_probas)

        return action, child_visits_probas

    def U(self, P, N): return P/(1 + N)

    def Q(self, W, N): return W/N

    def uct_choice(
        self, curr_node): return curr_node["children"][curr_node["max_uct"]]

    def select(self, root_node):
        curr_node = root_node

        while curr_node.children is not None:
            curr_node = self.uct_choice(curr_node)

        return curr_node

    def expand(self, leaf_node):
        policy, value = self.net(leaf_node["state"])

        leaf_node["children"] = []
        leaf_node["max_uct"] = {}
        leaf_node["max_N"] = 0

        max_U = 0
        max_U_idx = None

        for i, p in enumerate(policy):
            U = self.U(P=p, N=0)

            child = {
                "N": 0,
                "W": 0,
                "Q": 0,
                "U": U,
                "P": p,
                "children": None,
                "parent": leaf_node,
                "idx": i,
                "state": None
            }

            leaf_node["children"].extend([child])

            if U > max_U:
                max_U = U
                max_U_idx = i

        leaf_node["max_uct"] = {
            "score": max_U, "idx": max_U_idx
        }

        return leaf_node, value

    def backup(self, expanded_node, value):
        node = expanded_node
        while node["parent"] is not None:
            node = self.update_node(node, value)
            node = node["parent"]

        return node

    def update_node(self, node, value):
        node["N"] += 1
        node["W"] += value
        node["Q"] = self.Q(W=node["W"], N=node["N"])
        node["U"] = self.U(P=node["P"], N=node["N"])
        UCT = node["Q"] + node["U"]
        if UCT > node["parent"]["max_uct"]["score"]:
            node["parent"]["max_uct"]["score"] = UCT
            node["parent"]["max_uct"]["idx"] = node["idx"]

        return node

    def add_dirichlet_noise(self, node, alpha=.8, epsilon=.2):
        nu = np.random.dirichlet([alpha] * len(self.actions))*epsilon

        for i, child in enumerate(node["children"]):
            child["P"] = child["P"]*(1-epsilon) + nu[i]

        return node

    def train(self, training_loops=10, batch_size=128):
        for _ in range(training_loops):
            minibatch = np.random.sample(
                self.memories, min(batch_size, len(self.memories)))

            self.train_minibatch(minibatch)

    def V(self, tensor):
        var = Variable(tensor)
        if self.cuda:
            var = var.cuda()

        return var

    def train_minibatch(self, minibatch):
        sellf.net_optim.zero_grad()

        rewards = torch.from_numpy(
            np.zeros((minibatch.shape[0], 1)).astype("float32"))
        search_probas = torch.from_numpy(np.zeros((len(minibatch),
                                                   minibatch[0]["search_probas"].shape)).astype("float32"))
        states = torch.from_numpy(np.zeros((len(minibatch),
                                            minibatch[0]["state"].shape)).astype("float32"))

        for i, memory in enumerate(minibatch):
            rewards[i] = memory["reward"]
            search_probas[i] = memory["search_probas"]
            states[i] = memory["state"]

        rewards = self.V(rewards)
        search_probas = self.V(search_probas)
        states = self.V(states)

        policies, values = self.net(states)

        value_loss = F.mse_loss(values, rewards)

        policy_loss = 0
        for search_p, pi in zip(search_probas, policies):
            search_p.unsqueeze(0)
            pi.unsqueeze(-1)
            policy_loss += search_p.mm(pi)

        total_loss = value_loss + policy_loss
        total_loss.backward()

        self.net_optim.step()
