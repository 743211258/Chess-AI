import chess
import numpy as np
import copy
import torch
import torch.nn.functional as F
from cnn import CNN, masked_policy
from game import move_to_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Node:
    def __init__(self, parent, prior):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.Q = 0
        self.U = 0
        self.prior = prior

    def puct(self, c):
        self.U = (c * self.prior *
                   np.sqrt(self.parent.visits) / (1 + self.visits))
        return self.Q + self.U
    
    def select(self, c):
        return max(self.children.items(),
                   key=lambda node: node[1].puct(c))

    def expand(self, prior):
        for action, prob in prior:
            if action not in self.children:
                self.children[action] = Node(self, prob)
    
    def update(self, rollout):
        self.visits += 1
        self.Q += 1.0 * (rollout - self.Q) / self.visits

    def back_propogate(self, rollout):
        if self.parent:
            self.parent.back_propogate(-rollout)
        self.update(rollout)
    
    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

    def count_nodes(self, node):
        total = 1
        for child in node.children.values():
            total += self.count_nodes(child)
        return total


class MCTS():
    def __init__(self, policy, network, c = 5, iteration = 100):
        self.root = Node(None, 1.0)
        self.policy = policy
        self.network = network
        self.c = c
        self.iteration = iteration

    def algorithm(self, board):
        node = self.root
        while True:
            if node.is_leaf():
                break
            move, node = node.select(self.c)
            board.push(move)
        policy, value = self.policy(board, self.network)
        if not board.is_game_over():
            node.expand(policy)
        else:
            result = board.result()
            if result == "1/2-1/2":
                value = 0.0
            elif result == "1-0":
                value = 1.0 if board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                value = 1.0 if board.turn == chess.BLACK else -1.0
        node.back_propogate(-value)

    def reinforced_policy_output(self, board, temp = 0.1):
        for x in range(self.iteration):
            board_copy = board.copy()
            self.algorithm(board_copy)
            if x % 100 == 0:
                print(f"已搜索{x}个局面")
                print(f"当前树大小（节点数）: {self.root.count_nodes(self.root)}")

        act_visits= [(act, node.visits)
            for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        visits_tensor = torch.tensor(visits, dtype=torch.float32)
        act_probs = F.softmax(1.0 / temp * np.log(visits_tensor + 1e-10))
        return acts, act_probs
    
    def update_tree(self, selected_move):
        if selected_move in self.root.children:
            self.root = self.root.children[selected_move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)

class MCTSplayer():
    def __init__(self, policy, network, c = 5, iteration = 400, is_selfplay = True):
        self.mcts = MCTS(policy, network, c, iteration)
        self.is_selfplay = is_selfplay
        
    def play(self, board, temp = 1e-3, is_return_prob = True):
        #move_probs = np.zeros(1968)
        acts, acts_probs = self.mcts.reinforced_policy_output(board, temp)
        indices = [move_to_index(move) for move in acts]
        valid = [(idx, prob) for idx, prob in zip(indices, acts_probs) if idx != -1]
        #indices = [move_to_index(move) for move in acts]
        #valid = [(idx, prob) for idx, prob in zip(indices, acts_probs) if idx != -1]
        #for idx, prob in valid:
        #    move_probs[idx] = prob
        if self.is_selfplay == True:
            p = p = 0.75 * acts_probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(acts_probs)))
            p /= p.sum()
            move = np.random.choice(acts, p = p)
            self.mcts.update_tree(move)
        if is_return_prob:
            sparse_indices = [idx for idx, _ in valid]
            sparse_probs = [prob for _, prob in valid]
            return move, (sparse_indices, sparse_probs)
        else:
            return move
        
