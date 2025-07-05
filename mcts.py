import chess
import chess.engine
import random
import math
from evaluation import evaluation, analyze, eval_to_winrate
import time

class Node:
    def __init__(self, state, parent, childs, prior):
        self.state = state # type board
        self.parent = parent # type node
        self.childs = childs # type move->node dict
        self.Q = 0
        self.visit = 0 # type float/double
        self.N = 0
        self.win = 0 # type int
        self.P = prior

    
    # fully expanded: all potential legal moves are examined
    # not fully expanded: at least one legal moves haven't been examined
    def is_fully_expanded(self,):
        '''max_expansion = min(len(list(self.state.legal_moves)), int(math.sqrt(self.visit)) + 1)
        return len(self.childs) >= max_expansion'''
        return len(self.childs) == len(list(self.state.legal_moves))

    # find the best child with the highest uct value
    def best_uct(self,):
        #c = math.sqrt(2)
        c = 5
        return max(self.childs.values(), key = lambda child: child.win / child.visit + c * math.sqrt(math.log(self.visit) / child.visit))

# traverse the tree till encountered an unvisited node
def selection(root):
    node = root
    while node.is_fully_expanded() and node.childs:
        node = node.best_uct()
    return node

# find a random unexamined possible move
def expansion(node):
    unvisited_moves = []
    for move in node.state.legal_moves:
        if move not in node.childs.keys():
            unvisited_moves.append(move)
    if len(unvisited_moves) == 0:
        return node
    move = random.choice(unvisited_moves)
    new_state = node.state.copy()
    new_state.push(move)
    child_node = Node(new_state, parent = node, childs = {})
    node.childs[move] = child_node
    return child_node

'''def rollout_policy(board):
    return random.choice(list(board.legal_moves))

def rollout(node, root):
    board = node.state.copy()
    while not (board.is_checkmate() or board.is_stalemate()):
        move = rollout_policy(board)
        board.push(move)
    if board.is_checkmate():
        if root.state.turn == chess.WHITE:
            return 1 # white win
        else:
            return 0 # black win
    else:
        return 0.5 # stalemate'''

def rollout(node, root, engine):
    if node.state.is_checkmate() and node.state.turn != root.state.turn:
        return 1
    elif node.state.is_checkmate() and node.state.turn == root.state.turn:
        return 0
    elif node.state.is_stalemate():
        return 0.5
    eval = evaluation(node.state, engine)
    return eval_to_winrate(eval, root.state.turn)

'''def rollout(node, root):
    board = node.state.copy()
    
    while not board.is_game_over():
        move = random.choice(list(board.legal_moves))
        board.push(move)

    if board.is_checkmate():
        return 1.0 if board.turn != root.state.turn else 0.0
    else:
        return 0.5'''

def back_progagation(node, result):
    if node == None:
        return
    node.visit += 1
    node.win += result
    back_progagation(node.parent, result)

def best_child(root):
    return max(root.childs.values(), key = lambda node: node.visit)


def main(root, engine = None, iteration = 1000):
    for x in range(iteration):
        if x % 100 == 0:
            print("%d boards analyzed." %(x))
        not_fully_expanded_node = selection(root)
        leaf = expansion(not_fully_expanded_node)
        #print(leaf.state)
        simulation = rollout(leaf, root, engine)
        back_progagation(leaf, simulation)
    return best_child(root)

root = Node(chess.Board("1Bb3BN/R2Pk2r/1Q5B/4q2R/2bN4/4Q1BK/1p6/1bq1R1rb w - - 0 1"), None, {})
print(root.state.legal_moves)
engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\ericz\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
move = main(root, engine)
engine.quit()

print(move.state)
print(move.win, move.visit)

for move, child in root.childs.items():
    print(move, child.win, child.visit)