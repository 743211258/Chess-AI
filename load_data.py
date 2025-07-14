import torch
import json
import random
from input import board_to_tensor
from game import move_id, move_to_index
import chess

def recover_policy(pi_indices, pi_probs, total_moves=3820):
    policy = torch.zeros(total_moves)
    for idx, prob in zip(pi_indices, pi_probs):
        if 0 <= idx < total_moves:
            policy[idx] = prob
    return policy

def load_random_sample(filename="training_data.jsonl"):
    with open(filename, 'r') as f:
        lines = f.readlines()
    line = random.choice(lines)
    item = json.loads(line)

    board = chess.Board(item["state_fen"])
    state_tensor = board_to_tensor(board)
    pi_indices = item["pi_indices"]
    pi_probs = item["pi_probs"]
    z = item["z"]
    policy_tensor = recover_policy(pi_indices, pi_probs)
    return state_tensor, policy_tensor, z

def load_random_batch(filename="training_data.jsonl", batch_size=64):
    with open(filename, 'r') as f:
        lines = f.readlines()
    samples = random.sample(lines, batch_size)

    state_list = []
    policy_list = []
    value_list = []

    for line in samples:
        item = json.loads(line)
        board = chess.Board(item["state_fen"])
        state_tensor = board_to_tensor(board)                       # [13, 8, 8]
        pi_tensor = recover_policy(item["pi_indices"], item["pi_probs"])  # [3820]
        z = float(item["z"])

        state_list.append(state_tensor)
        policy_list.append(pi_tensor)
        value_list.append(z)

    state_batch = torch.stack(state_list)                           # [B, 13, 8, 8]
    policy_batch = torch.stack(policy_list)                         # [B, 3820]
    value_batch = torch.tensor(value_list, dtype=torch.float32)     # [B]

    return state_batch, policy_batch, value_batch


