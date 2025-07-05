# load_data.py
import torch
import json
from input import board_to_tensor
from game import move_id, move_to_index
import chess

def recover_policy(pi_indices, pi_probs, total_moves=3820):
    policy = torch.zeros(total_moves)
    for idx, prob in zip(pi_indices, pi_probs):
        if 0 <= idx < total_moves:
            policy[idx] = prob
    return policy

def load_training_data(filename="training_data.json"):
    with open(filename, 'r') as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        board = chess.Board(item["state_fen"])
        state_tensor = board_to_tensor(board)
        pi_indices = item["pi_indices"]
        pi_probs = item["pi_probs"]
        z = item["z"]
        policy_tensor = recover_policy(pi_indices, pi_probs)
        data.append((state_tensor, policy_tensor, z))

    return data
