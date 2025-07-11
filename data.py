import chess
from deep_reinforcement import MCTSplayer
from cnn import masked_policy, CNN
import torch
import json
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os


class collect_data():
    def __init__(self, model = None, model_path = None):
        if model is None:
            model = CNN(512, 10).to(device)

        # 如果模型文件存在则加载
        if os.path.exists(model_path):
            print(f"✅ Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"⚠️ No model found at {model_path}, using a new model.")

        self.model = model
        self.board = chess.Board()

    def self_play(self, player):
        states = []
        mcts_probs = []
        current_player = []
        win = []
        round = 0
        while not self.board.is_game_over():
            round += 1
            move, moves_probs = player.play(self.board)
            states.append(self.board.fen())
            mcts_probs.append(moves_probs)
            current_player.append(self.board.turn)
            self.board.push(move)
            print(self.board)
        result = self.board.result()
        print(result, self.board.outcome())
        winner = None
        if result == "1-0":
            winner = chess.WHITE
        elif result == "0-1":
            winner = chess.BLACK
        else:
            winner = None
        for current in current_player:
            if winner is None:
                win.append(0)
            elif winner == current:
                win.append(1)
            else:
                win.append(-1)
        training_data = list(zip(states, mcts_probs, win))
        serializable_data = []

        with open("training_data.jsonl", 'a') as f:
            for state, (pi_indices, pi_probs), z in training_data:
                if isinstance(pi_indices, torch.Tensor):
                    pi_indices = pi_indices.tolist()
                if isinstance(pi_probs, torch.Tensor):
                    pi_probs = pi_probs.tolist()
                else:
                    pi_probs = [float(p) for p in pi_probs]
                entry = {
                    'state_fen': state,
                    'pi_indices': pi_indices,
                    'pi_probs': pi_probs,
                    'z': int(z)
                }
                f.write(json.dumps(entry) + '\n')
            


for x in range(10):
    data = collect_data(None, model_path = "C:\Chess\latest_model.pth")   
    data.self_play(MCTSplayer(masked_policy, CNN(512, 10).to(device)))           
    print(f"iteration {x}")            

