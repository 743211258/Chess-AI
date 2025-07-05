import chess
import chess.engine
from input import board_to_tensor
from cnn import CNN
import torch
import torch.optim as optim

def evaluation(board, engine):
    limit = chess.engine.Limit(0.01)
    best_score = None
    #engine.configure({"Use NNUE": True})
    with engine.analysis(board, limit) as analysis:
        for info in analysis:
            if "score" in info:
                best_score = info["score"].white().score(mate_score=10000)
    return best_score


def analyze(board):
    model = CNN(kernel_size=3, num_kernel_in_first_layer=13, num_kernel_in_second_layer=64, padding=1)
    weights = torch.load('model_checkpoint.pth', weights_only=True)
    model.load_state_dict(weights['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(weights['optimizer_state_dict'])
    model.eval()
    tensor_board = board_to_tensor(board).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor_board)
        score = output.item()
    return score

def eval_to_winrate(score_cp, current_player, k=0.0008):
    score_tensor = torch.tensor(score_cp, dtype=torch.float32)
    white_winrate = torch.sigmoid(k * score_tensor).item()
    #white_winrate = abs(score_cp/10000)
    if current_player:
        return white_winrate
    else:
        return 1 - white_winrate
board = chess.Board("1Bb3BN/R2Pk2r/1Q6/4q1BR/2bN4/4Q1BK/1p6/1bq1R1rb b - - 1 1")
score = analyze(board)
print(score)
print(eval_to_winrate(score, chess.WHITE))
engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\ericz\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
#score = evaluation(board, engine)
score = 9999
print(score)
current_board = chess.Board("1Bb3BN/R2Pk2r/1Q6/4q1BR/2bN4/4Q1BK/1p6/1bq1R1rb b - - 1 1")
print(eval_to_winrate(9998, chess.WHITE))
print(current_board.legal_moves)
engine.quit()