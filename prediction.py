'''import chess
from evaluation import analyze
board = chess.Board("8/2k5/1p5R/3B4/3P4/1P6/P1P4P/R3K3 b Q - 0 30")
moves = board.legal_moves
if not board.is_game_over():
    max_score = -10000
    min_score = 19000
    best_move = None
    for move in moves:
        board.push(move)
        score = analyze(board)
        board.pop()
        if board.turn:
            if score > max_score:
                max_score = score
                best_move = move
        else:
            if score < min_score:
                min_score = score
                best_move = move
    board.push(best_move)
    print(board)
else:
    print("checkmate")'''

import chess
from evaluation import analyze

def minimax(board, depth, maximizing_player, alpha, beta):
    if depth == 0 or board.is_game_over():
        return analyze(board), None

    best_move = None
    if maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, False, alpha, beta)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, True, alpha, beta)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

# board to be analyzed
board = chess.Board("rn3b1r/1pkq1p1p/3p3n/6p1/1p6/4PQ2/PPPP1PPP/R1B2RK1 w - - 0 14")

if not board.is_game_over():
    _, best_move = minimax(board, 3, board.turn, -float('inf'), float('inf'))
    if best_move is not None:
        board.push(best_move)
        print(board)
    else:
        print("No best move found.")
else:
    print("checkmate")
