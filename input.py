import chess.pgn
import chess
import torch

PIECES_TO_NUMBER_WHITE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

PIECES_TO_NUMBER_BLACK = {
    chess.PAWN: 6,
    chess.KNIGHT: 7,
    chess.BISHOP: 8,
    chess.ROOK: 9,
    chess.QUEEN: 10,
    chess.KING: 11,
}

class Train():
    def __init__(self, file_name, max_games):
        self.file_name = file_name
        self.max_games = max_games
    def extract_data(self):
        boards = []
        try:
            with open(self.file_name) as pgn_file:
                for x in range(self.max_games):
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    else:
                        board = game.board()
                        boards.append(board)
                        for move in game.mainline_moves():
                            board.push(move)
                            boards.append(board.copy())
        except FileNotFoundError as e:
            print("The file cannot be found.")
        except Exception as e:
            print("Unknown error occurs.")
        return boards
def board_to_tensor(board):
    tensor = torch.zeros(13, 8, 8)
    for square, piece in board.piece_map().items():
        row = square // 8
        column = square % 8
        if piece.color == chess.WHITE:
            index = PIECES_TO_NUMBER_WHITE[piece.piece_type]
        else:
            index = PIECES_TO_NUMBER_BLACK[piece.piece_type]
        tensor[index][row][column] = 1
    if board.turn:
        tensor[12].fill_(1.0)
    else:
        tensor[12].fill_(0.0)
    return tensor;


