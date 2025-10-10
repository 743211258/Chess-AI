import chess
import string

move_id = []
letters = "abcdefgh"
promotion = ["q", "r", "n", "b"]

# all potential special legal moves for white pawns
for letter in letters:
    for type in promotion:
        move_id.append(letter + "7" + letter + "8" + type)
        move_id.append(letter + "2" + letter + "1" + type)
        if letter != "a":
            move_id.append(letter + "7" + chr(ord(letter) - 1) + "8" + type)
            move_id.append(letter + "2" + chr(ord(letter) - 1) + "1" + type)
        if letter != "h":
            move_id.append(letter + "7" + chr(ord(letter) + 1) + "8" + type)
            move_id.append(letter + "2" + chr(ord(letter) + 1) + "1" + type)
print(len(move_id))

# all potential legal moves horizontally and vertically
moves = [(-1, 0), (1, 0),
             (0, -1), (0, 1),
             (-1, 1), (1, 1),
             (-1, -1), (1, -1)]
for number in range(1, 9):
    for letter in letters:
        for x, y in moves:
            column = ord(letter)
            row = number
            while ord('a') <= column + x <= ord('h') and 1 <= row + y <= 8:
                column = column + x
                row = row + y
                move_id.append(letter + str(number) + chr(column) + str(row))

#all potential legal moves for white/black horses:
knight_moves = [(-2, 1), (2, 1),
               (-2, -1), (2, -1),
               (1, -2), (-1, -2),
               (1, 2), (-1, 2)]
for number in range(1, 9):
    for letter in letters:
        for x, y in knight_moves:
            new_column = ord(letter) + x
            new_row = number + y
            if ord('a') <= new_column <= ord('h') and 1 <= new_row <= 8:
                move_id.append(letter + str(number) + chr(new_column) + str(new_row))

uci_to_index = {uci: idx for idx, uci in enumerate(move_id)}

def move_to_index(move: chess.Move) -> int:
    uci_str = move.uci()
    return uci_to_index.get(uci_str, -1) 


