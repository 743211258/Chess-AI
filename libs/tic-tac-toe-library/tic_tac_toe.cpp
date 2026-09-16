#include <iostream>
#include <array>
#include <stdbool.h>
#include <vector>

#include "tic_tac_toe.hpp"

constexpr std::array<std::array<int,3>,48> WINS {{
    // rows {0,1,2}
    {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0},
    // rows {3,4,5}
    {3,4,5}, {3,5,4}, {4,3,5}, {4,5,3}, {5,3,4}, {5,4,3},
    // rows {6,7,8}
    {6,7,8}, {6,8,7}, {7,6,8}, {7,8,6}, {8,6,7}, {8,7,6},
    // cols {0,3,6}
    {0,3,6}, {0,6,3}, {3,0,6}, {3,6,0}, {6,0,3}, {6,3,0},
    // cols {1,4,7}
    {1,4,7}, {1,7,4}, {4,1,7}, {4,7,1}, {7,1,4}, {7,4,1},
    // cols {2,5,8}
    {2,5,8}, {2,8,5}, {5,2,8}, {5,8,2}, {8,2,5}, {8,5,2},
    // diag {0,4,8}
    {0,4,8}, {0,8,4}, {4,0,8}, {4,8,0}, {8,0,4}, {8,4,0},
    // diag {2,4,6}
    {2,4,6}, {2,6,4}, {4,2,6}, {4,6,2}, {6,2,4}, {6,4,2}
}};

namespace Tic_Tac_Toe {
  namespace Board {
    std::ostream& operator<<(std::ostream& os, const Board& b) {
      for (int i = 0; i < BOARD_SIZE; ++i) {
	os << b.board[i];
	if (i + 1 < BOARD_SIZE) os << ' ';
      }
      os << '\n';

      for (size_t i = 0; i < b.pieces.size(); ++i) {
	os << b.pieces[i];
	if (i + 1 < b.pieces.size()) os << ' ';
      }
      os << '\n';

      os << b.player << '\n';

      os << b.turn;

      return os;
    }
  }

  namespace Move {
    std::vector<int> get_legal_move(Tic_Tac_Toe::Board::Board& board) {
      std::vector<int> result;
      for (int i = 0; i < BOARD_SIZE; ++i) {
        if (board.board[i] == Tic_Tac_Toe::Board::EMPTY) {
          result.push_back(i);
        }
      }
      return result;
    }

    bool move(Tic_Tac_Toe::Board::Board& board, int move) {
      Tic_Tac_Toe::Board::STATE& state = board.board[move];
      if (state != Tic_Tac_Toe::Board::EMPTY) {
        return false;
      } else {
        ++board.turn;
        if (board.player == Tic_Tac_Toe::Board::CROSS) {
          state = Tic_Tac_Toe::Board::CROSS;
          board.player = Tic_Tac_Toe::Board::NOUGHT;
        } else {
          state = Tic_Tac_Toe::Board::NOUGHT;
          board.player = Tic_Tac_Toe::Board::CROSS;
        }
      }
      board.pieces.push_back(move);
      if (board.pieces.size() > MAX_PIECES) {
        int index = board.pieces.front();
        board.pieces.pop_front();
        board.board[index] = Tic_Tac_Toe::Board::EMPTY;
      }
      return true;
    }
  }

  namespace Result {
    Tic_Tac_Toe::Result::STATE result_state(Tic_Tac_Toe::Board::Board& board) {
      if (board.turn >= MAX_TURNS) {
        return Tic_Tac_Toe::Result::DRAW;
      }
      if (board.pieces.size() < MAX_PIECES - 1) {
        return Tic_Tac_Toe::Result::ONGOING;
      }
      std::array<int, 3> current;
      if (board.pieces.size() == MAX_PIECES - 1) {
        current = {board.pieces[0], board.pieces[2], board.pieces[4]};
      } else {
        current = {board.pieces[1], board.pieces[3], board.pieces[5]};
      }
      
      for (auto const& line : WINS) {
        if (current == line) {
          return Tic_Tac_Toe::Result::WIN;
        }
      }
      return Tic_Tac_Toe::Result::ONGOING;
    }
  }
}
