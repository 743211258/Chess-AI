#pragma once

#define BOARD_SIZE 9
#define MAX_PIECES 6
#define MAX_TURNS 30

#include <deque>
#include <iostream>
#include <stdbool.h>
#include <vector>

namespace Tic_Tac_Toe {
  namespace Board {
    enum STATE {
      NOUGHT,
      CROSS,
      EMPTY
    };
    struct Board {
      Tic_Tac_Toe::Board::STATE board[BOARD_SIZE] = {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY};
      std::deque<int> pieces;
      Tic_Tac_Toe::Board::STATE player = CROSS;
      int turn = 0;
    };

    std::ostream& operator<<(std::ostream& os, const Tic_Tac_Toe::Board::Board& b);
  }

  namespace Move {
    std::vector<int> get_legal_move(Tic_Tac_Toe::Board::Board& board);
    bool move(Tic_Tac_Toe::Board::Board& board, int move);

  }

  namespace Result {
    enum STATE {
      ONGOING,
      WIN,
      LOSS,
      DRAW
    };
    Tic_Tac_Toe::Result::STATE result_state(Tic_Tac_Toe::Board::Board& board);
  }
}
