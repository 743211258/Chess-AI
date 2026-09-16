#pragma once

#include <deque>

#include "tic_tac_toe.hpp"
#include "torch/torch.h"

torch::Tensor board_to_tensor(const Tic_Tac_Toe::Board::Board& board);

class board_history {
  private:
    int max_history;
    std::deque<Tic_Tac_Toe::Board::Board> history;
  public:
    board_history(size_t max_history);
    void push(Tic_Tac_Toe::Board::Board board);
    std::deque<Tic_Tac_Toe::Board::Board> get_history();
    torch::Tensor to_tensor() const;

};
