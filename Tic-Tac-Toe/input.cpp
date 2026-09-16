#include "pch.hpp"

#include "Tic_Tac_Toe.hpp"
#include "input.hpp"

torch::Tensor board_to_tensor(const Tic_Tac_Toe::Board::Board& board, Tic_Tac_Toe::Board::STATE player) {
  torch::Tensor tensor = torch::zeros({2, 3, 3});

  for (int i = 0; i < 9; ++i) {
    int row = i / 3;
    int column = i % 3;
    if (player == board.board[i]) {
      tensor[0][row][column] = 1.0f;
    } else if (board.board[i] != Tic_Tac_Toe::Board::EMPTY) {
      tensor[1][row][column] = 1.0f;
    }
  }
  return tensor;
}

board_history::board_history(size_t max_history) {
  this->max_history = max_history;
}

void board_history::push(Tic_Tac_Toe::Board::Board board) {
  if (history.size() == max_history) {
    this->history.pop_front();
  }
  this->history.push_back(board);
}

std::deque<Tic_Tac_Toe::Board::Board> board_history::get_history() {
  return this->history;
}

torch::Tensor board_history::to_tensor() const {
  torch::Tensor tensor = torch::zeros({max_history * 4, 3, 3});
  int index = 0;
  for (const Tic_Tac_Toe::Board::Board& board : history) {
    torch::Tensor slice = tensor.index({torch::indexing::Slice(index * 4, index * 4 + 2)});
    slice.copy_(board_to_tensor(board, this->history.back().player));
    if (board.pieces.size() == 5) {
      int remove_index = board.pieces.front();
      int row = remove_index / 3;
      int column = remove_index % 3;
      tensor[index * 4 + 3][row][column].fill_(1.0f);
    } else if (board.pieces.size() == 6) {
      int remove_index = board.pieces.front();
      int row = remove_index / 3;
      int column = remove_index % 3;
      tensor[index * 4 + 2][row][column].fill_(1.0f);
      
      remove_index = board.pieces[1];
      row = remove_index / 3;
      column = remove_index % 3;
      tensor[index * 4 + 3][row][column].fill_(1.0f);
    }
    index++;
  }
  return tensor;
}

void print_tensor(const torch::Tensor& t) {
    auto sizes = t.sizes();
    for (int c = 0; c < sizes[0]; ++c) {
        std::cout << "Channel " << c << ":" << std::endl;
        for (int i = 0; i < sizes[1]; ++i) {
            for (int j = 0; j < sizes[2]; ++j) {
                std::cout << t[c][i][j].item<float>() << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
