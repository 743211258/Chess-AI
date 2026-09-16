#include "pch.hpp"

#include "tic_tac_toe.hpp"
#include "cnn.hpp"
#include "input.hpp"

#include "mcts.cpp"


class collect_data {
  private:
    std::unique_ptr<neural_network> network_pointer;
    board_history boards;
  public:
    collect_data(int in_channels, int out_channels, int nums_res_blocks, int max_history) :
      network_pointer(std::make_unique<neural_network>(in_channels, out_channels, nums_res_blocks)),
      boards(max_history)
    {}
    bool self_play(MCTSplayer& player, Tic_Tac_Toe::Board::Board& board, std::string& filename) {
      boards.push(board);
      std::vector<std::vector<Tic_Tac_Toe::Board::Board>> history;
      std::vector<std::vector<move_probability>> vector_of_move_probs;
      std::string win;
      bool is_nought_win = false;
      int turns = 0;
      while (Tic_Tac_Toe::Result::result_state(boards.get_history().back()) == Tic_Tac_Toe::Result::ONGOING) {
        std::pair<int, std::vector<move_probability>> output;
        if (turns < 4) {
          output = player.play(boards, 0.5f);
        } else {
          output = player.play(boards, 1e-7);
        }
        std::deque<Tic_Tac_Toe::Board::Board> board_queue = this->boards.get_history();
        std::vector<Tic_Tac_Toe::Board::Board> temp;
        for (Tic_Tac_Toe::Board::Board element : board_queue) {
          temp.push_back(element);
        }
	history.push_back(temp);
        vector_of_move_probs.push_back(output.second);
	Tic_Tac_Toe::Board::Board copy_of_board = boards.get_history().back();
	Tic_Tac_Toe::Move::move(copy_of_board, output.first);
        boards.push(copy_of_board);
        turns++;
      }
      Tic_Tac_Toe::Board::Board endgame = boards.get_history().back();
      Tic_Tac_Toe::Result::STATE result = Tic_Tac_Toe::Result::result_state(endgame);
      if (result == Tic_Tac_Toe::Result::DRAW) {
        win = "1/2-1/2";
      } else {
        if (endgame.player == Tic_Tac_Toe::Board::CROSS) {
          win = "0-1";
	  is_nought_win = true;
        } else {
          win = "1-0";
        }
      }
      std::ofstream file(filename, std::ios::out | std::ios::app);
      for (int i = 0; i < history.size(); ++i) {
	file << history[i].size() << std::endl;
        for (int j = 0; j < history[i].size(); ++j) {
          file << history[i][j] << std::endl;
        }
        for (int j = 0; j < vector_of_move_probs[i].size(); ++j) {
          file << vector_of_move_probs[i][j].move << " ";
          file << vector_of_move_probs[i][j].prob << std::endl;
        }
        file << win << std::endl;
      }
      file.close();
      return is_nought_win;
    }
};

std::pair<std::vector<move_probability>, float> masked_policy_wrapper(board_history& board, neural_network& net) {
    return masked_output(board, net);
}

int main(int argc, char *argv[]) {
  torch::manual_seed(54384739);
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  std::string filepath = argv[1];
  int rounds = std::stoi(argv[2]);
  int iterations = std::stoi(argv[3]);

  int in_channels = 12;
  int out_channels = 32;
  int num_res_blocks = 4;

  float c = 1.4f;

  float temperature = 1.0f;
  auto net = std::make_shared<neural_network>(in_channels, out_channels, num_res_blocks);
  std::ifstream file("model.pt");
  if (file.is_open()) {
    torch::load(net, "model.pt");
    std::cout << "Loading existing model!!" << std::endl;
  }
  net->eval();
  for (int i = 0; i < rounds; ++i) {
    Tic_Tac_Toe::Board::Board board;
    /*board.board[1] = Tic_Tac_Toe::Board::STATE::CROSS;
    board.board[6] = Tic_Tac_Toe::Board::STATE::NOUGHT;
    board.pieces.push_back(1);
    board.pieces.push_back(6);*/
    MCTSplayer player = MCTSplayer(masked_policy_wrapper, net.get(), c, iterations);
    collect_data collect(in_channels, out_channels, num_res_blocks, 3);
    collect.self_play(player, board, filepath);
  }
  return 0;
}
