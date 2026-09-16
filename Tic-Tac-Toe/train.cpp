#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "tic_tac_toe.hpp"
#include "cnn.hpp"
#include "input.hpp"

static const std::vector<std::vector<int>> SYMMETRY_MAPS {
  {0, 1, 2, 3, 4, 5, 6, 7, 8}, // 0: 原始 (Identity)
  {2, 5, 8, 1, 4, 7, 0, 3, 6}, // 1: 逆时针 90°
  {8, 7, 6, 5, 4, 3, 2, 1, 0}, // 2: 逆时针 180°
  {6, 3, 0, 7, 4, 1, 8, 5, 2}, // 3: 逆时针 270°
  {2, 1, 0, 5, 4, 3, 8, 7, 6}, // 4: 水平镜像
  {6, 7, 8, 3, 4, 5, 0, 1, 2}, // 5: 垂直镜像
  {0, 3, 6, 1, 4, 7, 2, 5, 8}, // 6: 主对角线镜像
  {8, 5, 2, 7, 4, 1, 6, 3, 0}  // 7: 副对角线镜像
};

struct training_tuple {
  std::vector<Tic_Tac_Toe::Board::Board> boards;
  std::vector<move_probability> move_probs;
  int result;
};

struct batch {
  torch::Tensor states;
  torch::Tensor policy;
  torch::Tensor value;
};

void merge_files(const std::vector<std::string>& input_files, const std::string& output_file) {
  std::ofstream output(output_file, std::ios::binary);
  for (const std::string file : input_files) {
    std::ifstream input(file, std::ios::binary);
    if (!input.is_open()) {
      std::cout << "Cannot load " << file << std::endl;
    } else {
      std::cout << "Can load " << file << std::endl;
    }
    output << input.rdbuf();
    input.close();
  }
  output.close();
}

std::vector<training_tuple> read_file(std::string filename) {
  std::ifstream file(filename);
  std::vector<training_tuple> outcome;
  if (!file.is_open()) {
    perror("Unable to read the file");
    return outcome;
  }
  std::string line;
  std::stringstream ss;
  int num;
  int loop;
  while (file >> loop) {
    file >> std::ws;
    training_tuple tuple;
    for (int i = 0; i < loop; ++i) {
      Tic_Tac_Toe::Board::Board board;
      if (!std::getline(file, line)) {
	std::cerr << "Error reading input." << std::endl;
	return outcome;
      }
      ss.clear();
      ss.str(line);
      int index = 0;
      while (ss >> num) {
	board.board[index++] = static_cast<Tic_Tac_Toe::Board::STATE>(num);
      }

      if (!std::getline(file, line)) {
	std::cerr << "Error reading input." << std::endl;
	return outcome;
      }
      ss.clear();
      ss.str(line);
      while (ss >> num) {
	board.pieces.push_back(num);
      }

      if (!std::getline(file, line)) {
	std::cerr << "Error reading input." << std::endl;
	return outcome;
      }
      ss.clear();
      ss.str(line);
      while (ss >> num) {
	board.player = static_cast<Tic_Tac_Toe::Board::STATE>(num);
      }

      if (!std::getline(file, line)) {
	std::cerr << "Error reading input." << std::endl;
	return outcome;
      }
      ss.clear();
      ss.str(line);
      while (ss >> num) {
	board.turn = num;
      }
      tuple.boards.push_back(board);
    }
    while (std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }
      if (line == "1-0") {
        Tic_Tac_Toe::Board::Board board = tuple.boards[tuple.boards.size() - 1];
	if (board.player) {
          tuple.result = 1;
	} else {
          tuple.result = -1;
	}
	break;
      } else if (line == "1/2-1/2") {
        tuple.result = 0;
	break;
      } else if (line == "0-1") {
        Tic_Tac_Toe::Board::Board board = tuple.boards[tuple.boards.size() - 1];
        if (board.player) {
          tuple.result = -1;
        } else {
          tuple.result = 1;
        }
	break;
      } else {
        std::istringstream iss(line);
        move_probability mp{};
        iss >> mp.move >> mp.prob;
        tuple.move_probs.push_back(mp);
      }
    }
    outcome.push_back(tuple);
  }
  return outcome;
}

batch create_batch( std::vector<training_tuple>& dataset, std::vector<int>& indices) {
  std::vector<torch::Tensor> state_list;
  std::vector<torch::Tensor> policy_list;
  std::vector<torch::Tensor> value_list;

  for (int index : indices) {
    training_tuple& one = dataset[index];
    board_history boards(3);
    for (Tic_Tac_Toe::Board::Board board : one.boards) {
      boards.push(board);
    }
    torch::Tensor state = boards.to_tensor();
    state_list.push_back(state);
    std::vector<int> id;
    std::vector<float> prob;
    for (move_probability mp : one.move_probs) {
      id.push_back(mp.move);
      prob.push_back(mp.prob);
    }
    torch::Tensor policy = torch::zeros({9}, torch::kFloat32);
    for (int i = 0; i < id.size(); ++i) {
      policy[id[i]] = prob[i];
    }
    policy_list.push_back(policy);
    torch::Tensor value = torch::tensor({one.result}, torch::kFloat32);
    value_list.push_back(value);
  }
  torch::Tensor states = torch::stack(state_list);
  torch::Tensor policies = torch::stack(policy_list);
  torch::Tensor values = torch::stack(value_list);
  return batch{states, policies, values};
}

batch create_augmented_batch(std::vector<training_tuple>& dataset, std::vector<int>& indices) {
    std::vector<torch::Tensor> state_list;
    std::vector<torch::Tensor> policy_list;
    std::vector<torch::Tensor> value_list;

    for (int index : indices) {
        training_tuple& one = dataset[index];

        // 遍历你定义的 8 种对称变换
        for (int s = 0; s < 8; ++s) {
            // --- 1. 处理 State (必须使用 reverse_push 保持时序一致性) ---
            board_history boards(3);
            // 注意：如果 one.boards 是按时间顺序排的，需要按原序 push 入 boards 对象
            for (const auto& raw_board : one.boards) {
                Tic_Tac_Toe::Board::Board transformed;
                transformed.player = raw_board.player;

                // 初始化为空，避免垃圾数据
                for(int i=0; i<9; ++i) transformed.board[i] = Tic_Tac_Toe::Board::EMPTY;

                // 核心映射：将原位置 i 的棋子挪到变换后的位置 SYMMETRY_MAPS[s][i]
                for (int i = 0; i < 9; ++i) {
                    transformed.board[SYMMETRY_MAPS[s][i]] = raw_board.board[i];
                }

                // 关键：pieces 队列存储的是位置索引，也必须同步变换
                // 这样 boards.to_tensor() 里的 Channel 2 和 3 (消失预警) 才会指向正确的位置
                for (int p_idx : raw_board.pieces) {
                    transformed.pieces.push_back(SYMMETRY_MAPS[s][p_idx]);
                }

                // 将变换后的棋盘放入历史
                boards.push(transformed);
            }
            state_list.push_back(boards.to_tensor());

            // --- 2. 处理 Policy (落子位置映射) ---
            torch::Tensor policy = torch::zeros({9}, torch::kFloat32);
            for (const auto& mp : one.move_probs) {
                // 原本在 mp.move 位置的概率，现在搬到了 SYMMETRY_MAPS[s][mp.move]
                policy[SYMMETRY_MAPS[s][mp.move]] = mp.prob;
            }
            policy_list.push_back(policy);

            // --- 3. 处理 Value (胜率不受旋转/镜像影响) ---
            value_list.push_back(torch::tensor({one.result}, torch::kFloat32));
        }
    }

    return batch{
        torch::stack(state_list),
        torch::stack(policy_list),
        torch::stack(value_list)
    };
}

void print_batch_samples(const batch& b, int n) {
    int batch_size = b.states.size(0);
    int print_n = std::min(n, batch_size);

    for (int k = 0; k < print_n; ++k) {
        std::cout << "==============================" << std::endl;
        std::cout << "Sample " << k << std::endl;

        // ---------- STATE ----------
        // states shape: [B, C, 3, 3]
        auto state = b.states[k];
        int channels = state.size(0);

        std::cout << "[State] channels = " << channels << std::endl;
        for (int c = 0; c < channels; ++c) {
            std::cout << " Channel " << c << ":" << std::endl;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    std::cout << state[c][i][j].item<float>() << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        // ---------- POLICY ----------
        // policy shape: [B, 9]
        auto policy = b.policy[k];
        std::cout << "[Policy]" << std::endl;
        for (int i = 0; i < 9; ++i) {
            std::cout << i << ":" << policy[i].item<float>() << " ";
        }
        std::cout << std::endl;

        // ---------- VALUE ----------
        // value shape: [B, 1]
        float value = b.value[k].item<float>();
        std::cout << "[Value] " << value << std::endl;
    }
}

int main() {
  /*(std::vector<std::string> input_list = {"output(1).txt", "output(2).txt", "output(3).txt", "output(4).txt", "output(5).txt", "output(6).txt", "output(7).txt", "output(8).txt", "output(9).txt", "output(10).txt", "output(11).txt", "output(12).txt", "output(13).txt", "output(14).txt", "output(15).txt"};
  merge_files(input_list, "merged_output_2.txt");
  //std::vector<training_tuple> data = read_file("output.txt");
  for (training_tuple tuple : data) {
    for (std::string each : tuple.fens) {
      std::cout << each << std::endl;
    }
  }*/
  //Todo the main process of training data.
  torch::manual_seed(54384739);
  torch::set_num_threads(4);
  torch::set_num_interop_threads(4);
  int BATCH_SIZE = 64;
  float C = 1e-3;
  int EPOCH = 10000;

  auto model = std::make_shared<neural_network>(12, 32, 4);
  std::ifstream file("model.pt");
  if (file.is_open()) {
    torch::load(model, "model.pt");
    std::cout << "Loading existing model!!" << std::endl;
  }
  model->train();
  torch::globalContext().setFlushDenormal(true);
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions((1e-3));
  std::vector<training_tuple> dataset = read_file("merged_output.txt");
  for (int i = 0; i < EPOCH + 1; ++i) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> indices(dataset.size());
    for (int i = 0; i < dataset.size(); ++i) {
      indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), gen);
    std::vector<int> selected_indices(indices.begin(), indices.begin() + BATCH_SIZE);
    batch one = create_batch(dataset, selected_indices);
    // ==========================
    // 打印检查 policy / label / loss
    // ==========================
    auto output = model->forward(one.states);
    torch::Tensor policy = output.first;
    torch::Tensor value = output.second;
    torch::Tensor logp = torch::log_softmax(policy, 1);
    torch::Tensor policy_loss = -(one.policy * logp).sum(1).mean();
    torch::Tensor value_loss = torch::mse_loss(value, one.value);

    // 打印第一条样本
    torch::Tensor probs = torch::softmax(policy, 1);
    std::cout << "First sample label policy: " << one.policy[0] << std::endl;
    std::cout << "First sample predicted policy: " << probs[0] << std::endl;
    std::cout << "Policy loss (batch): " << policy_loss.item<float>() << std::endl;
    std::cout << "Value loss (batch): " << value_loss.item<float>() << std::endl;

    torch::Tensor l2 = torch::zeros({1});
    for (auto& parameter : model->parameters()) {
      l2 += parameter.pow(2).sum();
    }
    l2 *= C;
    torch::Tensor loss = policy_loss + value_loss + l2;
    optimizer.zero_grad();
    loss.backward();
    torch::Tensor individual_p_losses = -(one.policy * logp).sum(1); 

    // 2. 统计分布数据
    float max_p = individual_p_losses.max().item<float>();
    float min_p = individual_p_losses.min().item<float>();
    
    // 计算“差样本”占比：Loss 大于 2.0 的样本数量
    torch::Tensor high_loss_mask = (individual_p_losses > 2.0).to(torch::kFloat32);
    float high_loss_ratio = high_loss_mask.mean().item<float>();
    if (i % 50 == 0) {
      std::cout << "Epoch " << i
                << " | Policy loss: " << policy_loss.item<float>()
                << " | Value loss: " << value_loss.item<float>() << std::endl;
    }
    if (i % 200 == 0) {
      std::string model_file = "model.pt";
      torch::save(model, model_file);
      std::cout << "Saved model to " << model_file << std::endl;
    }
  }
}
