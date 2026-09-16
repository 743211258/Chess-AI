#include <iostream>
#include <vector>
#include <deque>
#include <fstream>
#include <sstream>
#include "Tic_Tac_Toe.hpp"
#include "cnn.hpp"
#include "input.hpp"

struct training_tuple {
  std::vector<Tic_Tac_Toe::Board::Board> boards;
  std::vector<move_probability> move_probs;
  int result;
};

// 坐标转换逻辑
int rotate_index(int idx, int degree) {
    static const int rot90[]  = {6, 3, 0, 7, 4, 1, 8, 5, 2};
    static const int rot180[] = {8, 7, 6, 5, 4, 3, 2, 1, 0};
    static const int rot270[] = {2, 5, 8, 1, 4, 7, 0, 3, 6};
    if (degree == 90)  return rot90[idx];
    if (degree == 180) return rot180[idx];
    if (degree == 270) return rot270[idx];
    return idx;
}

// 旋转 Board 结构体
Tic_Tac_Toe::Board::Board rotate_board(const Tic_Tac_Toe::Board::Board& original, int degree) {
    if (degree == 0) return original;
    Tic_Tac_Toe::Board::Board rotated;
    rotated.player = original.player;
    rotated.turn = original.turn;
    for (int i = 0; i < 9; ++i) {
        rotated.board[rotate_index(i, degree)] = original.board[i];
    }
    for (int pos : original.pieces) {
        rotated.pieces.push_back(rotate_index(pos, degree));
    }
    return rotated;
}

// 执行单个局面的旋转泛化测试
void test_single_tuple(const training_tuple& tuple, neural_network& network) {
    int degrees[] = {0, 90, 180, 270};
    
    // 关键点：这里必须是 3，因为 3帧 * 3通道 = 9通道，对齐你的模型权重
    const int REQUIRED_HISTORY = 3; 

    for (int deg : degrees) {
        std::cout << ">>> Rotation: " << deg << " degrees" << std::endl;
        
        // 1. 创建 history 对象，大小必须对齐训练参数
        board_history bh(REQUIRED_HISTORY);

        // 2. 填充历史数据
        // 如果 tuple.boards 里的帧数不够，就用最后一帧重复填充
        // 如果够了，就按顺序压入旋转后的 board
        for (int i = 0; i < REQUIRED_HISTORY; ++i) {
            if (i < (int)tuple.boards.size()) {
                // 这里的顺序要和你训练时的 push 逻辑一致（通常是 push_front 或 push_back）
                bh.push(rotate_board(tuple.boards[i], deg));
            } else {
                // 如果历史不足，用已有的最后一帧补齐
                bh.push(rotate_board(tuple.boards.back(), deg));
            }
        }

        // 3. 执行推理
        // masked_output 内部会调用 to_tensor()，此时会生成 3*3=9 个通道
        auto output = masked_output(bh, network);
        auto policy = output.first;
        float value = output.second;

        // 4. 打印结果
        std::cout << "Value (Win Prob): " << value << std::endl;
        std::cout << "Policy probabilities:" << std::endl;
        for (auto& mp : policy) {
            std::cout << "  Move " << mp.move << ": " << mp.prob << std::endl;
        }
        std::cout << "------------------------------------" << std::endl;
    }
}

std::vector<training_tuple> read_file(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<training_tuple> outcome;
    std::string line;
    std::stringstream ss;
    int num;
    int loop;

    while (true) {
        training_tuple tuple;
        bool eof = false;
        if (!std::getline(file, line)) {
          break;
        }
        ss.clear();
        ss.str(line);
        ss >> loop;
        for (int i = 0; i < loop; ++i) {
            Tic_Tac_Toe::Board::Board board;
            // 每个 board 4 行
            for (int j = 0; j < 4; ++j) {
                if (!std::getline(file, line)) {
                    eof = true;
                    break;
                }
                ss.clear();
                ss.str(line);
                if (j == 0) { // pieces
                  int temp = 0;
                  while (ss >> num) {
                    board.board[temp] = static_cast<Tic_Tac_Toe::Board::STATE>(num);
                    ++temp;
                  }
                } else if (j == 1) {
                   while (ss >> num) board.pieces.push_back(num);
                } else if (j == 2) { // player
                    int player_int;
                    ss >> player_int;
                    board.player = static_cast<Tic_Tac_Toe::Board::STATE>(player_int);
                } else { // turn
                    ss >> board.turn;
                }
            }
            if (eof) break;
            tuple.boards.push_back(board);
        }
        if (eof) break;

        // 后续 move_probs + result
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (line == "1-0") {
              Tic_Tac_Toe::Board::Board board = tuple.boards[0];
              if (board.player == Tic_Tac_Toe::Board::CROSS) {
                tuple.result = 1;
              } else {
                tuple.result = -1;
              }
              break;
            } else if (line == "1/2-1/2") {
              tuple.result = 0;
              break;
            } else if (line == "0-1") {
              Tic_Tac_Toe::Board::Board board = tuple.boards[0];
              if (board.player == Tic_Tac_Toe::Board::NOUGHT) {
                tuple.result = 1;
              } else {
                tuple.result = -1;
              }
              break;
            } else {
                move_probability mp{};
                std::istringstream iss(line);
                iss >> mp.move >> mp.prob;
                tuple.move_probs.push_back(mp);
            }
        }

        outcome.push_back(tuple);
    }

    return outcome;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <data_file>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string data_file = argv[2];

    // 1. 初始化网络 (假设 max_history=3, 每帧 3 通道)
    auto network = std::make_shared<neural_network>(4 * 3, 32, 8); 
    torch::load(network, model_path);
    network->eval();

    // 2. 使用你提供的 read_file 函数读取局面
    std::vector<training_tuple> test_data = read_file(data_file);
    if (test_data.empty()) {
        std::cerr << "No data found in file." << std::endl;
        return 1;
    }

    // 3. 运行测试 (这里默认测试文件里的第一个 tuple)
    std::cout << "Found " << test_data.size() << " samples. Testing the first one..." << std::endl;
    test_single_tuple(test_data[0], *network);

    return 0;
}
