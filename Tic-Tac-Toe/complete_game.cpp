#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <random>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

#include "torch/torch.h"
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#endif
#include <string>
#include <limits>
#include <algorithm>

#include "Tic_Tac_Toe.hpp"
#include "cnn.hpp"
#include "input.hpp"

// --- 辅助工具函数 ---

// 安全读取整数输入，带范围检查
int get_safe_int(const std::string& prompt, int min_val, int max_val) {
    std::string input;
    int choice;
    while (true) {
        std::cout << prompt;
        if (!(std::cin >> choice)) {
            std::cout << "非法输入！请输入数字。\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (choice >= min_val && choice <= max_val) {
            return choice;
        }
        std::cout << "输入超出范围 [" << min_val << "-" << max_val << "]，请重试。\n";
    }
}

void debug_print_history_tensor(torch::Tensor states) {
    // states 维度通常是 [1, 12, 3, 3]
    auto s = states.squeeze(0); // 变成 [12, 3, 3]
    std::string frame_names[3] = {"上上帧 (T-2)", "上一帧 (T-1)", "最新帧 (T)"};
    std::string channel_names[4] = {"我方棋子", "对方棋子", "消失预警", "下回合消失预警"};

    std::cout << "\n========== AI 输入 Tensor 调试 ==========" << std::endl;
    for (int f = 0; f < 3; ++f) {
        std::cout << "--- " << frame_names[f] << " ---" << std::endl;
        for (int c = 0; c < 4; ++c) {
            int channel_idx = f * 4 + c;
            std::cout << "[" << channel_names[c] << "]:" << std::endl;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    float val = s[channel_idx][i][j].item<float>();
                    std::cout << (val > 0.5f ? " 1 " : " . ");
                }
                std::cout << "\n";
            }
        }
    }
    std::cout << "========================================\n" << std::endl;
}

// 打印棋盘
void render_board(const Tic_Tac_Toe::Board::Board& b) {
    std::cout << "\n当前棋盘状态 (1-9 对应位置):\n";
    for (int i = 0; i < 9; ++i) {
        char symbol = '.';
        if (b.board[i] == Tic_Tac_Toe::Board::CROSS) symbol = 'X';
        else if (b.board[i] == Tic_Tac_Toe::Board::NOUGHT) symbol = 'O';
        
        // 如果该子即将消失（在 pieces 队列的最前端且队列已满 6 个）
        if (!b.pieces.empty() && b.pieces.size() == 6 && b.pieces.front() == i) {
            std::cout << "(" << symbol << ")"; // 用括号标记即将消失的子
        } else {
            std::cout << " " << symbol << " ";
        }

        if ((i + 1) % 3 == 0) std::cout << "\n";
    }
    std::cout << "当前轮到: " << (b.player == Tic_Tac_Toe::Board::CROSS ? "X" : "O") 
              << " | 回合数: " << b.turn << "\n";
}

// --- 游戏逻辑执行 ---

void play_game(neural_network* network) {
    while (true) {
        int mode = get_safe_int("请选择模式 (1: 人类 vs 人类, 2: 人类 vs 电脑): ", 1, 2);
        
        Tic_Tac_Toe::Board::Board board; // 自动初始化
        board_history history(3); // 假设 max_history 为 3
        
        int human_side = 0; // 0: N/A, 1: 先手(X), 2: 后手(O)
        if (mode == 2) {
            human_side = get_safe_int("请选择你的棋子 (1: X先手, 2: O后手): ", 1, 2);
        }
	history.push(board);
        while (Tic_Tac_Toe::Result::result_state(board) == Tic_Tac_Toe::Result::ONGOING) {
            render_board(board);

            int move_idx = -1;
            // 判断当前该谁走
            bool is_human_turn = (mode == 1) || 
                                 (human_side == 1 && board.player == Tic_Tac_Toe::Board::CROSS) ||
                                 (human_side == 2 && board.player == Tic_Tac_Toe::Board::NOUGHT);

            if (is_human_turn) {
                while (true) {
                    move_idx = get_safe_int("请输入落子位置 (1-9): ", 1, 9) - 1;
                    if (board.board[move_idx] == Tic_Tac_Toe::Board::EMPTY) break;
                    std::cout << "该位置已有棋子，请选择空位！\n";
                }
            } else {
                std::cout << "电脑思考中...\n";
                torch::NoGradGuard no_grad; // 推理模式关闭
                auto input_tensor = history.to_tensor().unsqueeze(0);
                debug_print_history_tensor(input_tensor); // <--- 调用调试打印
                auto output = network->forward(history.to_tensor().unsqueeze(0)); // Batch size 1
                
                // 获取 Policy (第一个输出是 policy)
                torch::Tensor policy = std::get<0>(output).exp(); // 假设网络输出的是 log_softmax
                
                // 筛选合法动作
                std::vector<int> legal_moves = Tic_Tac_Toe::Move::get_legal_move(board);
                float max_prob = -1.0f;
                for (int m : legal_moves) {
                    float p = policy[0][m].item<float>();
		    std::cout << p << std::endl;
                    if (p > max_prob) {
                        max_prob = p;
                        move_idx = m;
                    }
                }
                std::cout << "电脑落子: " << move_idx + 1 << "\n";
            }
            Tic_Tac_Toe::Move::move(board, move_idx);
	    history.push(board);
        }

        // 游戏结束
        render_board(board);
        auto result = Tic_Tac_Toe::Result::result_state(board);
        if (result == Tic_Tac_Toe::Result::WIN) {
            // 注意：result_state 判定的是刚才落子的人是否达成连线
            // 此时 board.player 已经切换，所以赢家是“非当前玩家”
            std::cout << "游戏结束！获胜者是: " << (board.player == Tic_Tac_Toe::Board::NOUGHT ? "X" : "O") << "\n";
        } else {
            std::cout << "游戏结束！平局。\n";
        }

        int retry = get_safe_int("是否再来一局？(1: 是, 0: 退出): ", 0, 1);
        if (retry == 0) break;
    }
}

int main() {
    #ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    #endif
    try {
        std::cout << "正在初始化系统...\n";

        // 使用 shared_ptr，这样 torch::load(network, ...) 就能匹配成功
        auto network = std::make_shared<neural_network>(12, 32, 4);

        if (std::ifstream("model.pt").is_open()) {
            torch::load(network, "model.pt");
            std::cout << "成功加载模型 model.pt\n";
        } else {
            std::cout << "未找到 model.pt，使用随机权重。\n";
        }

        network->eval();
        play_game(network.get()); // 传入裸指针给 play_game

    } catch (const std::exception& e) {
        std::cerr << "程序发生致命错误: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
