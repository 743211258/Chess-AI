#include <iostream>
#ifdef _WIN32
#include <windows.h>
#endif

#include "tic_tac_toe.hpp"

using namespace Tic_Tac_Toe;

void print_board(Board::Board& board) {
    for (int i = 0; i < BOARD_SIZE; ++i) {
        char c = '.';
        if (board.board[i] == Board::CROSS) c = 'X';
        else if (board.board[i] == Board::NOUGHT) c = 'O';
        std::cout << c;
        if ((i + 1) % 3 == 0) std::cout << "\n";
        else std::cout << " ";
    }
}

int main() {
    #ifdef _WIN32
        // 设置控制台输入输出为 UTF-8
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    #endif
    Board::Board board; // 自动初始化
    std::cout << "=== 消失版 Tic-Tac-Toe ===\n";
    std::cout << "X 先手，O 后手\n";

    while (Result::result_state(board) == Result::ONGOING) {
        print_board(board);
        std::vector<int> moves = Move::get_legal_move(board);

        if (moves.empty()) break;

        int m = -1;
	bool valid_input = false;

	while (!valid_input) {
	    if (board.player) {
		std::cout << "玩家 X 输入位置 (1-9): ";
	    } else {
		std::cout << "玩家 O 输入位置 (1-9): ";
	    }

	    std::string line;
	    std::getline(std::cin, line);  // 用 getline 避免输入缓冲问题

	    try {
		m = std::stoi(line);  // 尝试将字符串转为整数
	    } catch (...) {
		std::cout << "输入无效，请输入 1-9 的数字\n";
		continue;
	    }

	    if (m < 1 || m > 9) {
		std::cout << "输入超出范围，请输入 1-9 的数字\n";
		continue;
	    }

	    if (!Move::move(board, m - 1)) {
		std::cout << "位置已被占用，请重新输入\n";
		continue;
	    }

	    valid_input = true; // 输入合法且成功落子
	}

    }

    print_board(board);

    Result::STATE state = Result::result_state(board);
    if (state == Result::WIN) {
        std::cout << "玩家 " << (board.player ? "O" : "X") << " 赢了！\n";
    } else if (state == Result::DRAW) {
        std::cout << "平局！\n";
    }
    std::cout << "按任意键退出..." << std::endl;
    std::cin.get();
    return 0;
}

