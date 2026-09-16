#include "pch.hpp"

#include "Tic_tac_Toe.hpp"
#include "cnn.hpp"
#include "input.hpp"

inline std::vector<float> dirichlet_noise(int size, float alpha) {
  std::vector<float> normalized_noise(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::gamma_distribution<float> gamma_dist(alpha, 1.0f);
  for (int i = 0; i < size; ++i) {
    normalized_noise[i] = gamma_dist(gen);
  }
  float sum = std::accumulate(normalized_noise.begin(),
                               normalized_noise.end(),
                               0.0f);
  for (float& element : normalized_noise) {
    element /= sum;
  }
  return normalized_noise;
}

class Node {
  private:
    Node *parent;
    std::unordered_map<int, std::unique_ptr<Node>> children;
    int visits;
    float Q;
    float policy;
  public:
    Node(Node *parent, float policy) {
      this->parent = parent;
      this->children = std::unordered_map<int, std::unique_ptr<Node>>();
      this->visits = 0;
      this->Q = 0.0f;
      this->policy = policy;
    }

    float puct(float c) {
      return this->Q + (c * this->policy * std::sqrt((float)this->parent->visits) / (1.0f + this->visits));
    }

    std::unordered_map<int, std::unique_ptr<Node>>::iterator select(float c) {
      float highest = -std::numeric_limits<float>::infinity();
      auto best_children = this->children.end();
      for (auto element = this->children.begin(); element != this->children.end(); ++element) {
        float current = element->second->puct(c);
        if (current > highest) {
          highest = current;
          best_children = element;
        }
      }
      return best_children;
    }

    void expand(std::vector<move_probability>& output) {
      for (move_probability element : output) {
        this->children.insert({element.move, std::make_unique<Node>(this, element.prob)});
      }
    }

    void update(float rollout) {
      this->visits++;
      this->Q += (rollout - this->Q) / this->visits;
    }

    void backpropagation(float rollout) {
      if (this->parent != nullptr) {
        this->parent->backpropagation(-rollout);
      }
      this->update(rollout);
    }

    bool is_leaf() {
      return this->children.empty();
    }

    bool is_root() {
      return this->parent == nullptr;
    }

    void setParent(Node *parent) {
      this->parent = parent;
    }

    std::unordered_map<int, std::unique_ptr<Node>>& getChildren() {
      return children;
    }

    int getVisits() {
      return visits;
    }

    float getPolicy() {
      return policy;
    }
    void setPolicy(float policy) {
      this->policy = policy;
    }
};

class MCTS {
  private:
    std::unique_ptr <Node> root;
    std::pair<std::vector<move_probability>, float> (*masked_policy_pointer) (board_history&, neural_network&);
    neural_network* network_pointer;
    float c;
    int iteration;
  public:
    MCTS(std::pair<std::vector<move_probability>, float> (*function_pointer) (board_history&, neural_network&),
         neural_network* network_pointer,
         float c, int iteration) {
      this->root = std::make_unique<Node>(nullptr, 1.0f);
      this->masked_policy_pointer = function_pointer;
      this->network_pointer = network_pointer;
      this->c = c;
      this->iteration = iteration;
    }

    void algorithm(board_history boards) {
      Node* node = this->root.get();
      float value = 0.0;
      Tic_Tac_Toe::Board::Board sim_board;
      sim_board = boards.get_history().back();
      while (true) {
        if (node->is_leaf()) {
          break;
        }
        std::unordered_map<int, std::unique_ptr<Node>>::iterator best_children = node->select(this->c);
	Tic_Tac_Toe::Move::move(sim_board, best_children->first);
        boards.push(sim_board);
        node = best_children->second.get();
      }
      if (Tic_Tac_Toe::Result::result_state(sim_board) != Tic_Tac_Toe::Result::ONGOING) {
	Tic_Tac_Toe::Result::STATE result = Tic_Tac_Toe::Result::result_state(sim_board);
        if (result == Tic_Tac_Toe::Result::DRAW) {
          value = 0.0f;
        } else {
          value = 1.0f;
        }
      } else {
        std::pair<std::vector<move_probability>, float> output = this->masked_policy_pointer(boards, *network_pointer);
        if (node == this->root.get()) {
          std::vector<float> noise = dirichlet_noise(output.first.size(), 0.7f);
          int i = 0;
          for (auto& [key, value] : output.first) {
            value = 0.75f * value + 0.25f * noise[i];
            ++i;
          }
        }
        node->expand(output.first);
        value = output.second;
      }
      node->backpropagation(value);
    }

    std::vector<move_probability> mcts_output(board_history boards, float temp) {
      for (int i = 0; i < this->iteration; ++i) {
        this->algorithm(boards);
        if (i % 50 == 0) {
          std::cout << i << " iterations." << std::endl;
        }
      }
      std::vector<move_probability> moves_and_probs;
      std::vector<int> node_moves;
      std::vector<float> node_visits;
      for (auto& [move, node] : root->getChildren()) {
        node_moves.push_back(move);
        node_visits.push_back(node->getVisits());
        std::cout << move << " " << node->getVisits() << std::endl;
      }
      torch::Tensor visits_tensor = torch::tensor(node_visits, torch::kFloat32);
      torch::Tensor chosen_probability;
      if (temp < 0.5f) {
        auto max_index = visits_tensor.argmax().item<int>();
        chosen_probability = torch::zeros_like(visits_tensor);
        chosen_probability[max_index] = 1.0f;
      } else {
        chosen_probability = torch::exp(torch::log(visits_tensor + 1e-10f) / temp);
        chosen_probability /= chosen_probability.sum();
      }
      for (int i = 0; i < root->getChildren().size(); ++i) {
        moves_and_probs.push_back(move_probability{node_moves[i], chosen_probability[i].item<float>()});
      }
      return moves_and_probs;
    }

    void update_tree(int selected_move) {
      if (this->root->getChildren().find(selected_move) != this->root->getChildren().end()) {
        this->root = std::move(this->root->getChildren()[selected_move]);
        this->root->setParent(nullptr);
        if (this->root->getChildren().size() != 0) {
          std::vector<float> noise = dirichlet_noise(this->root->getChildren().size(), 0.7f);
          int i = 0;
          for (auto& [key, value] : this->root->getChildren()) {
            value->setPolicy(0.75f * value->getPolicy() + 0.25f * noise[i]);
            ++i;
          }
        }
      } else {
        this->root = std::make_unique<Node>(nullptr, 1.0f);
      }
    }
};

class MCTSplayer {
  private:
    std::unique_ptr<MCTS> mcts;
    std::mt19937 gen;
  public:
    MCTSplayer(std::pair<std::vector<move_probability>, float> (*function_pointer) (board_history&, neural_network&),
               neural_network* network_pointer,
               float c, int iteration) {
      this->mcts = std::make_unique<MCTS>(function_pointer, network_pointer, c, iteration);
      this->gen = std::mt19937(std::random_device{}());
    }

    std::pair<int, std::vector<move_probability>> play(board_history boards, float temp) {
      std::vector<move_probability> output = this->mcts->mcts_output(boards, temp);
      if (output.size() == 0) {
        return std::make_pair(-1, std::vector<move_probability>());
      }
      std::vector<float> probs;
      for (auto& each : output) {
        probs.push_back(each.prob);
      }
      int index;
      if (temp < 0.5) {
        index = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
      } else {
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        index = dist(this->gen);
      }
      move_probability best_move = output[index];
      int move = best_move.move;
      this->mcts->update_tree(move);
      return std::make_pair(move, output);
    }
};
