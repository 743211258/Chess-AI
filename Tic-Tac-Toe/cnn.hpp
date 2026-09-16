#ifndef CNN_HPP
#define CNN_HPP

#include "Tic_Tac_Toe.hpp"
#include "input.hpp"
#include "torch/torch.h"

#include <string>
#include <utility>
#include <unordered_map>

struct move_probability {
  int move;
  float prob;
};

struct residual_block : torch::nn::Module {
  residual_block(int64_t channels);
  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
};

struct neural_network : torch::nn::Module {
  neural_network(int64_t in_channels, int64_t out_channels, int64_t nums_res_blocks);
  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

private:
  torch::nn::Conv2d conv{nullptr}, policy_conv{nullptr}, value_conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr}, policy_bn{nullptr}, value_bn{nullptr};
  torch::nn::Sequential resnet18{nullptr};
  torch::nn::Linear policy_fc{nullptr}, value_fc{nullptr};
};

std::pair<std::vector<move_probability>, float> masked_output(
    board_history& boards,
    neural_network& network
);

#endif
