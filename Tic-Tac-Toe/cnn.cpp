#include "pch.hpp"

#include "Tic_Tac_Toe.hpp"
#include "cnn.hpp"
#include "input.hpp"

residual_block::residual_block(int64_t channels) {
  conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).stride(1).padding(1).bias(false)));
  conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).stride(1).padding(1).bias(false)));
  bn1 = register_module("bn1", torch::nn::BatchNorm2d(channels));
  bn2 = register_module("bn2", torch::nn::BatchNorm2d(channels));
}

torch::Tensor residual_block::forward(torch::Tensor x) {
  torch::Tensor out = torch::relu(bn1(conv1(x)));
  out = bn2(conv2(out));
  out += x;
  return torch::relu(out);
}

neural_network::neural_network(int64_t in_channels, int64_t out_channels, int64_t nums_res_blocks) {
  conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(1).bias(false)));
  policy_conv = register_module("policy_conv",
                                  torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, 8, 1).stride(1).padding(0).bias(false)));
  value_conv = register_module("value_conv",
                                 torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, 8, 1).stride(1).padding(0).bias(false)));
  bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
  policy_bn = register_module("policy_bn", torch::nn::BatchNorm2d(8));
  value_bn = register_module("value_bn", torch::nn::BatchNorm2d(8));
  resnet18 = register_module("resnet18", torch::nn::Sequential());
  for (int i = 0; i < nums_res_blocks; i++) {
    auto block = std::make_shared<residual_block>(out_channels);
    resnet18->push_back(block);
  }
  policy_fc = register_module("policy_fc", torch::nn::Linear(8 * 3 * 3, 9));
  value_fc = register_module("value_fc", torch::nn::Linear(8 * 3 * 3, 1));
}

std::pair<torch::Tensor, torch::Tensor> neural_network::forward(torch::Tensor x) {
  x = torch::relu(bn(conv(x)));
  x = resnet18->forward(x);

  torch::Tensor policy = torch::relu(policy_bn(policy_conv(x)));
  policy = policy.view({policy.size(0), -1});
  policy = policy_fc(policy);

  torch::Tensor value = torch::relu(value_bn(value_conv(x)));
  value = value.view({value.size(0), -1});
  value = torch::tanh(value_fc(value));

  return {policy, value};
}


std::pair<std::vector<move_probability>, float> masked_output(board_history& boards, neural_network& network) {
  network.eval();

  torch::Tensor tensor = boards.to_tensor().unsqueeze(0);

  std::vector<int> indices = Tic_Tac_Toe::Move::get_legal_move(boards.get_history().back());

  auto output = network.forward(tensor);
  torch::Tensor unmasked_policy = output.first;
  torch::Tensor value = output.second;

  unmasked_policy = unmasked_policy.flatten();

  torch::Tensor masked_logits = torch::full_like(unmasked_policy, -std::numeric_limits<float>::infinity());
  for (size_t index : indices) {
    masked_logits[index] = unmasked_policy[index];
  }
  torch::Tensor probs = torch::softmax(masked_logits, -1);
  std::vector<move_probability> move_probs;
  for (size_t i = 0; i < indices.size(); ++i) {
    move_probs.push_back(move_probability{indices[i], probs[indices[i]].item<float>()});
  }
  return std::make_pair(move_probs, value.item<float>());
}
