import torch
import torch.nn.functional as F
from torch import nn
from game import move_id
import chess
from input import board_to_tensor
'''class CNN(nn.Module):
    def __init__(self, kernel_size, num_kernel_in_first_layer, num_kernel_in_second_layer, padding):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.num_kernel_in_first_layer = num_kernel_in_first_layer
        self.num_kernel_in_second_layer = num_kernel_in_second_layer
        self.padding = padding
        self.cnn_one = nn.Conv2d(in_channels = num_kernel_in_first_layer,
                                 out_channels = num_kernel_in_second_layer,
                                 kernel_size = kernel_size,
                                 padding = padding)
        
        self.bn1 = nn.BatchNorm2d(num_kernel_in_second_layer)
        self.dropout1 = nn.Dropout2d(0.3)

        self.cnn_two = nn.Conv2d(in_channels = num_kernel_in_second_layer,
                                 out_channels = num_kernel_in_second_layer * 2,
                                 kernel_size = kernel_size,
                                 padding = padding)
        
        self.bn2 = nn.BatchNorm2d(num_kernel_in_second_layer * 2)
        self.dropout2 = nn.Dropout2d(0.3)

        self.fully_connect_one = nn.Linear(num_kernel_in_second_layer * 2 * 8 * 8,
                                           num_kernel_in_second_layer * 2)
        
        self.bn3 = nn.BatchNorm1d(num_kernel_in_second_layer * 2)

        self.fully_connect_two = nn.Linear(num_kernel_in_second_layer * 2, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.cnn_one(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.cnn_two(x)))
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn3(self.fully_connect_one(x)))
        return self.fully_connect_two(x)'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Resnet(nn.Module):
    def __init__(self, filters):
        super(Resnet, self).__init__()
        self.cnn_one = nn.Conv2d(in_channels = filters,
                                 out_channels = filters,
                                 kernel_size = 3,
                                 padding = 1)
        
        self.bn1 = nn.BatchNorm2d(filters)

        self.cnn_two = nn.Conv2d(in_channels = filters,
                                 out_channels = filters,
                                 kernel_size = 3,
                                 padding = 1)
        
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.cnn_one(x)))
        x = self.bn2(self.cnn_two(x))
        x = x + identity
        return F.relu(x)

class CNN(nn.Module):
    def __init__(self, filters, nums_res_block):
        super(CNN, self).__init__()
        self.cnn_one = nn.Conv2d(in_channels = 13,
                                 out_channels = filters,
                                 kernel_size = 3,
                                 padding = 1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.res_blocks = nn.ModuleList()

        for _ in range(nums_res_block):
            block = Resnet(filters)
            self.res_blocks.append(block)
        
        self.policy_cnn = nn.Conv2d(in_channels = filters,
                                    out_channels = 16,
                                    kernel_size = 1)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fully_connect = nn.Linear(16 * 8 * 8, 3820)
        
        self.net_cnn = nn.Conv2d(in_channels = filters,
                                 out_channels = 8,
                                 kernel_size = 1)
        self.net_bn = nn.BatchNorm2d(8)
        self.net_fully_connect = nn.Linear(8 * 8 * 8, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.cnn_one(x)))
        for block in self.res_blocks:
            x = block(x)
        
        policy = F.relu(self.policy_bn(self.policy_cnn(x)))
        policy = torch.flatten(policy, 1)
        policy = self.policy_fully_connect(policy)

        value = F.relu(self.net_bn(self.net_cnn(x)))
        value = torch.flatten(value, 1)
        value = self.net_fully_connect(value)
        value = F.tanh(value)
        
        return policy, value

def masked_policy(board, network):
    network.eval()
    uci = []
    tensor = board_to_tensor(board).unsqueeze(0).to(device)
    network = network.to(device)
    legal_moves = board.legal_moves
    uci = [move.uci() for move in legal_moves]
    indices = [i for i, item in enumerate(move_id) if item in uci]
    with torch.no_grad():
        unmasked_policy, value = network(tensor)
    unmasked_policy = unmasked_policy.flatten()
    masked_logits = torch.full_like(unmasked_policy, float('-inf'))
    masked_logits[indices] = unmasked_policy[indices]

    probs = F.softmax(masked_logits, dim=0)
    move_probs = list(zip(legal_moves, probs[indices]))
    return move_probs, value.item()

'''board = chess.Board("1BQ1qkBN/R6r/1Q6/6BR/2bN4/4Q1BK/1p6/1bq1R1rb w - - 1 3")

class policy_value_network():
    def __init__(self, model = None):
        self.model = model
        self.network = CNN(512, 10)
        self.network.eval()
    
    def policy_value(self, board):
        act_probs, value = masked_policy(board, self.network)'''