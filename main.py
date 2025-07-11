# main.py
import torch
import os
from cnn import CNN
from data import collect_data
from deep_reinforcement import MCTSplayer
from cnn import masked_policy
from load_data import load_random_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(num_iterations=1000):
    model = CNN(512, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 载入已有模型
    if os.path.exists("latest_model.pth"):
        model.load_state_dict(torch.load("latest_model.pth", map_location=device))
        print("✅ Loaded existing model.")

    model.train()

    for iteration in range(num_iterations):
        # 加载一个随机样本
        state, target_pi, target_v = load_random_sample("training_data.jsonl")
        state = state.unsqueeze(0).to(device)           # [1, 13, 8, 8]
        target_pi = target_pi.unsqueeze(0).to(device)   # [1, 3820]
        target_v = torch.tensor([target_v], dtype=torch.float32).to(device)  # [1]

        # 前向传播
        pred_pi, pred_v = model(state)                  # pred_pi: [1, 3820], pred_v: [1, 1]

        # AlphaZero loss
        value_loss = (pred_v.view(-1) - target_v).pow(2).mean()
        policy_loss = -torch.sum(target_pi * torch.log_softmax(pred_pi, dim=1))
        loss = value_loss + policy_loss  # L2 正则已包含在 Adam 的 weight_decay 中

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印日志
        if (iteration + 1) % 50 == 0:
            print(f"🧠 Iter {iteration+1}/{num_iterations} | Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

        # 定期保存模型
        if (iteration + 1) % 200 == 0:
            torch.save(model.state_dict(), "latest_model.pth")
            print("✅ Model saved.\n")

if __name__ == "__main__":
    main(num_iterations=6000)

