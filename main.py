# main.py
import torch
import os
from cnn import CNN
from data import collect_data
from deep_reinforcement import MCTSplayer
from cnn import masked_policy
from train import train
from load_data import load_training_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = CNN(512, 10).to(device)

    # 加载已有模型参数
    if os.path.exists("latest_model.pth"):
        model.load_state_dict(torch.load("latest_model.pth"))
        print("✅ Loaded existing model.")

    for iteration in range(10):  # 你想训练多少次就循环多少次
        print(f"🔄 Training iteration {iteration + 1}")

        # 1. 自我对弈生成训练数据
        print("🎮 Self-play generating training data...")
        data_gen = collect_data()
        player = MCTSplayer(masked_policy, model)
        data_gen.self_play(player)

        # 2. 载入训练数据
        print("📦 Loading training data...")
        data = load_training_data("training_data.json")

        # 3. 训练模型
        print("🧠 Training begins...")
        train(model, data, batch_size=64, epochs=5)

        # 4. 保存模型
        torch.save(model.state_dict(), "latest_model.pth")
        print("✅ Model saved to latest_model.pth\n")

if __name__ == "__main__":
    main()

