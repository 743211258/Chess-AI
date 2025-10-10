import torch
import os
from cnn import CNN
from data import collect_data
from deep_reinforcement import MCTSplayer
from cnn import masked_policy
from load_data import load_random_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(num_iterations=6000, batch_size=128):
    model = CNN(512, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    if os.path.exists("latest_model.pth"):
        model.load_state_dict(torch.load("latest_model.pth", map_location=device))
        print("✅ Loaded existing model.")

    model.train()

    for iteration in range(num_iterations):
        state_batch, target_pi_batch, target_v_batch = load_random_batch("training_data.jsonl", batch_size)

        state_batch = state_batch.to(device)              # [B, 13, 8, 8]
        target_pi_batch = target_pi_batch.to(device)      # [B, 3820]
        target_v_batch = target_v_batch.to(device)        # [B]

        pred_pi, pred_v = model(state_batch)              # pred_pi: [B, 3820], pred_v: [B, 1]
        pred_v = pred_v.view(-1)                          # [B]

        # AlphaZero loss
        value_loss = (pred_v - target_v_batch).pow(2).mean()
        policy_loss = -torch.sum(target_pi_batch * torch.log_softmax(pred_pi, dim=1)) / batch_size
        loss = value_loss + policy_loss 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 50 == 0:
            print(f"🧠 Iter {iteration+1}/{num_iterations} | Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

        if (iteration + 1) % 200 == 0:
            torch.save(model.state_dict(), "latest_model.pth")
            print("✅ Model saved.\n")

if __name__ == "__main__":
    main(num_iterations=6000, batch_size=128)


