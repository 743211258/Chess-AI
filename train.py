# train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(network, data, batch_size=64, epochs=5, lr=0.001, weight_decay=1e-4):
    network.train()
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        total_policy_loss = 0
        total_value_loss = 0
        total_samples = 0

        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            mini_batch = data[i:i+batch_size]

            states = torch.stack([d[0] for d in mini_batch]).to(device)           # shape: [B, 13, 8, 8]
            target_pis = torch.stack([d[1] for d in mini_batch]).to(device)      # shape: [B, 3820]
            target_vs = torch.tensor([d[2] for d in mini_batch], dtype=torch.float32).to(device)  # shape: [B]

            optimizer.zero_grad()
            out_pis, out_vs = network(states)  # out_pis: [B, 3820], out_vs: [B, 1]

            loss_policy = -torch.mean(torch.sum(target_pis * F.log_softmax(out_pis, dim=1), dim=1))
            loss_value = F.mse_loss(out_vs.view(-1), target_vs)
            loss = loss_policy + loss_value

            loss.backward()
            optimizer.step()

            total_policy_loss += loss_policy.item() * len(mini_batch)
            total_value_loss += loss_value.item() * len(mini_batch)
            total_samples += len(mini_batch)

        avg_policy_loss = total_policy_loss / total_samples
        avg_value_loss = total_value_loss / total_samples
        print(f"Epoch {epoch+1}/{epochs}: Policy Loss={avg_policy_loss:.4f}, Value Loss={avg_value_loss:.4f}")
