# ----------------------------------------
# 🗳️ Election Prediction using ANN (Simple)
# ----------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ----------------------------------------
# 1️⃣ Dataset
# ----------------------------------------

# Input features:
# [voter_turnout, youth_votes, rural_support, urban_support, anti_incumbency]

X = np.array([
    [72, 40, 60, 55, 1],
    [68, 35, 55, 60, 0],
    [75, 45, 65, 58, 1],
    [80, 50, 70, 62, 1],
    [66, 33, 52, 61, 0],
    [70, 38, 58, 59, 0],
    [78, 48, 68, 64, 1],
    [74, 44, 63, 60, 1],
    [69, 36, 57, 57, 0],
    [77, 47, 66, 63, 1]
], dtype=float)

# Output:
# DMK = 1, AIADMK = 0
y = np.array([
    [1], [0], [1], [1], [0],
    [0], [1], [1], [0], [1]
], dtype=float)

# ----------------------------------------
# 2️⃣ Convert to PyTorch Tensors
# ----------------------------------------

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# ----------------------------------------
# 3️⃣ Define ANN Model
# ----------------------------------------

class ElectionANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(5, 4)   # 5 inputs → 4 neurons
        self.output = nn.Linear(4, 1)   # 1 output

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))  # sigmoid for classification
        return x

model = ElectionANN()

# ----------------------------------------
# 4️⃣ Loss & Optimizer
# ----------------------------------------

criterion = nn.BCELoss()              # Binary classification loss
optimizer = optim.SGD(model.parameters(), lr=0.001)

# ----------------------------------------
# 5️⃣ Training Loop
# ----------------------------------------

for epoch in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ----------------------------------------
# 6️⃣ Predict 2026 Election
# ----------------------------------------

# Example 2026 scenario
test_input = torch.tensor([[76, 46, 67, 63, 1]], dtype=torch.float32)

with torch.no_grad():
    prob = model(test_input)

winner = "DMK" if prob.item() >= 0.5 else "AIADMK"

print("\n🔮 2026 Election Prediction")
print("Winner:", winner)
print("Winning Probability:", prob.item())
