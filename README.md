# 🗳️ Election Prediction using ANN

An Artificial Neural Network (ANN) based election prediction system built using PyTorch.

This project predicts election outcomes using simulated political and voting-related features.

---

# 🚀 Features

- Artificial Neural Network implementation
- Binary election prediction
- PyTorch-based deep learning model
- Training using voter-related data
- Future election prediction simulation

---

# 📊 Input Features

The model uses:

1. Voter Turnout
2. Youth Votes
3. Rural Support
4. Urban Support
5. Anti-Incumbency

---

# 🎯 Output

Prediction:

- DMK → 1
- AIADMK → 0

The model predicts:
- probable winner
- winning probability

---

# 🧠 Technologies Used

- Python
- PyTorch
- NumPy

---

# ⚙️ ANN Architecture

Input Layer:
- 5 neurons

Hidden Layer:
- 4 neurons
- ReLU activation

Output Layer:
- 1 neuron
- Sigmoid activation

Loss Function:
- Binary Cross Entropy Loss (BCELoss)

Optimizer:
- SGD (Stochastic Gradient Descent)

---

# 📂 Project Structure

```

election-prediction-ann/
│
├── main.py
├── requirements.txt
└── README.md

```

---

# ▶️ How to Run

## Install dependencies

```bash
pip install torch numpy
```

## Run the project

```bash
python main.py
```

---

# 🔮 Example Prediction

Example 2026 election scenario:

```python
[76, 46, 67, 63, 1]
```

Output:

```text
Winner: DMK
Winning Probability: 0.87
```

---

# 📌 Future Improvements

- Real-world election datasets
- More political features
- GUI/Web App
- Better deep learning architecture
- Visualization dashboards

---

# 👨‍💻 Author

Mithun Ram

---

# ⭐ GitHub

If you like this project, consider starring the repository.
