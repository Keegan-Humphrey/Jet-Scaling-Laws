import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ========== Load Data ==========
df = pd.read_csv('log_efps_labeled.csv', header=None, skiprows=1)

# Use only column 62 and label (undo log10)
X = 10 ** df.iloc[:, 62].values.astype(np.float32).reshape(-1, 1)
y = df.iloc[:, 314].values.astype(np.int64)

# Remove mismatch if needed
min_len = min(len(X), len(y))
X = X[:min_len]
y = y[:min_len]

# Normalize input
X_mean = X.mean()
X_std = X.std()
X = (X - X_mean) / X_std

# Train/test split (with shuffling)
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]
num_train = int(0.8 * len(X))
X_train, X_test = X[:num_train], X[num_train:]
y_train, y_test = y[:num_train], y[num_train:]

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.nn.functional.one_hot(torch.tensor(y_train), num_classes=2).float()
y_test = torch.nn.functional.one_hot(torch.tensor(y_test), num_classes=2).float()

# DataLoaders
train_ds = torch.utils.data.TensorDataset(X_train, y_train)
test_ds = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)


# ========== Define Residual MLP Model ==========
class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(1, 16)
        self.hidden = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.output = nn.Linear(16, 2)

    def forward(self, x):
        x = self.input(x)
        residual = x
        x = self.hidden(x)
        x += residual  # Residual connection
        return self.output(x)


# ========== Long Training Loop ==========
def train_long(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=10000):
    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

    return train_losses, test_losses


# ========== Log-Log Loss Plot ==========
def plot_log_losses(train_losses, test_losses):
    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(train_losses) + 1)
    plt.loglog(epochs, train_losses, label='Train Loss', marker='.')
    plt.loglog(epochs, test_losses, label='Test Loss', marker='.')
    plt.xlabel('Epoch (log scale)')
    plt.ylabel('Loss (log scale)')
    plt.title('log(Training/Test Loss) vs log(Epoch)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("Loss.png")
    plt.show()


# ========== Run Everything ==========
model = ResidualMLP()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
criterion = nn.MSELoss()

train_losses, test_losses = train_long(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=10000)
plot_log_losses(train_losses, test_losses)

