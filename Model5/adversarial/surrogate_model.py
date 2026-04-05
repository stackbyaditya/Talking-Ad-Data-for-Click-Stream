"""Differentiable surrogate neural network for gradient-based tabular attacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Small multi-layer perceptron used as a differentiable surrogate."""

    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SurrogateModel:
    """Wrapper that trains an MLP surrogate and exposes gradient computation."""

    def __init__(self, input_dim: int | None = None, num_classes: int = 3, device: str | None = None, model_path: Path | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.model: MLP | None = None
        if model_path and model_path.exists():
            self.load(model_path)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 40, batch_size: int = 256, lr: float = 1e-3) -> dict[str, float]:
        """Train the surrogate on processed tabular features."""
        self.input_dim = X.shape[1]
        self.model = MLP(self.input_dim, self.num_classes).to(self.device)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        history = {"final_loss": 0.0}
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            scheduler.step()
            history["final_loss"] = total_loss / max(len(loader), 1)
            if (epoch + 1) % 10 == 0:
                print(f"  Surrogate epoch {epoch + 1}/{epochs} loss={history['final_loss']:.4f}")
        return history

    def get_gradients(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Return input gradients of the loss with respect to features."""
        if self.model is None:
            raise RuntimeError("Surrogate model has not been trained or loaded.")
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True, device=self.device)
        y_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        logits = self.model(X_tensor)
        loss = nn.CrossEntropyLoss()(logits, y_tensor)
        loss.backward()
        return X_tensor.grad.detach().cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from the surrogate."""
        if self.model is None:
            raise RuntimeError("Surrogate model has not been trained or loaded.")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model(X_tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        """Persist the surrogate checkpoint."""
        if self.model is None:
            raise RuntimeError("Surrogate model has not been trained.")
        torch.save({"state_dict": self.model.state_dict(), "input_dim": self.input_dim}, path)

    def load(self, path: Path) -> None:
        """Load a saved surrogate checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = int(checkpoint["input_dim"])
        self.model = MLP(self.input_dim, self.num_classes).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
