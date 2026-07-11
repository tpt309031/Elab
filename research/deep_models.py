from __future__ import annotations

import copy
import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchSequenceClassifier(ClassifierMixin, BaseEstimator):
    """Small CPU sequence candidate with strict chronological early stopping."""

    def __init__(
        self,
        architecture: str = "lstm",
        lookback: int = 14,
        input_features: int = 20,
        hidden_size: int = 32,
        epochs: int = 18,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> None:
        self.architecture = architecture
        self.lookback = lookback
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state

    def _prepare_fit(self, values: object) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        self.median_ = np.nanmedian(array, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        array = np.where(np.isfinite(array), array, self.median_)
        self.mean_ = array.mean(axis=0)
        self.std_ = array.std(axis=0)
        self.std_ = np.where(self.std_ > 1e-6, self.std_, 1.0)
        return ((array - self.mean_) / self.std_).reshape(-1, self.lookback, self.input_features)

    def _prepare(self, values: object) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        array = np.where(np.isfinite(array), array, self.median_)
        return ((array - self.mean_) / self.std_).reshape(-1, self.lookback, self.input_features)

    def fit(self, values: object, labels: object) -> "TorchSequenceClassifier":
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for sequence candidates") from exc

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        x = self._prepare_fit(values)
        y = np.asarray(labels, dtype=np.int64)
        if x.shape[1:] != (self.lookback, self.input_features):
            raise ValueError("Sequence feature shape does not match lookback and input_features")
        validation_size = max(45, int(len(x) * 0.15))
        if len(x) <= validation_size + 120:
            raise ValueError("Not enough chronological samples for sequence training")
        x_train, x_validation = x[:-validation_size], x[-validation_size:]
        y_train, y_validation = y[:-validation_size], y[-validation_size:]

        class LstmNetwork(nn.Module):
            def __init__(self, input_features: int, hidden_size: int) -> None:
                super().__init__()
                self.encoder = nn.LSTM(input_features, hidden_size, batch_first=True)
                self.norm = nn.LayerNorm(hidden_size)
                self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(hidden_size, 3))

            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                encoded, _ = self.encoder(batch)
                return self.head(self.norm(encoded[:, -1]))

        class TransformerNetwork(nn.Module):
            def __init__(self, input_features: int, hidden_size: int, lookback: int) -> None:
                super().__init__()
                width = max(24, int(math.ceil(hidden_size / 4) * 4))
                self.project = nn.Linear(input_features, width)
                self.position = nn.Parameter(torch.zeros(1, lookback, width))
                layer = nn.TransformerEncoderLayer(
                    d_model=width,
                    nhead=4,
                    dim_feedforward=width * 2,
                    dropout=0.15,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.norm = nn.LayerNorm(width)
                self.head = nn.Linear(width, 3)

            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                encoded = self.encoder(self.project(batch) + self.position)
                return self.head(self.norm(encoded.mean(dim=1)))

        if self.architecture == "transformer":
            model = TransformerNetwork(self.input_features, self.hidden_size, self.lookback)
        else:
            model = LstmNetwork(self.input_features, self.hidden_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.002)
        counts = np.bincount(y_train, minlength=3).astype(float)
        weights = counts.sum() / np.maximum(counts, 1)
        weights = weights / weights.mean()
        loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        train_x = torch.tensor(x_train, dtype=torch.float32)
        train_y = torch.tensor(y_train, dtype=torch.long)
        validation_x = torch.tensor(x_validation, dtype=torch.float32)
        validation_y = torch.tensor(y_validation, dtype=torch.long)
        best_state = copy.deepcopy(model.state_dict())
        best_loss = math.inf
        stale_epochs = 0
        for _ in range(self.epochs):
            model.train()
            order = torch.randperm(len(train_x))
            for start in range(0, len(order), self.batch_size):
                batch_indices = order[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(train_x[batch_indices]), train_y[batch_indices])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(loss_function(model(validation_x), validation_y).item())
            if validation_loss < best_loss - 1e-4:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= 4:
                break
        model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.classes_ = np.array([0, 1, 2], dtype=int)
        self.validation_loss_ = best_loss
        return self

    def predict_proba(self, values: object) -> np.ndarray:
        import torch

        x = torch.tensor(self._prepare(values), dtype=torch.float32)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size * 2):
                logits = self.model_(x[start : start + self.batch_size * 2])
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outputs, axis=0)
