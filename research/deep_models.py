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
        device: str = "auto",
    ) -> None:
        self.architecture = architecture
        self.lookback = lookback
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        if self.device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = self.device
        self.device_ = torch.device(selected_device)
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
                self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
                self.norm = nn.LayerNorm(width)
                self.head = nn.Linear(width, 3)

            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                encoded = self.encoder(self.project(batch) + self.position)
                return self.head(self.norm(encoded.mean(dim=1)))

        class TemporalConvolutionNetwork(nn.Module):
            def __init__(self, input_features: int, hidden_size: int) -> None:
                super().__init__()
                width = max(24, hidden_size)
                layers: list[nn.Module] = []
                channels = input_features
                for dilation in (1, 2, 4):
                    layers.extend([
                        nn.Conv1d(channels, width, kernel_size=3, padding=dilation, dilation=dilation),
                        nn.GELU(),
                        nn.BatchNorm1d(width),
                        nn.Dropout(0.12),
                    ])
                    channels = width
                self.encoder = nn.Sequential(*layers)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.head = nn.Linear(width, 3)

            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                encoded = self.encoder(batch.transpose(1, 2))
                return self.head(self.pool(encoded).squeeze(-1))

        class PatchTransformerNetwork(nn.Module):
            def __init__(self, input_features: int, hidden_size: int, lookback: int) -> None:
                super().__init__()
                width = max(24, int(math.ceil(hidden_size / 4) * 4))
                patch_size = max(4, min(8, lookback // 5))
                stride = max(2, patch_size // 2)
                patch_count = max(1, (lookback - patch_size) // stride + 1)
                self.patch = nn.Conv1d(input_features, width, kernel_size=patch_size, stride=stride)
                self.position = nn.Parameter(torch.zeros(1, patch_count, width))
                layer = nn.TransformerEncoderLayer(
                    d_model=width,
                    nhead=4,
                    dim_feedforward=width * 2,
                    dropout=0.12,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
                self.norm = nn.LayerNorm(width)
                self.head = nn.Linear(width, 3)

            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                tokens = self.patch(batch.transpose(1, 2)).transpose(1, 2)
                encoded = self.encoder(tokens + self.position[:, : tokens.shape[1]])
                return self.head(self.norm(encoded.mean(dim=1)))

        class CompactTftNetwork(nn.Module):
            def __init__(self, input_features: int, hidden_size: int) -> None:
                super().__init__()
                width = max(24, int(math.ceil(hidden_size / 4) * 4))
                self.variable_gate = nn.Linear(input_features, input_features)
                self.project = nn.Linear(input_features, width)
                self.temporal = nn.LSTM(width, width, batch_first=True)
                self.attention = nn.MultiheadAttention(width, num_heads=4, dropout=0.12, batch_first=True)
                self.gated_residual = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(0.12))
                self.norm = nn.LayerNorm(width)
                self.head = nn.Linear(width, 3)

            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                gates = torch.softmax(self.variable_gate(batch), dim=-1)
                projected = self.project(batch * gates)
                temporal, _ = self.temporal(projected)
                attended, _ = self.attention(temporal, temporal, temporal, need_weights=False)
                encoded = self.norm(temporal + attended + self.gated_residual(attended))
                return self.head(encoded[:, -1])

        class InvertedTransformerNetwork(nn.Module):
            def __init__(self, input_features: int, hidden_size: int, lookback: int) -> None:
                super().__init__()
                width = max(24, int(math.ceil(hidden_size / 4) * 4))
                self.temporal_projection = nn.Linear(lookback, width)
                self.variable_embedding = nn.Parameter(torch.zeros(1, input_features, width))
                layer = nn.TransformerEncoderLayer(
                    d_model=width,
                    nhead=4,
                    dim_feedforward=width * 2,
                    dropout=0.12,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
                self.norm = nn.LayerNorm(width)
                self.head = nn.Linear(width, 3)

            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                tokens = self.temporal_projection(batch.transpose(1, 2)) + self.variable_embedding
                encoded = self.encoder(tokens)
                return self.head(self.norm(encoded.mean(dim=1)))

        architectures = {
            "lstm": lambda: LstmNetwork(self.input_features, self.hidden_size),
            "transformer": lambda: TransformerNetwork(self.input_features, self.hidden_size, self.lookback),
            "tcn": lambda: TemporalConvolutionNetwork(self.input_features, self.hidden_size),
            "patchtst": lambda: PatchTransformerNetwork(self.input_features, self.hidden_size, self.lookback),
            "tft": lambda: CompactTftNetwork(self.input_features, self.hidden_size),
            "itransformer": lambda: InvertedTransformerNetwork(self.input_features, self.hidden_size, self.lookback),
        }
        if self.architecture not in architectures:
            raise ValueError(f"Unsupported sequence architecture: {self.architecture}")
        model = architectures[self.architecture]().to(self.device_)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.002)
        counts = np.bincount(y_train, minlength=3).astype(float)
        weights = counts.sum() / np.maximum(counts, 1)
        weights = weights / weights.mean()
        loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device_),
        )
        train_x = torch.tensor(x_train, dtype=torch.float32, device=self.device_)
        train_y = torch.tensor(y_train, dtype=torch.long, device=self.device_)
        validation_x = torch.tensor(x_validation, dtype=torch.float32, device=self.device_)
        validation_y = torch.tensor(y_validation, dtype=torch.long, device=self.device_)
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

        x = torch.tensor(self._prepare(values), dtype=torch.float32, device=self.device_)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size * 2):
                logits = self.model_(x[start : start + self.batch_size * 2])
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outputs, axis=0)
