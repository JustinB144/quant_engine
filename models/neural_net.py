"""
Tabular Neural Network — feedforward network for tabular financial data.

Experimental — only use after tree-based models show signal.
Requires PyTorch.

Architecture:
    - Input layer with batch normalisation
    - Hidden layers with ReLU, batch norm, and dropout
    - Linear output layer (regression)

If PyTorch is not installed, the class raises NotImplementedError on
fit/predict so importing the module never fails.
"""
from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


class TabularNet:
    """Feedforward network for tabular financial data.

    Experimental — only use after tree-based models show signal.
    Requires PyTorch.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 50,
    ):
        """Initialize TabularNet."""
        if hidden_dims is None:
            hidden_dims = [64, 32]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self._model: Optional[object] = None
        self._fitted = False

        if _TORCH_AVAILABLE:
            self._build_model()

    def _build_model(self) -> None:
        """Construct the PyTorch Sequential model."""
        layers: list = []
        in_dim = self.input_dim

        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_dim = h_dim

        # Output layer — single regression target
        layers.append(nn.Linear(in_dim, 1))

        self._model = nn.Sequential(*layers)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "TabularNet":
        """Train the network on tabular features.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        y : np.ndarray
            Target vector (n_samples,).
        sample_weight : np.ndarray, optional
            Per-sample weights for the loss function.
        verbose : bool
            Print epoch-level training loss.

        Returns
        -------
        TabularNet
            Self, for method chaining.

        Raises
        ------
        NotImplementedError
            If PyTorch is not installed.
        """
        if not _TORCH_AVAILABLE:
            raise NotImplementedError(
                "PyTorch is required for TabularNet. "
                "Install it with: pip install torch"
            )

        X_t = torch.tensor(np.asarray(X, dtype=np.float32))
        y_t = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1))

        if sample_weight is not None:
            w_t = torch.tensor(np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1))
        else:
            w_t = torch.ones_like(y_t)

        dataset = TensorDataset(X_t, y_t, w_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction="none")

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch, w_batch in loader:
                optimizer.zero_grad()
                pred = self._model(X_batch)
                loss_raw = criterion(pred, y_batch)
                loss = (loss_raw * w_batch).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                print(f"  TabularNet epoch {epoch+1}/{self.epochs}, loss: {avg_loss:.6f}")

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for the given features.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted values (n_samples,).

        Raises
        ------
        NotImplementedError
            If PyTorch is not installed.
        RuntimeError
            If the model has not been fitted yet.
        """
        if not _TORCH_AVAILABLE:
            raise NotImplementedError(
                "PyTorch is required for TabularNet. "
                "Install it with: pip install torch"
            )
        if not self._fitted:
            raise RuntimeError("TabularNet has not been fitted yet. Call .fit() first.")

        self._model.eval()
        X_t = torch.tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            pred = self._model(X_t).numpy().flatten()
        return pred

    @property
    def feature_importances_(self) -> np.ndarray:
        """Approximate feature importance from first-layer weights.

        Uses the absolute values of the first linear layer's weights,
        summed across output neurons, as a proxy for feature importance.
        This is a rough heuristic; use permutation importance for
        rigorous analysis.
        """
        if not _TORCH_AVAILABLE or self._model is None:
            return np.array([])
        # Find the first Linear layer
        for layer in self._model:
            if isinstance(layer, nn.Linear):
                weights = layer.weight.detach().numpy()
                return np.abs(weights).sum(axis=0)
        return np.array([])
