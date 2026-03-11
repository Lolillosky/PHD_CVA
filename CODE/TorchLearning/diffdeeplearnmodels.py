import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DenseModel(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_hidden_layers: int,
        num_neurons_per_hidden_layer: int,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()

        layers = []
        in_features = num_inputs

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, num_neurons_per_hidden_layer, dtype=dtype))
            layers.append(nn.Softplus())
            in_features = num_neurons_per_hidden_layer

        layers.append(nn.Linear(in_features, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffDeepLearning(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We need gradients w.r.t. inputs because the model outputs [y, dy/dx]
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        y = self.model(x)  # shape: (B, 1)

        grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=self.training,   # needed during training for derivative loss
            retain_graph=True,
            only_inputs=True,
        )[0]

        return torch.cat([y, grad], dim=1)


class DiffDeepLearningDataset(Dataset):
    """
    Expected:
      X.shape == (N, D)
      y.shape == (N, D + 1)

    where:
      y[:, 0]   = scalar target
      y[:, 1:]  = sensitivities dy/dx
    """

    def __init__(
        self,
        x_file: str,
        y_file: str,
        dtype: torch.dtype = torch.float64,
        eps: float = 1e-8,
    ):
        X = np.load(x_file)
        y = np.load(y_file)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 2:
            raise ValueError(f"y must be 2D, got shape {y.shape}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows, got {X.shape[0]} and {y.shape[0]}"
            )
        if y.shape[1] != X.shape[1] + 1:
            raise ValueError(
                f"y must have shape (N, D+1). Got X.shape={X.shape}, y.shape={y.shape}"
            )

        self.dtype = dtype
        self.num_inputs = X.shape[1]

        # Normalization stats
        self.X_mu = np.mean(X, axis=0)
        self.X_sigma = np.std(X, axis=0) + eps

        self.y_mu = np.mean(y[:, 0])
        self.y_sigma = np.std(y[:, 0]) + eps

        # Copy target array so original input is untouched
        y_scaled = y.copy()

        # Scale sensitivities to match normalized variables:
        # x_scaled = (x - X_mu) / X_sigma
        # y_scaled = (y - y_mu) / y_sigma
        # => d(y_scaled)/d(x_scaled) = dy/dx * X_sigma / y_sigma
        y_scaled[:, 1:] *= self.X_sigma / self.y_sigma

        # Per-dimension normalization for derivative loss
        self.dydX_scaled_L2_norm = np.mean(y_scaled[:, 1:] ** 2, axis=0) + eps

        # Scale scalar target
        y_scaled[:, 0] = (y_scaled[:, 0] - self.y_mu) / self.y_sigma

        # Store tensors
        self.X = torch.tensor((X - self.X_mu) / self.X_sigma, dtype=dtype)
        self.y = torch.tensor(y_scaled, dtype=dtype)

        self.len = self.X.shape[0]

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.len


class DiffLearningLoss(nn.Module):
    def __init__(
        self,
        alpha: float,
        dydX_scaled_L2_norm,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()

        self.alpha = alpha
        norm_t = torch.as_tensor(dydX_scaled_L2_norm, dtype=dtype)
        self.register_buffer("dydX_scaled_L2_norm", norm_t)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        value_loss = torch.nn.functional.mse_loss(pred[:, 0], target[:, 0])

        deriv_loss = torch.mean(
            (pred[:, 1:] - target[:, 1:]) ** 2 / self.dydX_scaled_L2_norm
        )

        return value_loss + self.alpha * deriv_loss

import torch


class DiffLearningFullModel:
    def __init__(
        self,
        num_inputs: int,
        num_hidden_layers: int,
        num_neurons_per_hidden_layer: int,
        alpha: float,
        train_dataloader=None,
        val_dataloader=None,
        lr: float = 1e-3,
        dtype: torch.dtype = torch.float64,
        device=None,
    ):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_hidden_layer = num_neurons_per_hidden_layer
        self.lr = lr

        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        base_model = DenseModel(
            num_inputs=num_inputs,
            num_hidden_layers=num_hidden_layers,
            num_neurons_per_hidden_layer=num_neurons_per_hidden_layer,
            dtype=dtype,
        )
        self.model = DiffDeepLearning(base_model).to(self.device)

        self.alpha = alpha
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # These will be filled either from dataloader or from checkpoint load
        self.X_mu = None
        self.X_sigma = None
        self.y_mu = None
        self.y_sigma = None
        self.dydX_scaled_L2_norm = None

        self.X_mu_t = None
        self.X_sigma_t = None
        self.y_mu_t = None
        self.y_sigma_t = None

        self.loss_fn = None

        if self.train_dataloader is not None:
            self._get_normalization_params_and_set_loss()

    def _get_normalization_params_and_set_loss(self):
        dataset = self.train_dataloader.dataset

        # NumPy / python-side values
        self.X_mu = dataset.X_mu
        self.X_sigma = dataset.X_sigma
        self.y_mu = dataset.y_mu
        self.y_sigma = dataset.y_sigma
        self.dydX_scaled_L2_norm = dataset.dydX_scaled_L2_norm

        # Tensor values on correct device
        self.X_mu_t = torch.as_tensor(self.X_mu, dtype=self.dtype, device=self.device)
        self.X_sigma_t = torch.as_tensor(self.X_sigma, dtype=self.dtype, device=self.device)
        self.y_mu_t = torch.as_tensor(self.y_mu, dtype=self.dtype, device=self.device)
        self.y_sigma_t = torch.as_tensor(self.y_sigma, dtype=self.dtype, device=self.device)

        self.loss_fn = DiffLearningLoss(
            alpha=self.alpha,
            dydX_scaled_L2_norm=self.dydX_scaled_L2_norm,
            dtype=self.dtype,
        ).to(self.device)

    def _ensure_ready_for_inference(self):
        required = [
            self.X_mu,
            self.X_sigma,
            self.y_mu,
            self.y_sigma,
            self.X_mu_t,
            self.X_sigma_t,
            self.y_mu_t,
            self.y_sigma_t,
        ]
        if any(v is None for v in required):
            raise RuntimeError(
                "Model normalization parameters are not initialized. "
                "Train the model first or load a checkpoint with DiffLearningFullModel.load(...)."
            )

    def _ensure_ready_for_training(self):
        if self.train_dataloader is None:
            raise RuntimeError("train_dataloader is None. Cannot train without a training dataloader.")
        if self.loss_fn is None:
            raise RuntimeError(
                "loss_fn is not initialized. Make sure normalization parameters were loaded or computed."
            )

    def fit(self, epochs: int, writer=None):
        self._ensure_ready_for_training()

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for X, y in self.train_dataloader:
                X = X.to(self.device, dtype=self.dtype).detach().requires_grad_(True)
                y = y.to(self.device, dtype=self.dtype)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_dataloader)

            val_loss = None
            if self.val_dataloader is not None:
                self.model.eval()
                val_loss = 0.0

                # Keep grad enabled because derivative outputs may rely on autograd
                with torch.set_grad_enabled(True):
                    for X, y in self.val_dataloader:
                        X = X.to(self.device, dtype=self.dtype).detach().requires_grad_(True)
                        y = y.to(self.device, dtype=self.dtype)

                        pred = self.model(X)
                        loss = self.loss_fn(pred, y)
                        val_loss += loss.item()

                val_loss /= len(self.val_dataloader)

            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                if val_loss is not None:
                    writer.add_scalar("Loss/val", val_loss, epoch)

            if val_loss is not None:
                print(
                    f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )
            else:
                print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}")

    def predict(self, X):
        """
        Returns predictions in original units.

        Input:
          X: np.ndarray or torch.Tensor of shape (N, D)

        Output:
          {
            "y":    np.ndarray of shape (N,),
            "sens": np.ndarray of shape (N, D)
          }
        """
        self._ensure_ready_for_inference()
        self.model.eval()

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        else:
            X = X.to(self.device, dtype=self.dtype)

        if X.ndim == 1:
            X = X.unsqueeze(0)

        X_scaled = ((X - self.X_mu_t) / self.X_sigma_t).detach().requires_grad_(True)

        with torch.set_grad_enabled(True):
            pred_scaled = self.model(X_scaled)

        pred_scaled = pred_scaled.detach().cpu().numpy()

        # Undo scaling:
        # y = y_mu + y_scaled * y_sigma
        # dy/dx = d(y_scaled)/d(x_scaled) * y_sigma / X_sigma
        y = self.y_mu + pred_scaled[:, 0] * self.y_sigma
        sens = pred_scaled[:, 1:] * self.y_sigma / self.X_sigma

        return {"y": y, "sens": sens}

    def predict_tensor(self, X: torch.Tensor) -> dict:
        """
        Same as predict(), but returns torch tensors on the model device.
        """
        self._ensure_ready_for_inference()
        self.model.eval()

        X = X.to(self.device, dtype=self.dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)

        X_scaled = ((X - self.X_mu_t) / self.X_sigma_t).detach().requires_grad_(True)

        with torch.set_grad_enabled(True):
            pred_scaled = self.model(X_scaled)

        y = self.y_mu_t + pred_scaled[:, 0] * self.y_sigma_t
        sens = pred_scaled[:, 1:] * self.y_sigma_t / self.X_sigma_t

        return {"y": y, "sens": sens}

    def save(self, path: str):
        """
        Save full checkpoint, including:
        - model weights
        - optimizer state
        - architecture
        - normalization params
        - training hyperparameters
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_inputs": self.num_inputs,
            "num_hidden_layers": self.num_hidden_layers,
            "num_neurons_per_hidden_layer": self.num_neurons_per_hidden_layer,
            "lr": self.lr,
            "alpha": self.alpha,
            "dtype": "float64" if self.dtype == torch.float64 else "float32",
            "X_mu": self.X_mu,
            "X_sigma": self.X_sigma,
            "y_mu": self.y_mu,
            "y_sigma": self.y_sigma,
            "dydX_scaled_L2_norm": self.dydX_scaled_L2_norm,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device=None, load_optimizer: bool = True):
        device = torch.device(device) if device is not None else torch.device("cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        dtype_str = checkpoint.get("dtype", "float64")
        if dtype_str == "float64":
            dtype = torch.float64
        elif dtype_str == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype in checkpoint: {dtype_str}")

        obj = cls(
            num_inputs=checkpoint["num_inputs"],
            num_hidden_layers=checkpoint["num_hidden_layers"],
            num_neurons_per_hidden_layer=checkpoint["num_neurons_per_hidden_layer"],
            alpha=checkpoint["alpha"],
            train_dataloader=None,
            val_dataloader=None,
            lr=checkpoint.get("lr", 1e-3),
            dtype=dtype,
            device=device,
        )

        obj.model.load_state_dict(checkpoint["model_state_dict"])
        obj.model.to(device)

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            obj.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        obj.X_mu = checkpoint["X_mu"]
        obj.X_sigma = checkpoint["X_sigma"]
        obj.y_mu = checkpoint["y_mu"]
        obj.y_sigma = checkpoint["y_sigma"]
        obj.dydX_scaled_L2_norm = checkpoint["dydX_scaled_L2_norm"]

        obj.X_mu_t = torch.as_tensor(obj.X_mu, dtype=obj.dtype, device=obj.device)
        obj.X_sigma_t = torch.as_tensor(obj.X_sigma, dtype=obj.dtype, device=obj.device)
        obj.y_mu_t = torch.as_tensor(obj.y_mu, dtype=obj.dtype, device=obj.device)
        obj.y_sigma_t = torch.as_tensor(obj.y_sigma, dtype=obj.dtype, device=obj.device)

        obj.loss_fn = DiffLearningLoss(
            alpha=obj.alpha,
            dydX_scaled_L2_norm=obj.dydX_scaled_L2_norm,
            dtype=obj.dtype,
        ).to(obj.device)

        return obj