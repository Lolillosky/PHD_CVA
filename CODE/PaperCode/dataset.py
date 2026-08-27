import numpy as np
import torch
from torch.utils.data import Dataset

class DeepLearningCVADataset(Dataset):
    
    def __init__(
        self,
        x_file: str,
        y_file: str,
        dtype: torch.dtype = torch.float64,
        eps: float = 1e-8,
        x_mean = None,
        x_std = None,
        y_mean = None,
        y_std = None
    ):

        
        X = np.load(x_file)
        y = np.load(y_file)

        if x_mean is not None and x_std is not None and y_mean is not None and y_std is not None:
            self.X_mu = x_mean
            self.X_sigma = x_std
            self.y_mu = y_mean
            self.y_sigma = y_std
        elif x_mean is not None or x_std is not None or y_mean is not None or y_std is not None:
            raise ValueError("If one of x_mean, x_std, y_mean, y_std is provided, all must be provided.")
        else:
            self.X_mu = np.mean(X, axis=0)
            self.X_sigma = np.std(X, axis=0) + eps
            self.y_mu = np.mean(y[:, 0])
            self.y_sigma = np.std(y[:, 0]) + eps

        if X.ndim != 3:
            raise ValueError(f"X must be 3D, got shape {X.shape}")
        if y.ndim != 2:
            raise ValueError(f"y must be 2D, got shape {y.shape}")
        if y.shape[0] != X.shape[0] or y.shape[1] != X.shape[1]:
            raise ValueError(
                f"X and y must have the same number of rows and columns, got {X.shape} and {y.shape}"
            )
    
        self.dtype = dtype
        self.num_inputs = X.shape[1]



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

