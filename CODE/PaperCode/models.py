import torch
import torch.nn as nn
import enums
from torch_config import resolve_device_dtype
import lightning.pytorch as pl



class CCRDeepModel(nn.Module):
    def __init__(self, number_risk_factors, num_rnn_layers, num_rnn_hidden_units, rnn_type, 
                 num_deep_layers, deep_hidden_units):

        super(CCRDeepModel, self).__init__()

        self.rnn_type = rnn_type
        if num_rnn_layers < 0:
            raise ValueError("num_rnn_layers must be non-negative")
        if num_deep_layers < 0:
            raise ValueError("num_deep_layers must be non-negative")

        # If num_rnn_layers is zero, skip the recurrent block.
        if num_rnn_layers == 0:
            self.rnn = None
            input_size = number_risk_factors
        elif rnn_type == enums.RNNType.GRU:
            self.rnn = nn.GRU(input_size=number_risk_factors, hidden_size=num_rnn_hidden_units, num_layers=num_rnn_layers, batch_first=True)
            input_size = num_rnn_hidden_units
        elif rnn_type == enums.RNNType.LSTM:
            self.rnn = nn.LSTM(input_size=number_risk_factors, hidden_size=num_rnn_hidden_units, num_layers=num_rnn_layers, batch_first=True)
            input_size = num_rnn_hidden_units
        else:
            raise ValueError("Unsupported rnn_type: {}. Use RNNType.GRU or RNNType.LSTM.".format(rnn_type))

        # Feed-forward head maps each timestep feature vector to a scalar CVA value.
        deep_layers = []
        for _ in range(num_deep_layers):
            deep_layers.append(nn.Linear(input_size, deep_hidden_units))
            deep_layers.append(nn.Softplus())
            input_size = deep_hidden_units
        self.deep = nn.Sequential(*deep_layers)

        self.output_layer = nn.Linear(input_size, 1)

    def forward(self, x):
        # x shape: (batch_size, time_steps, number_risk_factors)
        if self.rnn is None:
            sequence_features = x
        else:
            sequence_features, _ = self.rnn(x)

        # Keep every timestep: this is many-to-many, not many-to-one.
        deep_out = self.deep(sequence_features)
        output = self.output_layer(deep_out)
        # output shape: (batch_size, time_steps), matching the dataset target y.
        return output.squeeze(-1)


class CCRDeepModelTrainer:
    def __init__(
        self,
        number_risk_factors,
        num_rnn_layers,
        num_rnn_hidden_units,
        rnn_type,
        num_deep_layers,
        deep_hidden_units,
        train_dataloader=None,
        val_dataloader=None,
        lr: float = 1e-3,
        loss_fn=None,
        device=None,
        dtype=None,
    ):
        self.device, self.dtype = resolve_device_dtype(device, dtype)

        self.number_risk_factors = number_risk_factors
        self.num_rnn_layers = num_rnn_layers
        self.num_rnn_hidden_units = num_rnn_hidden_units
        self.rnn_type = rnn_type
        self.num_deep_layers = num_deep_layers
        self.deep_hidden_units = deep_hidden_units
        self.lr = lr

        self.model = CCRDeepModel(
            number_risk_factors=number_risk_factors,
            num_rnn_layers=num_rnn_layers,
            num_rnn_hidden_units=num_rnn_hidden_units,
            rnn_type=rnn_type,
            num_deep_layers=num_deep_layers,
            deep_hidden_units=deep_hidden_units,
        ).to(device=self.device, dtype=self.dtype)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.history = []
        self.epoch = 0
        self.best_val_loss = None

        self.X_mu = None
        self.X_sigma = None
        self.y_mu = None
        self.y_sigma = None
        self._set_normalization_params_from_dataloader(train_dataloader)

    def _set_normalization_params_from_dataloader(self, dataloader):
        if dataloader is None or not hasattr(dataloader, "dataset"):
            return

        dataset = dataloader.dataset
        while dataset is not None and not any(
            hasattr(dataset, attr) for attr in ("X_mu", "X_sigma", "y_mu", "y_sigma")
        ):
            dataset = getattr(dataset, "dataset", None)

        if dataset is None:
            return

        self.X_mu = getattr(dataset, "X_mu", None)
        self.X_sigma = getattr(dataset, "X_sigma", None)
        self.y_mu = getattr(dataset, "y_mu", None)
        self.y_sigma = getattr(dataset, "y_sigma", None)

    def _ensure_ready_for_training(self):
        if self.train_dataloader is None:
            raise RuntimeError("train_dataloader is None. Cannot train without a training dataloader.")
        if len(self.train_dataloader) == 0:
            raise RuntimeError("train_dataloader is empty. Cannot train without batches.")

    def _prepare_batch(self, X, y):
        X = X.to(self.device, dtype=self.dtype)
        y = y.to(self.device, dtype=self.dtype)
        return X, y

    def _run_epoch(self):
        self.model.train()
        train_loss = 0.0

        for X, y in self.train_dataloader:
            X, y = self._prepare_batch(X, y)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(self.train_dataloader)

    def _validate(self):
        if self.val_dataloader is None:
            return None
        if len(self.val_dataloader) == 0:
            return None

        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, y in self.val_dataloader:
                X, y = self._prepare_batch(X, y)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                val_loss += loss.item()

        return val_loss / len(self.val_dataloader)

    def fit(self, epochs: int, writer=None, print_progress: bool = True):
        self._ensure_ready_for_training()

        for _ in range(epochs):
            train_loss = self._run_epoch()
            val_loss = self._validate()

            self.epoch += 1
            epoch_history = {
                "epoch": self.epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            self.history.append(epoch_history)

            if val_loss is not None:
                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, self.epoch)
                if val_loss is not None:
                    writer.add_scalar("Loss/val", val_loss, self.epoch)

            if print_progress:
                if val_loss is not None:
                    print(
                        f"Epoch {self.epoch}, Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}"
                    )
                else:
                    print(f"Epoch {self.epoch}, Train Loss: {train_loss:.6f}")

        return self.history

    def save(self, path: str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "number_risk_factors": self.number_risk_factors,
            "num_rnn_layers": self.num_rnn_layers,
            "num_rnn_hidden_units": self.num_rnn_hidden_units,
            "rnn_type": self.rnn_type.name,
            "num_deep_layers": self.num_deep_layers,
            "deep_hidden_units": self.deep_hidden_units,
            "lr": self.lr,
            "dtype": self._dtype_to_string(self.dtype),
            "device": str(self.device),
            "epoch": self.epoch,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "X_mu": self.X_mu,
            "X_sigma": self.X_sigma,
            "y_mu": self.y_mu,
            "y_sigma": self.y_sigma,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str,
        train_dataloader=None,
        val_dataloader=None,
        device=None,
        load_optimizer: bool = True,
    ):
        checkpoint_device = torch.device(device) if device is not None else torch.device("cpu")
        checkpoint = torch.load(path, map_location=checkpoint_device, weights_only=False)

        dtype = cls._string_to_dtype(checkpoint.get("dtype", "float64"))
        rnn_type = enums.RNNType[checkpoint["rnn_type"]]

        obj = cls(
            number_risk_factors=checkpoint["number_risk_factors"],
            num_rnn_layers=checkpoint["num_rnn_layers"],
            num_rnn_hidden_units=checkpoint["num_rnn_hidden_units"],
            rnn_type=rnn_type,
            num_deep_layers=checkpoint["num_deep_layers"],
            deep_hidden_units=checkpoint["deep_hidden_units"],
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            lr=checkpoint.get("lr", 1e-3),
            device=checkpoint_device,
            dtype=dtype,
        )

        obj.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            obj.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        obj.epoch = checkpoint.get("epoch", 0)
        obj.history = checkpoint.get("history", [])
        obj.best_val_loss = checkpoint.get("best_val_loss")

        obj.X_mu = checkpoint.get("X_mu")
        obj.X_sigma = checkpoint.get("X_sigma")
        obj.y_mu = checkpoint.get("y_mu")
        obj.y_sigma = checkpoint.get("y_sigma")

        return obj

    @staticmethod
    def _dtype_to_string(dtype):
        if dtype == torch.float64:
            return "float64"
        if dtype == torch.float32:
            return "float32"
        raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def _string_to_dtype(dtype_str):
        if dtype_str == "float64":
            return torch.float64
        if dtype_str == "float32":
            return torch.float32
        raise ValueError(f"Unsupported dtype in checkpoint: {dtype_str}")



class LitGenericModel(pl.LightningModule):
    def __init__(self, model, loss, scorelist=None, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss", "scorelist"])

        self.model = model
        self.loss = loss
        self.scorelist = [] if scorelist is None else scorelist

    def forward(self, x):
        return self.model(x)

    def _model_dtype(self):
        parameter = next(self.parameters(), None)
        return None if parameter is None else parameter.dtype

    def _cast_floating_tensors(self, data, dtype):
        if dtype is None:
            return data
        if torch.is_tensor(data):
            return data.to(dtype=dtype) if torch.is_floating_point(data) else data
        if isinstance(data, tuple) and hasattr(data, "_fields"):
            return type(data)(*(self._cast_floating_tensors(item, dtype) for item in data))
        if isinstance(data, tuple):
            return tuple(self._cast_floating_tensors(item, dtype) for item in data)
        if isinstance(data, list):
            return [self._cast_floating_tensors(item, dtype) for item in data]
        if isinstance(data, dict):
            return {key: self._cast_floating_tensors(value, dtype) for key, value in data.items()}
        return data

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return self._cast_floating_tensors(batch, self._model_dtype())

    def _shared_step(self, batch):
        xs, ys = batch
        ys_estimate = self(xs)
        loss = self.loss(ys_estimate, ys)

        score = []

        for score_fn in self.scorelist:
            score.append(score_fn(ys_estimate, ys))
  
        return loss, score, ys_estimate, ys

    def training_step(self, batch, batch_idx):
        loss, score, ys_estimate, ys = self._shared_step(batch)
        self.log('loss.train', loss, on_epoch=True, on_step=False, prog_bar=True)
        for i, s in enumerate(score):
            self.log(f'score{i}.train', s, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, score, ys_estimate, ys = self._shared_step(batch)
        self.log('loss.valid', loss, on_epoch=True, on_step=False, prog_bar=True)
        for i, s in enumerate(score):
            self.log(f'score{i}.valid', s, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, score, ys_estimate, ys = self._shared_step(batch)
        self.log('loss.test', loss, on_epoch=True, on_step=False, prog_bar=True)
        for i, s in enumerate(score):
            self.log(f'score{i}.test', s, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        xs = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self(xs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
