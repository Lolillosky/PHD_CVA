import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, spearmanr
from torch.utils.data import DataLoader

from dataset import DeepLearningCVADataset
from models import CCRDeepModelTrainer
from option_formulas import basket_geom_asian_vectorized


def find_project_root(marker="pyproject.toml"):
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / marker).exists():
            return candidate
    raise RuntimeError(f"Could not find project root containing {marker}")


def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def get_gpu_device_name():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return None


def build_dataloaders(data_dir, model_dtype, batch_size):
    train_dataset = DeepLearningCVADataset(
        str(data_dir / "X_CVA_train.npy"),
        str(data_dir / "y_CVA_train.npy"),
        dtype=model_dtype,
    )
    test_dataset = DeepLearningCVADataset(
        str(data_dir / "X_CVA_test.npy"),
        str(data_dir / "y_CVA_test.npy"),
        dtype=model_dtype,
        x_mean=train_dataset.X_mu,
        x_std=train_dataset.X_sigma,
        y_mean=train_dataset.y_mu,
        y_std=train_dataset.y_sigma,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader


def predict_nn_values(trainer, test_dataset, test_loader):
    trainer.model.eval()
    scaled_predictions = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(trainer.device, dtype=trainer.dtype)
            scaled_predictions.append(trainer.model(X_batch).detach().cpu())

    scaled_predictions = torch.cat(scaled_predictions, dim=0).numpy()
    return test_dataset.y_mu + scaled_predictions * test_dataset.y_sigma


def compute_closed_form_values(
    test_paths,
    time_steps,
    risk_free_rate,
    num_risk_factors,
    volatility_array,
    correl_matrix,
    pricing_dtype,
    is_call=True,
):
    option_values = basket_geom_asian_vectorized(
        time_steps,
        risk_free_rate,
        num_risk_factors,
        volatility_array,
        correl_matrix,
        test_paths,
        is_call,
        device="cpu",
        dtype=pricing_dtype,
        keep_feature_dim=False,
    )
    return to_numpy(option_values)


def train_and_evaluate(
    device_name,
    model_save_path,
    data_dir,
    time_steps,
    risk_free_rate,
    num_risk_factors,
    volatility_array,
    correl_matrix,
    model_kwargs,
    model_dtype,
    pricing_dtype,
    batch_size,
    epochs,
    seed,
    lr=1e-3,
    is_call=True,
    print_progress=True,
):
    torch.manual_seed(seed)
    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(
        data_dir,
        model_dtype,
        batch_size,
    )

    trainer = CCRDeepModelTrainer(
        **model_kwargs,
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        lr=lr,
        device=device_name,
        dtype=model_dtype,
    )

    start = time.perf_counter()
    history = trainer.fit(epochs, print_progress=print_progress)
    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    training_seconds = time.perf_counter() - start

    trainer.save(str(model_save_path))

    test_paths = np.load(data_dir / "X_CVA_test.npy").astype(np.float32)
    closed_form_values = compute_closed_form_values(
        test_paths,
        time_steps,
        risk_free_rate,
        num_risk_factors,
        volatility_array,
        correl_matrix,
        pricing_dtype,
        is_call=is_call,
    )
    nn_values = predict_nn_values(trainer, test_dataset, test_loader)

    expected_shape = (len(test_dataset), len(time_steps))
    if closed_form_values.shape != expected_shape:
        raise ValueError(f"closed_form_values has shape {closed_form_values.shape}, expected {expected_shape}")
    if nn_values.shape != expected_shape:
        raise ValueError(f"nn_values has shape {nn_values.shape}, expected {expected_shape}")

    print(f"{device_name.upper()} training time: {training_seconds:.2f}s")
    print(f"Saved model to {model_save_path}")

    return {
        "device": device_name,
        "trainer": trainer,
        "history": history,
        "training_seconds": training_seconds,
        "test_paths": test_paths,
        "closed_form_values": closed_form_values,
        "nn_values": nn_values,
        "model_save_path": model_save_path,
    }


def _validate_comparison_inputs(closed_form_values, nn_values):
    closed_form_values = np.asarray(closed_form_values)
    nn_values = np.asarray(nn_values)

    if closed_form_values.shape != nn_values.shape:
        raise ValueError(
            f"closed_form_values and nn_values must have the same shape, "
            f"got {closed_form_values.shape} and {nn_values.shape}"
        )
    if closed_form_values.ndim != 2:
        raise ValueError(
            f"comparison arrays must have shape (simulations, time), got {closed_form_values.shape}"
        )

    return closed_form_values, nn_values


def compute_spearman_correlation_over_time(closed_form_values, nn_values):
    closed_form_values, nn_values = _validate_comparison_inputs(closed_form_values, nn_values)
    correlations = np.empty(closed_form_values.shape[1], dtype=float)

    for time_index in range(closed_form_values.shape[1]):
        closed_slice = closed_form_values[:, time_index]
        nn_slice = nn_values[:, time_index]

        if np.all(closed_slice == closed_slice[0]) or np.all(nn_slice == nn_slice[0]):
            correlations[time_index] = np.nan
        else:
            correlations[time_index] = spearmanr(closed_slice, nn_slice).correlation

    return correlations


def compute_ks_test_over_time(closed_form_values, nn_values, time_steps=None):
    closed_form_values, nn_values = _validate_comparison_inputs(closed_form_values, nn_values)

    if time_steps is None:
        time_steps = np.arange(closed_form_values.shape[1])
    time_steps = np.asarray(time_steps)
    if time_steps.shape[0] != closed_form_values.shape[1]:
        raise ValueError(
            f"time_steps length must match time dimension, got {time_steps.shape[0]} "
            f"and {closed_form_values.shape[1]}"
        )

    rows = []
    for time_index, time_value in enumerate(time_steps):
        result = ks_2samp(
            closed_form_values[:, time_index],
            nn_values[:, time_index],
            alternative="two-sided",
            method="auto",
        )
        rows.append(
            {
                "time": time_value,
                "ks_statistic": result.statistic,
                "p_value": result.pvalue,
            }
        )

    return pd.DataFrame(rows)


def compute_bayes_error_over_time(
    discounted_cashflows,
    closed_form_values,
    y_mean,
    y_std,
    time_steps=None,
):
    discounted_cashflows, closed_form_values = _validate_comparison_inputs(
        discounted_cashflows,
        closed_form_values,
    )

    if time_steps is None:
        time_steps = np.arange(closed_form_values.shape[1])
    time_steps = np.asarray(time_steps)
    if time_steps.shape[0] != closed_form_values.shape[1]:
        raise ValueError(
            f"time_steps length must match time dimension, got {time_steps.shape[0]} "
            f"and {closed_form_values.shape[1]}"
        )

    y_mean = np.asarray(y_mean)
    y_std = np.asarray(y_std)

    normalized_cashflows = (discounted_cashflows - y_mean) / y_std
    normalized_closed_form = (closed_form_values - y_mean) / y_std
    squared_error = (normalized_cashflows - normalized_closed_form) ** 2
    bayes_mse = np.mean(squared_error, axis=0)

    return pd.DataFrame(
        {
            "time": time_steps,
            "bayes_mse": bayes_mse,
            "bayes_rmse": np.sqrt(bayes_mse),
        }
    )


def plot_percentile_evolution(result, time_steps, percentiles, title):
    closed_form_values = result["closed_form_values"]
    nn_values = result["nn_values"]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(percentiles)))

    for q, color in zip(percentiles, colors):
        closed_form_profile = np.percentile(closed_form_values, q=q, axis=0)
        nn_profile = np.percentile(nn_values, q=q, axis=0)
        ax.plot(time_steps, closed_form_profile, color=color, linestyle="-", label=f"Closed form {q}th")
        ax.plot(time_steps, nn_profile, color=color, linestyle="--", label=f"NN {q}th")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Option value")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_spearman_correlation_over_time(time_steps, correlations, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_steps, correlations, marker="o", color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Spearman rank correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_ks_test_over_time(time_steps, ks_results, title):
    fig, ax_stat = plt.subplots(figsize=(10, 5))
    ax_pvalue = ax_stat.twinx()

    ax_stat.plot(
        time_steps,
        ks_results["ks_statistic"],
        marker="o",
        color="tab:blue",
        label="KS statistic",
    )
    ax_pvalue.plot(
        time_steps,
        ks_results["p_value"],
        marker="s",
        color="tab:orange",
        label="p-value",
    )

    ax_stat.set_title(title)
    ax_stat.set_xlabel("Time")
    ax_stat.set_ylabel("KS statistic", color="tab:blue")
    ax_pvalue.set_ylabel("p-value", color="tab:orange")
    ax_stat.tick_params(axis="y", labelcolor="tab:blue")
    ax_pvalue.tick_params(axis="y", labelcolor="tab:orange")
    ax_stat.set_ylim(bottom=0.0)
    ax_pvalue.set_ylim(0.0, 1.05)
    ax_stat.grid(True, alpha=0.25)

    lines = ax_stat.get_lines() + ax_pvalue.get_lines()
    labels = [line.get_label() for line in lines]
    ax_stat.legend(lines, labels, loc="best")
    fig.tight_layout()
    return fig, (ax_stat, ax_pvalue)


def plot_bayes_error_over_time(time_steps, bayes_error, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_steps, bayes_error["bayes_mse"], marker="o", label="Bayes MSE")
    ax.plot(time_steps, bayes_error["bayes_rmse"], marker="s", label="Bayes RMSE")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized error")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_training_history(result, title):
    history = pd.DataFrame(result["history"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Train")
    ax.plot(history["epoch"], history["val_loss"], marker="o", label="Test")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax
