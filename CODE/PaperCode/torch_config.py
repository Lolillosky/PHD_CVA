import torch


def resolve_device_dtype(device=None, dtype=None):
    resolved_device = torch.device("cpu" if device is None else device)

    if dtype is None:
        dtype = torch.float32 if resolved_device.type == "mps" else torch.float64

    if resolved_device.type == "mps" and dtype == torch.float64:
        raise ValueError("MPS does not support torch.float64. Use torch.float32 instead.")

    return resolved_device, dtype


def standard_normal_cdf(x):
    sqrt_two = torch.sqrt(torch.as_tensor(2.0, device=x.device, dtype=x.dtype))
    return 0.5 * (1.0 + torch.erf(x / sqrt_two))
