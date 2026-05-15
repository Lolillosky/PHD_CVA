import torch


def basket_geom_asian_cashflows(
    init_time_array,
    risk_free_rate,
    num_assets,
    price_history,
    IsCall,
    strike=1.0,
    device=None,
    dtype=torch.float64,
    keep_feature_dim=True,
):
    """
    Generate discounted realized cash flows for the geometric basket Asian option.

    The path layout is RNN-style: (simulations, time, assets). The returned cash
    flows are zero at all non-terminal dates and contain the discounted terminal
    payoff at the final date.

    This payoff convention matches basket_geom_asian: prices are normalized by
    the initial fixing, the t=0 fixing is excluded from the Asian product, and
    the strike defaults to 1.0.
    """

    init_time_array = torch.as_tensor(init_time_array, device=device, dtype=dtype)
    risk_free_rate = torch.as_tensor(risk_free_rate, device=device, dtype=dtype)
    price_history = torch.as_tensor(price_history, device=device, dtype=dtype)
    strike = torch.as_tensor(strike, device=price_history.device, dtype=price_history.dtype)

    if price_history.ndim != 3:
        raise ValueError("price_history must have shape (simulations, time, assets)")

    if price_history.shape[2] != num_assets:
        raise ValueError("num_assets does not match price_history.shape[2]")

    if price_history.shape[1] != init_time_array.numel():
        raise ValueError("price_history time dimension must match len(init_time_array)")

    if init_time_array.numel() < 2:
        raise ValueError("init_time_array must contain at least two dates")

    relative_fixings = price_history[:, 1:, :] / price_history[:, 0:1, :]
    geom_average = torch.pow(
        torch.prod(relative_fixings.reshape(price_history.shape[0], -1), dim=1),
        1.0 / (num_assets * (init_time_array.numel() - 1)),
    )

    if IsCall:
        payoff = torch.maximum(geom_average - strike, torch.zeros_like(geom_average))
    else:
        payoff = torch.maximum(strike - geom_average, torch.zeros_like(geom_average))

    maturity = init_time_array[-1] - init_time_array[0]
    discounted_payoff = payoff * torch.exp(-risk_free_rate * maturity)

    cashflows = torch.zeros(
        price_history.shape[0],
        price_history.shape[1],
        device=price_history.device,
        dtype=price_history.dtype,
    )
    cashflows[:, -1] = discounted_payoff

    if keep_feature_dim:
        return cashflows.unsqueeze(-1)

    return cashflows
