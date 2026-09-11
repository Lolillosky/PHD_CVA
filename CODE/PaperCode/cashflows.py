import numpy as np
import torch

import enums
from torch_config import resolve_device_dtype


def basket_geom_asian_cashflows(
    init_time_array,
    risk_free_rate,
    num_assets,
    price_history,
    IsCall,
    strike=1.0,
    device=None,
    dtype=None,
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

    device, dtype = resolve_device_dtype(device, dtype)

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

    maturities = init_time_array[-1] - init_time_array
    discounted_payoffs = payoff.unsqueeze(-1) * torch.exp(-risk_free_rate * maturities)

    cashflows = torch.zeros(
        price_history.shape[0],
        price_history.shape[1],
        device=price_history.device,
        dtype=price_history.dtype,
    )

    cashflows[:, -1] = payoff

    return_dict = {
        "cashflows": cashflows,
        "discounted_cashflows": discounted_payoffs}


    if keep_feature_dim:
        return_dict["cashflows"] = cashflows.unsqueeze(-1)
        return_dict["discounted_cashflows"] = discounted_payoffs.unsqueeze(-1)

    return return_dict


class geometric_basket_asian_cashflows:
    _DATE_TOLERANCE = 1e-12

    def __init__(self, num_assets, init_fixing_date, payment_date,
                 fixing_frequency, end_broken_period=True, strike=1.0, is_call=True,
                 device=None, dtype=None, keep_feature_dim=False,
                 end_broken_pediod=None):

        self.num_assets = num_assets
        self.init_fixing_date = init_fixing_date
        self.payment_date = payment_date
        self.strike = strike
        self.is_call = is_call
        self.device = device
        self.dtype = dtype
        self.keep_feature_dim = keep_feature_dim
        self.fixing_frequency = fixing_frequency

        if not isinstance(self.fixing_frequency, enums.DateFrequency):
            raise ValueError("fixing_frequency must be an instance of DateFrequency")

        self.fixing_interval = float(self.fixing_frequency.value)
        if self.fixing_interval <= 0:
            raise ValueError("fixing_frequency must imply a positive interval")

        if self.payment_date <= self.init_fixing_date:
            raise ValueError("payment_date must be greater than init_fixing_date")

        if end_broken_pediod is not None:
            end_broken_period = end_broken_pediod

        self.end_broken_period = bool(end_broken_period)
        self.end_broken_pediod = self.end_broken_period
        self.product_dates = self._build_product_dates()

    def _build_product_dates(self):
        total_period = self.payment_date - self.init_fixing_date
        regular_periods = int(np.floor(total_period / self.fixing_interval + self._DATE_TOLERANCE))

        if self.end_broken_period:
            dates = self.init_fixing_date + self.fixing_interval * np.arange(regular_periods + 1)
            if np.isclose(dates[-1], self.payment_date, atol=self._DATE_TOLERANCE, rtol=0.0):
                dates[-1] = self.payment_date
            else:
                dates = np.append(dates, self.payment_date)
        else:
            dates = self.payment_date - self.fixing_interval * np.arange(regular_periods + 1)
            dates = dates[::-1]
            if np.isclose(dates[0], self.init_fixing_date, atol=self._DATE_TOLERANCE, rtol=0.0):
                dates[0] = self.init_fixing_date
            else:
                dates = np.insert(dates, 0, self.init_fixing_date)

        return dates


    def get_product_dates(self):
        return self.product_dates

    def set_product_indexes(self, indexes):
        self.product_indexes = indexes

    def compute_cashflows(self, asset_prices):
        if not hasattr(self, "product_indexes"):
            raise ValueError("Product indexes have not been set.")

        relevant_prices = asset_prices[:, self.product_indexes]
        geometric_average = np.exp(np.mean(np.log(relevant_prices), axis=1))
        if self.is_call:
            cashflows = np.maximum(geometric_average - self.strike, 0.0)
        else:
            cashflows = np.maximum(self.strike - geometric_average, 0.0)

        if self.keep_feature_dim:
            cashflows = cashflows[:, np.newaxis]

        return cashflows