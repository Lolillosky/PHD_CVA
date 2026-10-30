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


class GeometricBasketAsianCashflows:


    def __init__(self, num_assets, init_fixing_date, payment_date,
                 fixing_frequency, end_broken_period=True, strike=1.0, is_call=True,
                 current_geom_average=None, first_fixing=None, device=None, dtype=None):

        self.num_assets = num_assets
        self.device, self.dtype = resolve_device_dtype(device, dtype)
        self.init_fixing_date = torch.as_tensor(init_fixing_date, device=self.device, dtype=self.dtype)
        self.payment_date = torch.as_tensor(payment_date, device=self.device, dtype=self.dtype)
        self.strike = strike
        self.is_call = is_call
        self.fixing_frequency = fixing_frequency
        self.end_broken_period = end_broken_period
        self.current_geom_average = torch.as_tensor(current_geom_average, device=self.device, dtype=self.dtype) if current_geom_average is not None else None
        self.first_fixing = torch.as_tensor(first_fixing, device=self.device, dtype=self.dtype) if first_fixing is not None else None


        if not isinstance(self.fixing_frequency, enums.DateFrequency):
            raise ValueError("fixing_frequency must be an instance of DateFrequency")

        self.fixing_interval = float(self.fixing_frequency.value)
        if self.fixing_interval <= 0:
            raise ValueError("fixing_frequency must imply a positive interval")

        if self.payment_date <= self.init_fixing_date:
            raise ValueError("payment_date must be greater than init_fixing_date")

   
        self.product_dates = self._build_product_dates()

    def _build_product_dates(self):

        if not self.end_broken_period:
            dates = torch.arange(self.payment_date.item(), self.init_fixing_date.item(), -self.fixing_interval,
                                 device=self.device, dtype=self.dtype)
            dates = torch.cat((dates, self.init_fixing_date.reshape(1)))
            dates = dates.flip(0)


        else:
            dates = torch.arange(self.init_fixing_date.item(), self.payment_date.item(), self.fixing_interval,
                                 device=self.device, dtype=self.dtype)
            dates = torch.cat((dates, self.payment_date.reshape(1)))

        return dates

    def get_product_dates(self):
        return self.product_dates

    def set_product_indexes(self, indexes):
        self.product_indexes = indexes

    def compute_cashflows_and_path_dependency(self, asset_prices, cashflows_matrix,
                                              path_dependency, mask):
        if not hasattr(self, "product_indexes"):
            raise ValueError("Product indexes have not been set.")
        if asset_prices.shape[-1] != self.num_assets:
            raise ValueError("num_assets does not match asset_prices.shape[-1]")

        relevant_prices = asset_prices[:, self.product_indexes]
        if self.product_dates[0] >= 0 and self.current_geom_average is None and self.first_fixing is None:
            relative_prices = relevant_prices[:, 1:, :] / relevant_prices[:, :1, :]
            fixing_indexes = self.product_indexes[1:]
            initial_log_sum = asset_prices.new_zeros(asset_prices.shape[0])
            past_fixings = 0
        elif self.current_geom_average is not None and self.product_dates[0] < 0 and self.first_fixing is not None:
            relative_prices = relevant_prices / self.first_fixing
            fixing_indexes = self.product_indexes
            past_fixings = int(((self.product_dates[1:] < 0).sum()).item())
            initial_log_sum = self.current_geom_average.expand(asset_prices.shape[0]).log() * (self.num_assets * past_fixings)
        else:
            raise ValueError("Invalid state for product dates, current_geom_average and first_fixing")

        # Combine the historical average and new fixings with their observation counts.
        log_sums = torch.cat((initial_log_sum[:, None], initial_log_sum[:, None] +
                              relative_prices.log().sum(dim=-1).cumsum(dim=1)), dim=1)
        observations = (past_fixings + torch.arange(log_sums.shape[1], device=asset_prices.device)) * self.num_assets
        averages = torch.exp(log_sums / observations.clamp_min(1))
        geometric_average = averages[:, -1]
        grid_indexes = torch.arange(asset_prices.shape[1], device=asset_prices.device)
        counts = torch.searchsorted(fixing_indexes, grid_indexes, right=True)
        active = grid_indexes <= self.product_indexes[-1]
        if self.product_dates[0] >= 0:
            active = active & (grid_indexes >= self.product_indexes[0])
        path_dependency[:] = torch.where(active, averages[:, counts], 1.0)
        mask[:] = active

        strike = torch.as_tensor(self.strike, device=asset_prices.device, dtype=asset_prices.dtype)
        if self.is_call:
            cashflows = torch.clamp(geometric_average - strike, min=0.0)
        else:
            cashflows = torch.clamp(strike - geometric_average, min=0.0)

        cashflows_matrix[:, self.product_indexes[-1]] += cashflows
        
        
class CashFlowsHandler:

    def __init__(self, ccr_dates, cashflow_object_list, device=None, dtype=None):


        self.device, self.dtype = resolve_device_dtype(device, dtype)
        self.ccr_dates = torch.as_tensor(ccr_dates, device=self.device, dtype=self.dtype)

        self.cashflow_object_list = cashflow_object_list

        # Build the simulation date grid, the ccr indexes and product indexes for each cash flow
        self.simulation_dates, self.ccr_indexes = self.build_simulation_date_grid_set_indexes(self.ccr_dates, self.cashflow_object_list)


    def build_simulation_date_grid_set_indexes(self, ccr_dates, cashflow_object_list, tol = 1e-8):
        """
        Build the simulation date grid and map input dates into that grid.

        The simulation grid is the sorted union of CCR observation dates and product
        dates. Product dates are grouped by cash flow, and the returned product
        indexes preserve that grouping.

        Parameters
        ----------
        ccr_dates : array-like
            CCR observation dates.
        cashflow_object_list : sequence of Cashflow objects
            List of cash flow objects, each providing its fixing/payment dates.

        Returns
        -------
        tuple
            Tensors ``simulation_dates`` and ``ccr_indexes``. It also sets the product indexes on each cash flow object.
        """
        ccr_dates = torch.as_tensor(ccr_dates, device=self.device, dtype=self.dtype)
        ccr_dates = torch.round(ccr_dates / tol) * tol
        self.ccr_dates = ccr_dates
        product_dates = [
            torch.round(torch.as_tensor(cf.get_product_dates()[cf.get_product_dates()>=0], device=self.device, dtype=self.dtype) / tol) * tol
            for cf in cashflow_object_list
        ]
        all_product_dates = torch.cat([dates.reshape(-1) for dates in product_dates]) if product_dates else ccr_dates.new_empty(0)

        simulation_dates = torch.unique(torch.cat((ccr_dates, all_product_dates)), sorted=True)
        ccr_indexes = torch.searchsorted(simulation_dates, ccr_dates)
        product_indexes = [
            torch.searchsorted(simulation_dates, dates).reshape(dates.shape)
            for dates in product_dates
        ]

        for cf, indexes in zip(cashflow_object_list, product_indexes):
            cf.set_product_indexes(indexes)

        return simulation_dates, ccr_indexes
    
        
    def compute_cashflows(self, risk_free_rate, asset_prices):


        risk_free_rate = torch.as_tensor(risk_free_rate, device=self.device, dtype=self.dtype)
        cashflows_matrix = torch.zeros((asset_prices.shape[0], asset_prices.shape[1]), device=self.device, dtype=self.dtype)


        path_dependency = torch.ones((*cashflows_matrix.shape, len(self.cashflow_object_list)),
                                     device=self.device, dtype=self.dtype)
        mask = torch.zeros_like(path_dependency, dtype=torch.bool)
        for i, cf in enumerate(self.cashflow_object_list):
            cf.compute_cashflows_and_path_dependency(asset_prices, cashflows_matrix,
                                                    path_dependency[:, :, i], mask[:, :, i])

    
        indexes = torch.searchsorted(self.ccr_dates, self.simulation_dates, right=True) -1

        discount_cash_flows_to_previous_ccr_date = cashflows_matrix * torch.exp(-risk_free_rate * (self.simulation_dates - self.ccr_dates[indexes]))

        sum_discounted_cf_ccr_intervals = torch.zeros((cashflows_matrix.shape[0], len(self.ccr_dates)),dtype=cashflows_matrix.dtype, device=cashflows_matrix.device)

        sum_discounted_cf_ccr_intervals.index_add_(1, indexes, discount_cash_flows_to_previous_ccr_date)


        discounts = torch.exp(-risk_free_rate * (self.ccr_dates - self.ccr_dates[0]))
        sum_discounted_cf_ccr = sum_discounted_cf_ccr_intervals  * discounts

        sum_discounted_cf_ccr = (
            sum_discounted_cf_ccr.flip(1).cumsum(dim=1).flip(1) / discounts
        )

        path_dependency = path_dependency[:, self.ccr_indexes, :]
        mask = mask[:, self.ccr_indexes, :]
        return sum_discounted_cf_ccr, sum_discounted_cf_ccr_intervals, cashflows_matrix, path_dependency, mask

    


