import torch


class RiskFactorSimulator:

    def __init__(
        self,
        num_risk_factors,
        initial_spot_values,
        drift_array,
        volatility_array,
        correl_matrix,
        time_steps,
        device=None,
        dtype=torch.float64,
    ):

        self.num_risk_factors = num_risk_factors
        if device is None:
            resolved_device = "cpu"
        else:
            resolved_device = device

        self.device = torch.device(resolved_device)
        self.dtype = dtype

        self.initial_spot_values = torch.as_tensor(initial_spot_values, dtype=self.dtype, device=self.device)
        self.drift_array = torch.as_tensor(drift_array, dtype=self.dtype, device=self.device)
        self.volatility_array = torch.as_tensor(volatility_array, dtype=self.dtype, device=self.device)
        self.correl_matrix = torch.as_tensor(correl_matrix, dtype=self.dtype, device=self.device)
        self.time_steps = torch.as_tensor(time_steps, dtype=self.dtype, device=self.device)


    def simulate_paths(self, num_sims: int) -> torch.Tensor:
        """
        Simulate GBM paths for all risk factors with correlated Brownian increments.

        Parameters
        ----------
        num_sims : int
            Number of Monte Carlo paths.

        Returns
        -------
        torch.Tensor of shape (num_sims, num_steps, num_risk_factors)
            Simulated asset paths (starting AFTER t=0).
        """
        # --- time increments (handles irregular / odd time grids) ---
        # time_steps: shape (num_steps,), e.g. [0.25, 0.5, 1.0, 1.5]
        t_full = torch.cat([torch.zeros(1, dtype=self.dtype, device=self.device), self.time_steps])  # prepend 0
        delta_t = t_full[1:] - t_full[:-1]          # shape: (num_steps,)
        num_steps = delta_t.shape[0]

        # --- Cholesky factor of correlation matrix ---
        L = torch.linalg.cholesky(self.correl_matrix)   # shape: (num_rf, num_rf)

        # --- standard normal increments scaled by sqrt(delta_t) ---
        # shape: (num_sims, num_steps, num_rf)
        sqrt_dt = torch.sqrt(delta_t).unsqueeze(0).unsqueeze(-1)   # (1, num_steps, 1)
        inc_W = torch.randn(
            num_sims,
            num_steps,
            self.num_risk_factors,
            dtype=self.dtype,
            device=self.device,
        ) * sqrt_dt

        # --- correlate increments: inc_W_correl = inc_W @ L.T ---
        inc_W_correl = inc_W @ L.T    # (num_sims, num_steps, num_rf)

        # --- GBM log-returns for each step ---
        # sigma and mu shapes broadcast over (num_sims, num_steps, num_rf)
        mu    = self.drift_array   # using vol as drift (risk-neutral); override if needed
        sigma = self.volatility_array   # shape: (num_rf,)
        dt    = delta_t.unsqueeze(0).unsqueeze(-1)   # (1, num_steps, 1)

        log_increments = (mu - 0.5 * sigma ** 2) * dt + sigma * inc_W_correl

        # --- cumulative product = cumulative sum in log space ---
        # gross_rets[i,t,k] = S0[k] * exp(sum of log_increments up to t)
        cum_log = torch.cumsum(log_increments, dim=1)   # (num_sims, num_steps, num_rf)
        S0 = self.initial_spot_values.unsqueeze(0).unsqueeze(0)   # (1, 1, num_rf)
        paths = S0 * torch.exp(cum_log)   # (num_sims, num_steps, num_rf)

        return paths

    def simulate_paths_with_bridge(
        self,
        num_sims: int,
        pivot_step_idx: int,
        spot_at_pivot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate GBM paths using a Brownian bridge for the segment [0, t_pivot]
        and standard GBM for the segment (t_pivot, T].

        The Brownian bridge constrains the Brownian motion to pass through the
        realised log-return implied by `spot_at_pivot` at time `t_pivot`, while
        intermediate steps within [0, t_pivot] are sampled conditionally.

        Parameters
        ----------
        num_sims : int
            Number of Monte Carlo paths.
        pivot_step_idx : int
            Index into `self.time_steps` (0-based) that defines the pivot time t_pivot.
            The bridge covers steps 0 … pivot_step_idx (inclusive).
            The forward simulation covers steps pivot_step_idx+1 … num_steps-1.
        spot_at_pivot : torch.Tensor, shape (num_sims, num_risk_factors) or (num_risk_factors,)
            Observed / given spot values at t_pivot for each simulation and risk factor.
            If shape is (num_risk_factors,), the same values are broadcast over all sims.

        Returns
        -------
        torch.Tensor of shape (num_sims, num_steps, num_risk_factors)
            Full simulated paths over all time steps.
        """
        dev, dt = self.device, self.dtype
        num_steps = len(self.time_steps)

        if pivot_step_idx < 0 or pivot_step_idx >= num_steps:
            raise ValueError(
                f"pivot_step_idx={pivot_step_idx} out of range [0, {num_steps - 1}]."
            )

        # ------------------------------------------------------------------ #
        # Time grid: prepend t=0                                              #
        # ------------------------------------------------------------------ #
        t_full  = torch.cat([torch.zeros(1, dtype=dt, device=dev), self.time_steps])
        delta_t = t_full[1:] - t_full[:-1]                          # (num_steps,)
        t_pivot = t_full[pivot_step_idx + 1]                         # scalar

        # ------------------------------------------------------------------ #
        # Cholesky factor (shared by both segments)                           #
        # ------------------------------------------------------------------ #
        L    = torch.linalg.cholesky(self.correl_matrix)            # (num_rf, num_rf)
        L_inv = torch.linalg.inv(L)                                  # (num_rf, num_rf)

        mu    = self.drift_array                                     # (num_rf,)
        sigma = self.volatility_array                                # (num_rf,)
        S0    = self.initial_spot_values                             # (num_rf,)

        # ------------------------------------------------------------------ #
        # Spot at pivot → pinned correlated BM value W_pivot                  #
        # ------------------------------------------------------------------ #
        spot_at_pivot = torch.as_tensor(spot_at_pivot, dtype=dt, device=dev)
        if spot_at_pivot.dim() == 1:
            spot_at_pivot = spot_at_pivot.unsqueeze(0).expand(num_sims, -1)
        # shape: (num_sims, num_rf)

        # log(S/S0) = (mu - 0.5*sigma^2)*t + sigma * (L @ Z_t)
        # where Z_t ~ N(0, I) is the INDEPENDENT BM.
        # W_pivot (correlated BM) = L @ Z_pivot
        # => W_pivot[k] = (log_ret_pivot[k] - drift[k]) / sigma[k]  for each factor k
        log_ret_pivot = torch.log(spot_at_pivot / S0.unsqueeze(0))  # (num_sims, num_rf)
        drift_total   = (mu - 0.5 * sigma ** 2) * t_pivot           # (num_rf,)
        W_pivot       = (log_ret_pivot - drift_total) / sigma        # (num_sims, num_rf)  correlated BM

        # Back-transform to INDEPENDENT space: Z_pivot = L_inv @ W_pivot
        # (num_sims, num_rf) @ (num_rf, num_rf).T  — broadcast over sims
        Z_pivot = W_pivot @ L_inv.T                                  # (num_sims, num_rf)  independent BM

        # ================================================================== #
        # BRIDGE SEGMENT  [0 … pivot_step_idx]                               #
        # ================================================================== #
        # The bridge is constructed in INDEPENDENT BM space (Z), where each
        # factor is a separate standard BM. The scalar Brownian bridge formula
        # applies independently to each factor in Z-space:
        #   E[dZ_k | Z(T)] = ds/T * Z(T)
        #   Var[dZ_k | Z(T)] = ds * (T - s_k) / T
        # After sampling in Z-space, we apply L to get back to the correlated W.
        # ================================================================== #
        n_bridge = pivot_step_idx + 1

        if n_bridge > 0:
            s_times = t_full[1 : n_bridge + 1]                      # (n_bridge,)
            s_prev  = t_full[0 : n_bridge]                           # (n_bridge,)
            ds      = s_times - s_prev                               # (n_bridge,)

            # Conditional mean of dZ_k in independent space
            # shape: (num_sims, n_bridge, num_rf)
            cond_mean_dZ = (ds / t_pivot).unsqueeze(0).unsqueeze(-1) * Z_pivot.unsqueeze(1)

            # Conditional std of dZ_k (scalar per step, same for all factors)
            var_dZ_k = (ds * (t_pivot - s_times) / t_pivot).clamp(min=0.0)  # (n_bridge,)
            std_dZ_k = torch.sqrt(var_dZ_k)                          # (n_bridge,)

            # Sample interior steps; last step is deterministic (std=0 at pivot)
            n_interior = n_bridge - 1
            if n_interior > 0:
                z_interior = torch.randn(
                    num_sims, n_interior, self.num_risk_factors, dtype=dt, device=dev
                )
                # Scale in independent space, then add conditional mean
                dZ_interior = (
                    std_dZ_k[:n_interior].unsqueeze(0).unsqueeze(-1) * z_interior
                    + cond_mean_dZ[:, :n_interior, :]
                )                                                    # (num_sims, n_interior, num_rf)
            # Last step: exact deterministic pin (std=0)
            dZ_last = cond_mean_dZ[:, n_bridge - 1 : n_bridge, :]   # (num_sims, 1, num_rf)

            if n_interior > 0:
                dZ_bridge = torch.cat([dZ_interior, dZ_last], dim=1) # (num_sims, n_bridge, num_rf)
            else:
                dZ_bridge = dZ_last

            # Map back to correlated BM space: dW = dZ @ L.T
            dW_bridge = dZ_bridge @ L.T                              # (num_sims, n_bridge, num_rf)

            # GBM log increments
            dt_bridge      = delta_t[:n_bridge].unsqueeze(0).unsqueeze(-1)
            log_inc_bridge = (mu - 0.5 * sigma ** 2) * dt_bridge + sigma * dW_bridge
            cum_log_bridge = torch.cumsum(log_inc_bridge, dim=1)
            paths_bridge   = S0.unsqueeze(0).unsqueeze(0) * torch.exp(cum_log_bridge)

        # ================================================================== #
        # FORWARD SEGMENT  (pivot_step_idx … num_steps-1]                    #
        # ================================================================== #
        n_forward = num_steps - n_bridge

        if n_forward > 0:
            z_fwd = torch.randn(
                num_sims, n_forward, self.num_risk_factors, dtype=dt, device=dev
            )
            sqrt_dt_fwd = torch.sqrt(delta_t[n_bridge:]).unsqueeze(0).unsqueeze(-1)
            inc_W_fwd    = (z_fwd @ L.T) * sqrt_dt_fwd

            dt_fwd       = delta_t[n_bridge:].unsqueeze(0).unsqueeze(-1)
            log_inc_fwd  = (mu - 0.5 * sigma ** 2) * dt_fwd + sigma * inc_W_fwd

            cum_log_fwd  = torch.cumsum(log_inc_fwd, dim=1)
            # Start from spot_at_pivot
            paths_fwd = spot_at_pivot.unsqueeze(1) * torch.exp(cum_log_fwd)

        # ================================================================== #
        # Concatenate segments                                                #
        # ================================================================== #
        if n_bridge > 0 and n_forward > 0:
            paths = torch.cat([paths_bridge, paths_fwd], dim=1)
        elif n_bridge > 0:
            paths = paths_bridge
        else:
            paths = paths_fwd

        return paths                                                 # (num_sims, num_steps, num_rf)

    

