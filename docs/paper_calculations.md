# Paper Calculations Technical Note

This note summarizes the calculations implemented in `CODE/PaperCode` and
validated in `NOTEBOOKS/PaperNotebooks/PaperResults.ipynb`.

The current paper workflow focuses on a geometric basket Asian option under
correlated Black-Scholes dynamics. The code uses PyTorch tensors throughout and
organizes simulated paths in RNN-style layout:

```text
(simulations, time, assets)
```

## 1. Correlated Risk-Factor Simulation

Implemented in `CODE/PaperCode/factor_simulation.py`.

For each risk factor or asset `j`, the simulated process is a geometric
Brownian motion:

```math
dS_t^j = \mu_j S_t^j dt + \sigma_j S_t^j dW_t^j
```

with instantaneous correlation:

```math
Corr(dW_t^h, dW_t^k) = \rho_{hk}.
```

On a time grid:

```math
0 = t_0 < t_1 < \cdots < t_n,
```

the simulator computes time increments:

```math
\Delta t_i = t_i - t_{i-1}.
```

It draws independent standard normal shocks and correlates them through the
Cholesky factor `L` of the correlation matrix:

```math
\rho = LL^\top.
```

The correlated Brownian increment is:

```math
\Delta W_i = \sqrt{\Delta t_i} Z_i L^\top,
```

where:

```math
Z_i \sim N(0, I).
```

The log-Euler exact GBM update is:

```math
S_{t_i}^j =
S_{t_{i-1}}^j
\exp\left[
\left(\mu_j - \frac{1}{2}\sigma_j^2\right)\Delta t_i
+ \sigma_j \Delta W_i^j
\right].
```

Equivalently, in cumulative form from `t_0`:

```math
S_{t_i}^j =
S_{t_0}^j
\exp\left[
\sum_{k=1}^{i}
\left(
\left(\mu_j - \frac{1}{2}\sigma_j^2\right)\Delta t_k
+ \sigma_j \Delta W_k^j
\right)
\right].
```

The output includes the initial spot as the first time slice, so its shape is:

```text
(num_sims, num_time_points, num_risk_factors)
```

### Brownian Bridge Variant

`RiskFactorSimulator.simulate_paths_with_bridge` also supports conditioning
paths on a known spot at a pivot date. The code infers the Brownian value at the
pivot from the observed spot:

```math
W_{t_p}^j =
\frac{
\log(S_{t_p}^j / S_0^j)
- \left(\mu_j - \frac{1}{2}\sigma_j^2\right)t_p
}{
\sigma_j
}.
```

Because the model is correlated, the bridge is sampled in independent Brownian
coordinates:

```math
Z_{t_p} = W_{t_p}(L^{-1})^\top.
```

Intermediate bridge points are sampled conditionally, then mapped back to
correlated Brownian coordinates with `L^\top`.

## 2. Black Formula

Implemented in `CODE/PaperCode/option_formulas.py`.

The scalar `black` function and vectorized `black_vectorized` function price a
European option on a forward. For positive time to maturity:

```math
d_1 =
\frac{
\log(F/K) + \frac{1}{2}\sigma^2 T
}{
\sigma \sqrt{T}
},
\qquad
d_2 = d_1 - \sigma \sqrt{T}.
```

For a call:

```math
V = e^{-rT}\left(FN(d_1) - KN(d_2)\right).
```

For a put:

```math
V = e^{-rT}\left(-FN(-d_1) + KN(-d_2)\right).
```

At zero maturity, the vectorized implementation returns intrinsic value rather
than applying the Black formula.

## 3. Geometric Basket Asian Option

Implemented in:

- `basket_geom_asian`
- `basket_geom_asian_vectorized`
- `basket_geom_asian_cashflows`

The notebook prices a normalized geometric basket Asian payoff:

```math
V_T =
\left[
\left(
\prod_{i=1}^{n}
\prod_{j=1}^{m}
\frac{S_{t_i}^j}{S_{t_0}^j}
\right)^{1/(nm)}
- 1
\right]^+.
```

Here:

- `m` is the number of assets.
- `n` is the number of Asian fixing dates after `t_0`.
- `S_{t_i}^j` is asset `j` at fixing date `t_i`.
- The strike is fixed at `K = 1` in the current paper implementation.
- The `t_0` fixing is used as normalization and is not included in the product.

## 4. Closed-Form Valuation at an Intermediate Date

At value date `t_v`, the payoff is split into known fixings and future fixings.
The value is:

```math
V_{t_v}
=
E\left[
e^{-r(t_n-t_v)}
\left(
A_v B_v - 1
\right)^+
\mid \mathcal{F}_{t_v}
\right],
```

where the known component is:

```math
A_v =
\left(
\prod_{i=1}^{v}
\prod_{j=1}^{m}
\frac{S_{t_i}^j}{S_{t_0}^j}
\right)^{1/(nm)}.
```

Under correlated Black-Scholes, the future log-geometric component is normal.
The code computes:

```math
\mu_v =
\frac{1}{nm}
\sum_{i=v+1}^{n}
\sum_{j=1}^{m}
\left(r - \frac{1}{2}\sigma_j^2\right)
(t_i - t_v),
```

and:

```math
V_v =
\frac{1}{n^2m^2}
\sum_{h=1}^{m}
\sum_{k=1}^{m}
\sum_{i=v+1}^{n}
\sum_{\ell=v+1}^{n}
\sigma_h \sigma_k \rho_{hk}
\min(t_i - t_v, t_\ell - t_v).
```

The implementation computes this variance by:

1. Building the asset covariance matrix:

```math
\Sigma = diag(\sigma)\rho diag(\sigma).
```

2. Building a time covariance matrix over future fixing dates:

```math
M_{i\ell} = \min(\tau_i, \tau_\ell),
\qquad
\tau_i = t_i - t_v.
```

3. Combining them as:

```math
V_v =
\frac{
\sum_{h,k}\Sigma_{hk}
\sum_{i,\ell}M_{i\ell}
}{
n^2m^2
}.
```

The equivalent Black forward is:

```math
F_v =
A_v
\left(
\prod_{j=1}^{m}
\left(\frac{S_{t_v}^j}{S_{t_0}^j}\right)^{n-v}
\right)^{1/(nm)}
\exp\left(\mu_v + \frac{1}{2}V_v\right).
```

The price is then evaluated with Black using:

```math
K = 1,
\qquad
T = t_n - t_v,
\qquad
\hat{\sigma}_v = \sqrt{V_v / T}.
```

So:

```math
Price(t_v) = Black(F_v, 1, T, r, \hat{\sigma}_v).
```

## 5. Vectorized Pricing

`basket_geom_asian_vectorized` computes the same formula for all simulations
and all valuation dates at once.

The vectorized input shape is:

```text
(simulations, time, assets)
```

The default output shape is:

```text
(simulations, time, 1)
```

The notebook compares this vectorized implementation to the scalar
`basket_geom_asian` implementation path-by-path and date-by-date, excluding the
terminal date to avoid the scalar zero-maturity edge case.

Observed notebook validation:

```text
paths shape:        (500, 21, 3)
vectorized shape:   (500, 20, 1)
scalar shape:       (500, 20, 1)
max abs diff:       7.216e-16
mean abs diff:      1.375e-17
```

This confirms that the scalar and vectorized closed-form implementations are
numerically equivalent up to floating-point precision.

## 6. Discounted Cash-Flow Stream

Implemented in `CODE/PaperCode/cashflows.py`.

The cash-flow function produces a realized discounted payoff stream for the same
geometric basket Asian option. It first computes relative fixings:

```math
R_{i,j} = \frac{S_{t_i}^j}{S_{t_0}^j},
\qquad i = 1,\ldots,n.
```

The realized geometric average is:

```math
G =
\left(
\prod_{i=1}^{n}
\prod_{j=1}^{m}
R_{i,j}
\right)^{1/(nm)}.
```

For a call:

```math
Payoff = \max(G - 1, 0).
```

For a put:

```math
Payoff = \max(1 - G, 0).
```

The code discounts the payoff from maturity to the initial date:

```math
CF_T = Payoff \cdot e^{-r(t_n - t_0)}.
```

The returned stream is zero at all non-terminal dates and contains the
discounted payoff only at maturity:

```text
cashflows[:, :-1, :] = 0
cashflows[:, -1, 0] = CF_T
```

Default output shape:

```text
(simulations, time, 1)
```

## 7. Notebook Validation of Cashflows

The notebook checks that the discounted cash-flow representation is consistent
with the analytic time-zero price:

```math
E[CF_T] \approx Price(t_0).
```

It compares:

- Monte Carlo estimate: average terminal discounted payoff.
- Analytic estimate: average closed-form `t_0` value.
- Monte Carlo standard error.
- A 95% normal-approximation confidence interval.

Observed notebook validation:

```text
cashflows shape:       (500, 21, 1)
non-terminal abs sum:  0.000e+00
MC discounted payoff:  0.0588628776
analytic t0 price:     0.0627522690
abs diff:              3.889e-03
MC standard error:     4.221e-03
95% MC CI:             [0.0505894541, 0.0671363012]
analytic inside CI:    True
```

This validates that the realized payoff convention matches the closed-form
pricing convention within Monte Carlo sampling error.

## 8. Concrete Notebook Setup

The notebook uses:

```text
num_risk_factors = 3
initial_spot_values = [100.0, 60.0, 140.0]
drift_array = [0.03, 0.03, 0.03]
volatility_array = [0.10, 0.25, 0.40]
correl_matrix =
    [[1.0, 0.80, 0.10],
     [0.80, 1.0, -0.5],
     [0.10, -0.5, 1.0]]
time_steps = linspace(0, 5, 21)
num_simulated_paths_used = 500
```

The time grid has 21 points from 0 to 5 years, so there are 20 intervals and 20
Asian fixings after the initial normalization date.

## 9. Implementation Notes

- `factor_simulation.py` supports irregular time grids because it computes
  `delta_t` directly from adjacent entries in `time_steps`.
- `basket_geom_asian_vectorized` is the production-friendly implementation for
  batch pricing over all paths and dates.
- `basket_geom_asian` is useful as a scalar reference implementation.
- `basket_geom_asian_cashflows` uses the same payoff convention as the analytic
  formula, which is why the notebook can compare the Monte Carlo average of
  terminal discounted cashflows against the analytic time-zero value.
- In the displayed notebook source, the pricing cell refers to `corr_matrix`,
  while the setup cell defines `correl_matrix`. The executed output indicates
  the intended object is the correlation matrix defined in the setup.
