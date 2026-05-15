import torch


N = torch.distributions.Normal(loc=0.0, scale=1.0).cdf


def black(Forward: torch.FloatTensor, Strike: torch.FloatTensor, 
          TTM: torch.FloatTensor, rate: torch.FloatTensor, Vol: torch.FloatTensor,
          IsCall: bool,  device=None, dtype=torch.float64,) -> torch.FloatTensor  :

  '''
  Inputs:
  -------
    Forward (float): Forward value
    Strike (float): strike price
    TTM (float): time to maturity in years
    rate (float): risk free rate
    Vol (float): volatility
    IsCall (bool): True if call option, False if put option
  Outputs:
  --------
    Option premium (float)
  '''

  if TTM >0:

    d1 = (torch.log(Forward/Strike) + (Vol*Vol/2)*TTM)/(Vol*torch.sqrt(TTM))
    d2 = (torch.log(Forward/Strike) + (- Vol*Vol/2)*TTM)/(Vol*torch.sqrt(TTM))

    if IsCall:

      return (Forward*N(d1)-Strike*N(d2))*torch.exp(-rate*TTM)

    else:

      return (-Forward*N(-d1)+Strike*N(-d2))*torch.exp(-rate*TTM)

  else:

    if IsCall:

      return torch.maximum(Forward-Strike,0)

    else:

      return torch.maximum(-Forward+Strike,0)


def black_vectorized(Forward: torch.FloatTensor, Strike: torch.FloatTensor,
                     TTM: torch.FloatTensor, rate: torch.FloatTensor, Vol: torch.FloatTensor,
                     IsCall: bool, device=None, dtype=torch.float64,
                     eps: float = 1e-12) -> torch.FloatTensor:

  '''
  Vectorized Black option pricing.

  Inputs can be scalars or tensors with broadcast-compatible shapes.
  This function supports mixed batches where some maturities are zero.
  '''

  # Promote inputs to tensors for broadcasted elementwise operations.
  Forward = torch.as_tensor(Forward, device=device, dtype=dtype)
  Strike = torch.as_tensor(Strike, device=device, dtype=dtype)
  TTM = torch.as_tensor(TTM, device=device, dtype=dtype)
  rate = torch.as_tensor(rate, device=device, dtype=dtype)
  Vol = torch.as_tensor(Vol, device=device, dtype=dtype)

  positive_ttm = TTM > 0

  # Guard divisions/logs when TTM or Vol are zero; values are replaced for these entries later.
  safe_ttm = torch.clamp(TTM, min=eps)
  safe_vol = torch.clamp(Vol, min=eps)
  sqrt_ttm = torch.sqrt(safe_ttm)

  log_fk = torch.log(torch.clamp(Forward, min=eps) / torch.clamp(Strike, min=eps))
  d1 = (log_fk + 0.5 * safe_vol * safe_vol * safe_ttm) / (safe_vol * sqrt_ttm)
  d2 = d1 - safe_vol * sqrt_ttm

  discount = torch.exp(-rate * safe_ttm)
  call_price = (Forward * N(d1) - Strike * N(d2)) * discount
  put_price = (-Forward * N(-d1) + Strike * N(-d2)) * discount
  model_price = call_price if IsCall else put_price

  intrinsic = torch.maximum(Forward - Strike, torch.zeros_like(Forward)) if IsCall \
      else torch.maximum(Strike - Forward, torch.zeros_like(Forward))

  return torch.where(positive_ttm, model_price, intrinsic)
    


def basket_geom_asian(init_time_array, value_date_index, risk_free_rate, num_assets,
                     assets_vol, assets_correl, price_history, IsCall):
  '''
  Inputs:
  -------
  * init_time_array (array(float)): array of initial times for the asian dates.
  * value_date_index (int): index, within the array of asian dates indicating the value date.
  * risk_free_rate (float): risk free rate
  * num_assets (int): number of underlying assets.
  * assets_vol (array(float)): array of indiv assets vols.
  * assets_correl (array(float, float)): matrix of correlations
  * initial_maturity (float): maturity of the product as seen on initial spot fixing date.
  * price_history (array(float, float)): history of fixings of the underlyings up to value date. Assets per row, time steps per column.
  * IsCall (bool): True if call option, False if put option
  Outputs:
  --------
  * Option price (float)
  '''



  num_asian_dates = len(init_time_array)
  pending_times_array = init_time_array[value_date_index+1:] - init_time_array[value_date_index]

  mu = torch.sum(risk_free_rate - 0.5*assets_vol*assets_vol)*torch.sum(pending_times_array) / (num_assets * (num_asian_dates-1))

  diag_vol = torch.diag(assets_vol.reshape(-1))
  cov_matrix = torch.matmul(diag_vol, torch.matmul(assets_correl, diag_vol))

  xx, yy = torch.meshgrid(pending_times_array, pending_times_array, indexing='ij')
  z = torch.minimum(xx, yy)

  V = torch.sum(cov_matrix) * torch.sum(z) / (num_assets*num_assets*(num_asian_dates-1)*(num_asian_dates-1))

  Forward = torch.pow(torch.prod(price_history[:, 1:value_date_index+1] / price_history[:,0].reshape(-1,1)),1.0/(num_assets * (num_asian_dates-1)))

  Forward *= torch.pow(torch.prod(price_history[:,value_date_index] / price_history[:,0]), (num_asian_dates-value_date_index-1)/(num_assets * (num_asian_dates-1)))

  Forward *= torch.exp(mu + 0.5 * V)

  remaining_maturity = init_time_array[-1] - init_time_array[value_date_index]


  return black(Forward, 1.0, remaining_maturity, risk_free_rate, torch.sqrt(V / remaining_maturity), IsCall)


def basket_geom_asian_vectorized(init_time_array, risk_free_rate, num_assets,
                                 assets_vol, assets_correl, price_history, IsCall,
                                 device=None, dtype=torch.float64,
                                 keep_feature_dim=True):
  '''
  Inputs:
  -------
  * init_time_array (array(float)): array of initial times for the asian dates.
  * risk_free_rate (float): risk free rate
  * num_assets (int): number of underlying assets.
  * assets_vol (array(float)): array of indiv assets vols.
  * assets_correl (array(float, float)): matrix of correlations
  * initial_maturity (float): maturity of the product as seen on initial spot fixing date.
  * price_history (array): fixing history with RNN-style shape
    (simulations, time, assets).
  * IsCall (bool): True if call option, False if put option
  * keep_feature_dim (bool): when True, return (simulations, time, 1). When
    False, return (simulations, time).
  Outputs:
  --------
  * Option values with RNN-style shape (simulations, time, 1), or
    (simulations, time) when keep_feature_dim is False.
  '''

  init_time_array = torch.as_tensor(init_time_array, device=device, dtype=dtype)
  assets_vol = torch.as_tensor(assets_vol, device=device, dtype=dtype)
  assets_correl = torch.as_tensor(assets_correl, device=device, dtype=dtype)
  risk_free_rate = torch.as_tensor(risk_free_rate, device=device, dtype=dtype)
  price_history = torch.as_tensor(price_history, device=device, dtype=dtype)

  num_asian_dates = init_time_array.numel()

  if price_history.ndim != 3:
    raise ValueError('price_history must have shape (simulations, time, assets)')

  if price_history.shape[2] != num_assets:
    raise ValueError('num_assets does not match price_history.shape[2]')

  if price_history.shape[1] != num_asian_dates:
    raise ValueError('price_history time dimension must match len(init_time_array)')

  if num_asian_dates < 2:
    raise ValueError('init_time_array must contain at least two dates')

  # Precompute deterministic terms for each valuation date.
  mu = torch.zeros(num_asian_dates, device=price_history.device, dtype=price_history.dtype)
  V = torch.zeros_like(mu)

  drift_sum = torch.sum(risk_free_rate - 0.5 * assets_vol * assets_vol)

  diag_vol = torch.diag(assets_vol.reshape(-1))
  cov_matrix = torch.matmul(diag_vol, torch.matmul(assets_correl, diag_vol))

  for value_date_index in range(num_asian_dates):
    pending_times_array = init_time_array[value_date_index+1:] - init_time_array[value_date_index]
    if pending_times_array.numel() == 0:
      continue

    mu[value_date_index] = drift_sum * torch.sum(pending_times_array) / (num_assets * (num_asian_dates - 1))

    xx, yy = torch.meshgrid(pending_times_array, pending_times_array, indexing='ij')
    z = torch.minimum(xx, yy)
    V[value_date_index] = torch.sum(cov_matrix) * torch.sum(z) / (
        num_assets * num_assets * (num_asian_dates - 1) * (num_asian_dates - 1)
    )

  rel_prices = price_history / price_history[:, 0:1, :]

  # Product over historical fixings up to each valuation date.
  hist_factor = torch.ones((price_history.shape[0], num_asian_dates), device=price_history.device, dtype=price_history.dtype)
  cum_hist = torch.cumprod(rel_prices[:, 1:, :], dim=1)
  hist_factor[:, 1:] = torch.pow(
      torch.prod(cum_hist, dim=2),
      1.0 / (num_assets * (num_asian_dates - 1)),
  )

  # Factor from known current fixing at each valuation date.
  per_date_asset_prod = torch.prod(rel_prices, dim=2)
  remaining_fixings = (num_asian_dates - 1) - torch.arange(num_asian_dates, device=price_history.device, dtype=price_history.dtype)
  current_exponent = remaining_fixings / (num_assets * (num_asian_dates - 1))
  current_factor = torch.pow(per_date_asset_prod, current_exponent.unsqueeze(0))

  forward = hist_factor * current_factor * torch.exp((mu + 0.5 * V).unsqueeze(0))

  remaining_maturity = init_time_array[-1] - init_time_array
  vol_per_date = torch.sqrt(V / torch.clamp(remaining_maturity, min=1e-12))

  option_values = black_vectorized(
      forward,
      1.0,
      remaining_maturity.unsqueeze(0),
      risk_free_rate,
      vol_per_date.unsqueeze(0),
      IsCall,
      device=price_history.device,
      dtype=price_history.dtype,
  )

  if keep_feature_dim:
    return option_values.unsqueeze(-1)

  return option_values
