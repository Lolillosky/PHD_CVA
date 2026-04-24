import torch


N = torch.distributions.Normal(loc=0.0, scale=1.0)


def Black(Forward: torch.FloatTensor, Strike: torch.FloatTensor, 
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
    


def BasketGeomAsian(init_time_array, value_date_index, risk_free_rate, num_assets,
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


  return Black(Forward, 1.0, remaining_maturity, risk_free_rate, torch.sqrt(V / remaining_maturity), IsCall)

