import torch


N = torch.distributions.Normal(loc=0.0, scale=1.0)


def Black(Forward: torch.FloatTensor, Strike: torch.FloatTensor, 
          TTM: torch.FloatTensor, rate: torch.FloatTensor, Vol: torch.FloatTensor, IsCall: bool) -> torch.FloatTensor  :

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
