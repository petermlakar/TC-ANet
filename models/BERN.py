import torch
import torch.nn as nn
import torch.jit as jit

import numpy as np

import math

class ModelMarginal(nn.Module):

    def __init__(self, 
                 degree: int = 8, 
                 positive_support: bool = True, 
                 monotone: bool = True,
                 censored: bool = False):

        super().__init__()

        self.sfp: nn.Module = nn.Softplus()

        self.degree: int = degree
        self.nquantiles: int = 99

        ########################################################################
        # Enforce a positive support for the Bernstein quanilte function
        # and enforce the monotonicity of the coefficients which in turn
        # results in the monotonicity of the quantile function.
        self.positive_support: bool = positive_support
        self.monotone: bool = monotone
        self.censored: bool = censored

        # Bernstein quantile function coefficient regression neural network
        self.number_of_required_parameters: int = self.degree + 1

        ########################################################################
        # Compute binomial coefficients and the remaining required parameters
        
        step: float = 1.0/(self.nquantiles + 1)
        self.q: torch.Tensor = torch.arange(step, 1.0, step = step)

        self.w: torch.Tensor = torch.ones(self.degree + 1, dtype = torch.float32)
        for i in torch.arange(0, self.degree + 1, dtype = torch.int32):
            self.w[i] = math.comb(self.degree, i)
        self.j: torch.Tensor = torch.arange(0, self.degree + 1, dtype = torch.float32)

        self.q: torch.Tensor = self.q.view((1, self.nquantiles, 1))

        t0: torch.Tensor = torch.pow(self.q, self.j)
        t1: torch.Tensor = torch.pow(1.0 - self.q, self.degree - self.j)

        ########################################################################
        # Define as parameters such that they will be moved to gpu when .to is 
        # invoked on the module.

        self.j: torch.Tensor = nn.Parameter(self.j[None, None], requires_grad = False)
        self.w: torch.Tensor = nn.Parameter(self.w[None, None], requires_grad = False)

        self.q: torch.Tensor = nn.Parameter(self.q, requires_grad = False)
        self.q_pp: torch.Tensor = nn.Parameter(t0*t1*self.w, requires_grad = False)

    def forward(self):
        return None

    @jit.export
    def set_parameters(self, parameters: torch.Tensor) -> torch.Tensor:

        if self.monotone:
            return torch.cumsum(torch.cat([parameters[..., 0][..., None] if not self.positive_support else nn.functional.softplus(parameters[..., 0][..., None]),
                                           nn.functional.softplus(parameters[..., 1:])], dim = -1), dim = -1).view(parameters.shape[0]*parameters.shape[1], 1, -1)
        elif self.positive_support:
            return nn.functional.softplus(parameters).view(parameters.shape[0]*parameters.shape[1], 1, -1)
 
        return parameters.view(parameters.shape[0]*parameters.shape[1], 1, -1)
   
    @jit.export
    def icdf(self, p: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        
        coef: torch.Tensor = self.set_parameters(parameters)

        bs: int = p.shape[0]
        lt: int = p.shape[1]
        nq: int = p.shape[2]

        p = torch.reshape(p, (bs*lt, nq, 1))

        j: torch.Tensor = self.j.expand(bs*lt, -1, -1)
        w: torch.Tensor = self.w.expand(bs*lt, -1, -1)

        q: torch.Tensor = (torch.pow(p, j)*torch.pow(1.0 - p, self.degree - j)*w*coef).sum(dim = -1)

        return q.view((bs, lt, nq))

    @jit.export
    def sample_marginal_quantile(self, number_of_quantiles: int, parameters: torch.Tensor) -> torch.Tensor:

        percentile: torch.Tensor = torch.linspace(1.0/(number_of_quantiles + 1), 1.0 - 1.0/(number_of_quantiles + 1), steps = number_of_quantiles, dtype = torch.float32, device = parameters.device)
        percentile = percentile[None, None].expand(parameters.shape[0], parameters.shape[1], -1)

        return self.icdf(percentile, parameters)

    @jit.export
    def sample_marginal(self, number_of_samples: int, parameters: torch.Tensor) -> torch.Tensor:

        percentile: torch.Tensor = torch.rand((parameters.shape[0], parameters.shape[1], number_of_samples), dtype = torch.float32, device = parameters.device)

        return self.icdf(percentile, parameters)

    @jit.ignore
    def loss(self, y: torch.Tensor, parameters: torch.Tensor):

        coef: torch.Tensor = self.set_parameters(parameters)

        # Transform y into latent distribution
        
        lt: int = y.shape[1]
        bs: int = y.shape[0]

        q: torch.Tensor = self.q.expand(bs*lt, -1, -1)
        j: torch.Tensor = self.j.expand(bs*lt, -1, -1)
        w: torch.Tensor = self.w.expand(bs*lt, -1, -1)
        q_pp: torch.Tensor = self.q_pp.expand(bs*lt, -1, -1)

        Q: torch.Tensor = (q_pp*coef).sum(dim = -1)
        y = y.view((bs*lt, 1))

        idx: torch.Tensor = (~(torch.isnan(y))).squeeze()

        dif: torch.Tensor = y[idx] - Q[idx]
        q = q[..., 0]

        if self.censored:

            bellow: torch.Tensor = Q[idx] <= 0
            
            #print(bellow.sum()/Q[idx].numel())
            #print(Q[0])

            #

            dif[Q[idx] <= 0] = torch.nan

        i0: torch.Tensor = dif <  0.0
        i1: torch.Tensor = dif >= 0.0

        dif[i0] *= q[idx][i0] - 1.0
        dif[i1] *= q[idx][i1]

        return dif.nansum(dim = -1).nanmean()
        #return dif.nanmean(dim = 0).nansum()
 

