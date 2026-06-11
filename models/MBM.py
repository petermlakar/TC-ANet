import torch
import torch.nn as nn
import torch.jit as jit
from torch.distributions import Gamma

from typing import Tuple, List

import math

class Model(nn.Module):

    def __init__(self,
                 target_field_index: int,
                 number_of_forecast_fields: int,
                 
                 number_of_lead_times: int,
                 number_of_stations: int,

                 nembeddings_per_lead_time: int,
                 nembeddings_per_station: int,

                 number_of_features: int,
                
                 censored: bool = False):

        super().__init__()

        self.target_field_index = target_field_index

        self.embeddings_station   = nn.Parameter(torch.randn(number_of_stations, nembeddings_per_station, dtype = torch.float32), requires_grad = True)
        self.embeddings_lead_time = nn.Parameter(torch.randn(number_of_lead_times, nembeddings_per_lead_time, dtype = torch.float32), requires_grad = True)

        self.number_of_lead_times = number_of_lead_times

        self.f = nn.Sequential(nn.Linear(number_of_forecast_fields*2 + 2 + nembeddings_per_station + nembeddings_per_lead_time, number_of_features),
                               nn.ReLU(),
                               nn.Linear(number_of_features, number_of_features),
                               nn.ReLU(),
                               nn.Linear(number_of_features, 3))

    @jit.export
    def get_encodings(self,
                      day_of_year: torch.Tensor,
                      station_index: torch.Tensor) -> torch.Tensor:

        day_of_year = day_of_year[:, 0]

        # Construct encodings tensor
        return torch.cat([torch.cos(2*math.pi*day_of_year)[..., None],
                          torch.sin(2*math.pi*day_of_year)[..., None],
                          self.embeddings_station[station_index]], dim = -1)

    def forward(self, forecast: torch.Tensor, day_of_year: torch.Tensor, station_index: torch.Tensor) -> torch.Tensor:

        xmu: torch.Tensor = forecast.mean(dim = 2)
        xsd: torch.Tensor = forecast.std(dim = 2)

        encodings_s: torch.Tensor = self.get_encodings(day_of_year, station_index)[:, None].expand(-1, self.number_of_lead_times, -1)
        encodings_l: torch.Tensor = self.embeddings_lead_time[None].expand(forecast.shape[0], -1, -1)

        encodings: torch.Tensor = torch.cat([encodings_s, encodings_l], dim = -1)

        x: torch.Tensor = torch.cat([xmu, xsd, encodings], dim = -1)
        
        # Target forecast

        p: torch.Tensor = self.f(x)[:, :, None]
        xt: torch.Tensor = forecast[..., self.target_field_index]
        xt_mu: torch.Tensor = xmu[..., self.target_field_index][..., None]

        return p[..., 0] + p[..., 1]*xt_mu + nn.functional.softplus(p[..., 2])*(xt - xt_mu)

    @jit.ignore
    def loss(self, forecast: torch.Tensor, day_of_year: torch.Tensor, station_index: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:

        res: torch.Tensor = self(forecast, day_of_year, station_index)

        M: int = res.shape[-1]

        res = res.flatten(end_dim = 1)
        observations = observations.flatten(end_dim = 1)

        t0: torch.Tensor = (res - observations).abs().mean(dim = -1)
        t1: torch.Tensor = (res[..., None] - res[:, None]).abs().mean(dim = (-1, -2))*0.5

        return (t0 - t1).mean()

    @jit.ignore
    def loss_es(self, forecast: torch.Tensor, day_of_year: torch.Tensor, station_index: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:

        res: torch.Tensor = self(forecast, day_of_year, station_index)

        M: int = res.shape[-1]

        #i: torch.Tensor = torch.tril_indices(M, M, offset = -1, device = res.device)

        t0: torch.Tensor = torch.linalg.vector_norm(res - observations, dim = 1).mean(dim = -1)

        #t1: torch.Tensor = 0.5*(res[..., None] - res[:, :, None]).pow(2).clip(min = 1e-16, max = None).sum(dim = 1).sqrt()#.mean(dim = (-1, -2))
        t1: torch.Tensor = 0.5*torch.linalg.vector_norm((res[..., None] - res[:, :, None]), dim = 1).nanmean(dim = (-1, -2))

        return (t0 - t1).mean()

