import torch
import torch.nn as nn
import torch.jit as jit

from typing import Tuple, List, Optional

import math

class Model(nn.Module):

    def __init__(self,
                 number_of_forecast_fields: int,
                 
                 number_of_lead_times: int,
                 number_of_stations: int,

                 nembeddings_per_station: int,
                 nembeddings_per_lead_time: int,

                 number_of_features: int,
                
                 censored: bool,

                 model_marginal: nn.Module):

        super().__init__()

        self.embeddings_station = nn.Parameter(torch.randn(number_of_stations, nembeddings_per_station, dtype = torch.float32), requires_grad = True)
        self.embeddings_lead_time = nn.Parameter(torch.randn(number_of_lead_times, nembeddings_per_lead_time, dtype = torch.float32), requires_grad = True)

        self.number_of_lead_times = number_of_lead_times

        self.lead_time_modules = nn.ModuleList([])

        self.f = nn.Sequential(nn.Linear(number_of_forecast_fields*2 + 2 + nembeddings_per_station + nembeddings_per_lead_time, number_of_features),
                               nn.ReLU(),
                               nn.Linear(number_of_features, number_of_features),
                               nn.ReLU(),
                               nn.Linear(number_of_features, model_marginal.number_of_required_parameters))

        self.model_marginal = model_marginal
        self.model_marginal.censored = censored

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

        x: torch.Tensor = torch.cat([xmu, xsd, encodings_s, encodings_l], dim = -1)

        return self.f(x)

    @jit.ignore
    def loss(self, forecast: torch.Tensor, day_of_year: torch.Tensor, station_index: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:

        parameters: torch.Tensor = self(forecast, day_of_year, station_index)

        L: torch.Tensor = self.model_marginal.loss(observations, parameters)#.view(observations.shape)

        #return L.nanmean()
        return L
