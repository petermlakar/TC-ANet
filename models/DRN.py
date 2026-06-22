import torch
import torch.nn as nn
import torch.jit as jit

from typing import Tuple, List, Optional

import math

class Model(nn.Module):
    """
    Feed-forward marginal forecasting model that combines forecast
    ensemble statistics with station, lead-time, and seasonal
    encodings to predict the parameters of a marginal distribution.
    """

    def __init__(self,
                 number_of_forecast_fields: int,
                 
                 number_of_lead_times: int,
                 number_of_stations: int,

                 nembeddings_per_station: int,
                 nembeddings_per_lead_time: int,

                 number_of_features: int,
                
                 censored: bool,

                 model_marginal: nn.Module):
        """
        Initialize the model.

        Args:
            number_of_forecast_fields: Number of forecast variables.
            number_of_lead_times: Number of forecast lead times.
            number_of_stations: Number of stations/locations.
            nembeddings_per_station: Size of the learned station embedding.
            nembeddings_per_lead_time: Size of the learned lead-time embedding.
            number_of_features: Hidden layer width of the prediction network.
            censored: Whether the marginal model supports censored targets.
            model_marginal: Marginal distribution model used for loss
                computation and parameter interpretation.
        """

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
        """
        Construct seasonal and station-specific encodings.

        The seasonal encoding uses sine and cosine transforms of the
        day-of-year value, which are concatenated with a learned station
        embedding.

        Args:
            day_of_year: Normalized day-of-year values.
            station_index: Station indices.

        Returns:
            Tensor containing seasonal and station encodings.
        """

        day_of_year = day_of_year[:, 0]

        # Construct encodings tensor
        return torch.cat([torch.cos(2*math.pi*day_of_year)[..., None],
                          torch.sin(2*math.pi*day_of_year)[..., None],
                          self.embeddings_station[station_index]], dim = -1)

    def forward(self, forecast: torch.Tensor, day_of_year: torch.Tensor, station_index: torch.Tensor) -> torch.Tensor:
        """
        Predict marginal distribution parameters from forecast inputs.

        Ensemble means and standard deviations are computed for each
        forecast field and combined with seasonal, station, and lead-time
        embeddings before being passed through a feed-forward network.

        Args:
            forecast: Forecast ensemble tensor.
            day_of_year: Normalized day-of-year values.
            station_index: Station indices.

        Returns:
            Tensor of predicted marginal distribution parameters.
        """

        xmu: torch.Tensor = forecast.mean(dim = 2)
        xsd: torch.Tensor = forecast.std(dim = 2)

        encodings_s: torch.Tensor = self.get_encodings(day_of_year, station_index)[:, None].expand(-1, self.number_of_lead_times, -1)
        encodings_l: torch.Tensor = self.embeddings_lead_time[None].expand(forecast.shape[0], -1, -1)

        x: torch.Tensor = torch.cat([xmu, xsd, encodings_s, encodings_l], dim = -1)

        return self.f(x)

    @jit.ignore
    def loss(self, forecast: torch.Tensor, day_of_year: torch.Tensor, station_index: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute the marginal-model loss.

        Args:
            forecast: Forecast ensemble tensor.
            day_of_year: Normalized day-of-year values.
            station_index: Station indices.
            observations: Observed target values.

        Returns:
            Loss tensor produced by the marginal model.
        """

        parameters: torch.Tensor = self(forecast, day_of_year, station_index)

        return self.model_marginal.loss(observations, parameters)
