import torch
import torch.nn as nn
import torch.jit as jit

from typing import Tuple, List

import math

class Model(nn.Module):
    """
    Ensemble post-processing model that predicts calibrated ensemble
    samples for a target forecast variable.

    The model combines forecast ensemble statistics, seasonal encodings,
    station embeddings, and lead-time embeddings to learn an affine
    transformation of the target forecast ensemble. Training is performed
    using the Continuous Ranked Probability Score (CRPS).
    """

    def __init__(self,
                 target_field_index: int,
                 number_of_forecast_fields: int,
                 
                 number_of_lead_times: int,
                 number_of_stations: int,

                 nembeddings_per_lead_time: int,
                 nembeddings_per_station: int,

                 number_of_features: int,
                
                 censored: bool = False):
        """
        Initialize the model.

        Args:
            target_field_index: Index of the forecast variable to be
                calibrated.
            number_of_forecast_fields: Number of forecast variables.
            number_of_lead_times: Number of forecast lead times.
            number_of_stations: Number of stations/locations.
            nembeddings_per_lead_time: Size of the learned lead-time
                embedding.
            nembeddings_per_station: Size of the learned station embedding.
            number_of_features: Hidden layer width of the neural network.
            censored: Unused flag retained for interface compatibility.
        """

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
        """
        Construct seasonal and station-specific encodings.

        Seasonal information is represented using sine and cosine
        transformations of the day-of-year value and concatenated with
        a learned station embedding.

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
        Generate calibrated ensemble predictions.

        Forecast ensemble means and standard deviations are combined with
        station, lead-time, and seasonal encodings. The network predicts
        parameters that shift and scale the target ensemble members.

        Args:
            forecast: Forecast ensemble tensor.
            day_of_year: Normalized day-of-year values.
            station_index: Station indices.

        Returns:
            Calibrated ensemble predictions for the target variable.
        """

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
        """
        Compute the CRPS objective used for training.

        The loss is calculated using the ensemble approximation of CRPS,
        consisting of the mean absolute error term minus half of the
        expected pairwise ensemble-member distance.

        Args:
            forecast: Forecast ensemble tensor.
            day_of_year: Normalized day-of-year values.
            station_index: Station indices.
            observations: Observed target values.

        Returns:
            Scalar CRPS loss.
        """

        res: torch.Tensor = self(forecast, day_of_year, station_index)

        M: int = res.shape[-1]

        res = res.flatten(end_dim = 1)
        observations = observations.flatten(end_dim = 1)

        t0: torch.Tensor = (res - observations).abs().mean(dim = -1)
        t1: torch.Tensor = (res[..., None] - res[:, None]).abs().mean(dim = (-1, -2))*0.5

        return (t0 - t1).mean()
