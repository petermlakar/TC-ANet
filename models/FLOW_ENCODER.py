import torch
import torch.nn as nn
import torch.jit as jit

from typing import Tuple, List, Optional
import math

def get_lead_time_encodings(nlead: int, size: int):
    """
    Generate lead-time embeddings using a correlated random walk.

    Parameters
    ----------
    nlead : int
        Number of lead times to create encodings for.
    size : int
        Dimensionality of each lead-time encoding vector.

    Returns
    -------
    torch.Tensor
        Tensor of shape [nlead, size] containing lead-time encodings.
        Consecutive lead times are generated from the previous encoding
        with added Gaussian noise, producing smooth transitions between
        neighboring lead times.
    """

    e: torch.Tensor = torch.zeros((nlead, size), dtype = torch.float32)
    a: float = 0.5 

    e[0, :] = torch.randn((size,), dtype = torch.float32)

    for i in range(1,  nlead):
        e[i, :] = e[i -1, :] + torch.randn((size,), dtype = torch.float32)*a
        e[i, :] /= math.sqrt(1 + a*a)

    return e
        
class DropoutInf(nn.Module):

    def __init__(self, p: float = 0.1):
        """
        Create a dropout layer that replaces selected values with -inf.

        Parameters
        ----------
        p : float, optional
            Probability of masking an element. Default is 0.1.

        Notes
        -----
        This layer is intended for use before a softmax operation, where
        subtracting infinity effectively removes elements from attention.
        """

        super().__init__()

        self.p: float = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout by replacing randomly selected values with -inf.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor with randomly selected values replaced by -inf when
            the module is in training mode. Returns the input unchanged
            during evaluation.
        """

        if self.training:

            i: torch.Tensor = torch.rand_like(x) <= self.p
            x = x.clone()
            x[i] -= torch.inf

        return x

class MultiheadAttentionEncoder(nn.Module):

    def __init__(self, 
                 number_of_features: int,
                 number_of_heads: int = 4,
                 dropout: float = 0.1):
        """
        Multi-head attention layer using externally supplied attention scores.

        Parameters
        ----------
        number_of_features : int
            Feature dimension of the input and output tensors.
        number_of_heads : int, optional
            Number of attention heads. Must evenly divide
            number_of_features.
        dropout : float, optional
            Probability used by DropoutInf to mask attention logits.
        """
        super().__init__()

        assert (number_of_features % number_of_heads) == 0

        self.number_of_heads = number_of_heads
        self.number_of_features = number_of_features

        self.v_weights = nn.Linear(number_of_features, number_of_features, bias = False)

        self.f = nn.Linear(number_of_features, number_of_features, bias = False)

        self.d = DropoutInf(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self,
                v: torch.Tensor,
                e: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention using precomputed attention logits.

        Parameters
        ----------
        v : torch.Tensor
            Value tensor of shape [batch, sequence, features].
        e : torch.Tensor
            Attention logits tensor. The final dimension is normalized
            with softmax to produce attention weights.

        Returns
        -------
        torch.Tensor
            Attention output tensor with the same feature dimension as
            the input.
        """
       
        V = self.v_weights(v)
        V = V.view(V.shape[0], V.shape[1], self.number_of_heads, V.shape[2]//self.number_of_heads).swapaxes(-2, -3)

        W = torch.softmax(self.d(e), dim = -1)

        return self.f((W @ V).swapaxes(-2, -3).flatten(start_dim = 2))

class Encoder(nn.Module):

    def __init__(self,
                 number_of_features: int,
                 number_of_heads: int,
                 dropout: float):
        """
        Transformer-style encoder block consisting of attention and
        a residual feed-forward layer.

        Parameters
        ----------
        number_of_features : int
            Feature dimension of the hidden representation.
        number_of_heads : int
            Number of attention heads.
        dropout : float
            Attention dropout probability.
        """

        super().__init__()

        self.atn = MultiheadAttentionEncoder(number_of_features, number_of_heads, dropout)
        self.res = nn.Sequential(nn.SiLU(), nn.Linear(number_of_features, number_of_features)) 

    def forward(self,
                x: torch.Tensor,
                e: torch.Tensor) -> torch.Tensor:
        """
        Apply attention and feed-forward residual updates.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.
        e : torch.Tensor
            Attention logits supplied to the attention layer.

        Returns
        -------
        torch.Tensor
            Updated feature tensor after attention and residual
            feed-forward transformations.
        """

        x = x + self.atn(x, e)

        return x + self.res(x)

#####################################

#
# a_i = S_j^L w_j*f(f_j)
#
# Why do we need to know which lead time we a reforecasting for?
#
#


class ForecastEncoder(nn.Module):

    def __init__(self, 
                 number_of_forecast_fields: int,

                 number_of_input_lead_times: int, 
                 number_of_output_lead_times: int,

                 number_of_stations: int,

                 nencodings_per_station: int,
                 nencodings_per_lead_time: int,

                 number_of_features: int,
                 number_of_output_features: int,
                 number_of_encoders: int,
                 number_of_heads: int):
        """
        Forecast encoder that aggregates ensemble forecasts across lead
        times using learned attention mechanisms.

        Parameters
        ----------
        number_of_forecast_fields : int
            Number of forecast variables provided as input.
        number_of_input_lead_times : int
            Number of lead times available in the input forecast.
        number_of_output_lead_times : int
            Number of lead times to generate representations for.
        number_of_stations : int
            Total number of stations in the dataset.
        nencodings_per_station : int
            Dimension of the station embedding.
        nencodings_per_lead_time : int
            Dimension of the lead-time embedding.
        number_of_features : int
            Internal feature dimension used by encoder blocks.
        number_of_output_features : int
            Size of the final output representation.
        number_of_encoders : int
            Number of encoder blocks to stack.
        number_of_heads : int
            Number of attention heads used in each encoder block.
        """

        super().__init__()

        self.number_of_heads = number_of_heads
        self.number_of_features = number_of_features
        self.number_of_output_features = number_of_output_features
        self.number_of_output_lead_times = number_of_output_lead_times
        self.number_of_input_lead_times = number_of_input_lead_times

        self.nencodings_per_station = nencodings_per_station
        self.nencodings_per_lead_time = nencodings_per_lead_time

        #self.encodings_lead_time = nn.Parameter(torch.zeros((self.number_of_output_lead_times, self.nencodings_per_lead_time), dtype = torch.float32), requires_grad = True)
        self.encodings_lead_time = nn.Parameter(get_lead_time_encodings(number_of_output_lead_times, nencodings_per_lead_time), requires_grad = True)
        self.encodings_stations = nn.Parameter(torch.randn((number_of_stations, self.nencodings_per_station), dtype = torch.float32), requires_grad = True)

        self.forecast_to_embedding = nn.Linear(2*number_of_forecast_fields + 2 + self.nencodings_per_station + self.nencodings_per_lead_time, number_of_features) 

        predictor_count: int = 2*self.nencodings_per_lead_time + self.nencodings_per_station + 2
        self.encodings_to_attention = nn.Sequential(nn.Linear(predictor_count, 32),
                                                    nn.SiLU(),
                                                    nn.Linear(32, 1))

        self.encoders = nn.ModuleList()
        for i in range(number_of_encoders):
            self.encoders.append(Encoder(number_of_features, number_of_heads, 0.1))

        self.encoder_out: nn.Module = nn.Sequential(nn.SiLU(), nn.Linear(number_of_features, number_of_output_features)) if number_of_features != number_of_output_features else nn.Identity()

        self.e: torch.Tensor = torch.empty((1,))

    @jit.export
    def get_encodings(self,
                      day_of_year: torch.Tensor,
                      station_index: torch.Tensor) -> torch.Tensor:
        """
        Construct station and seasonal encodings.

        Parameters
        ----------
        day_of_year : torch.Tensor
            Tensor containing normalized day-of-year values. Shape:
            [batch, ...].
        station_index : torch.Tensor
            Station indices for each sample. Shape: [batch].

        Returns
        -------
        torch.Tensor
            Encoding tensor of shape
            [batch, number_of_output_lead_times, features]
            containing cyclic seasonal encodings and station embeddings.
        """

        # day_of_year, station_index: [batch]

        # Remap run times
        day_of_year = day_of_year[:, 0]

        # Construct encodings tensor
        encodings = torch.cat([torch.cos(2*math.pi*day_of_year)[..., None],
                               torch.sin(2*math.pi*day_of_year)[..., None],
                               self.encodings_stations[station_index]], dim = -1)

        return encodings[:, None].expand(-1, self.number_of_output_lead_times, -1) 

    def forward(self,
                forecast: torch.Tensor,
                day_of_year: torch.Tensor,
                station_index: torch.Tensor) -> torch.Tensor:
        """
        Encode ensemble forecast information into a learned feature space.

        Parameters
        ----------
        forecast : torch.Tensor
            Forecast tensor containing ensemble member predictions.
            Expected dimensions include batch, lead time, member,
            and forecast-variable axes.
        day_of_year : torch.Tensor
            Normalized day-of-year values used to generate seasonal
            encodings.
        station_index : torch.Tensor
            Integer station identifiers used to retrieve station
            embeddings.

        Returns
        -------
        torch.Tensor
            Encoded forecast representation of shape
            [batch, number_of_output_lead_times,
             number_of_output_features].
        """

        batch: int = forecast.shape[0]
        member: int = forecast.shape[2]

        fmu: torch.Tensor = forecast.mean(dim = 2)
        fsd: torch.Tensor = forecast.std(dim = 2)

        encodings_s: torch.Tensor = self.get_encodings(day_of_year, station_index)
        encodings_l: torch.Tensor = self.encodings_lead_time[None].expand(batch, -1, -1)
   
        e_predictors: torch.Tensor = torch.cat([encodings_l[:, :, None].expand(-1, -1, self.number_of_input_lead_times, -1),
                                                encodings_l[:, None].expand(-1, self.number_of_input_lead_times, -1, -1),
                                                encodings_s[:, None].expand(-1, self.number_of_input_lead_times, -1, -1)], dim = -1)


        e: torch.Tensor = self.encodings_to_attention(e_predictors)[:, None].swapaxes(1, -1)[..., 0]
        
        self.e = e

        x: torch.Tensor = self.forecast_to_embedding(torch.cat([fmu, fsd, encodings_s, encodings_l], dim = -1))

        for f in self.encoders:
            x = f(x, e)

        return self.encoder_out(x)
