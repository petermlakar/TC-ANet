import torch
import torch.nn as nn
import torch.jit as jit

from typing import Tuple, List, Optional
import math

class DecoderMarginal(nn.Module):
    """
    Decoder network that maps forecast encoder embeddings to the
    parameter space required by the marginal distribution model.
    """

    def __init__(self,
                 number_of_input_features: int,
                 number_of_features: int,
                 number_of_outputs: int):
        """
        Initialize the marginal decoder.

        Args:
            number_of_input_features: Size of the input embedding vector.
            number_of_features: Hidden layer size.
            number_of_outputs: Number of marginal distribution parameters
                to predict.
        """
        super().__init__()

        self.f = nn.Sequential(nn.SiLU(), nn.Linear(number_of_input_features, number_of_features), nn.SiLU(), nn.Linear(number_of_features, number_of_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute marginal model parameters from embeddings.

        Args:
            x: Input embedding tensor.

        Returns:
            Tensor containing predicted marginal parameters.
        """
        return self.f(x)

class Model(nn.Module):
    """
    Combined model consisting of:

    - A forecast encoder that produces embeddings.
    - A marginal model that predicts the distribution of individual targets.
    - A joint model that generates samples through a learned dynamics process.

    Optionally supports censored targets by modeling both continuous and
    discrete components.
    """

    def __init__(self,
                 forecast_encoder: nn.Module,
                 model_marginal: nn.Module,
                 model_joint: nn.Module,
                 marginal_internal_features: int,
                 censored: bool = False):
        """
        Initialize the full model.

        Args:
            forecast_encoder: Module that encodes forecast inputs into
                latent embeddings.
            model_marginal: Marginal distribution model.
            model_joint: Joint dynamics model used for sampling.
            marginal_internal_features: Hidden size of the marginal decoder.
            censored: Whether the target contains a censored/discrete
                component.
        """

        super().__init__()

        self.forecast_encoder = forecast_encoder
        self.model_marginal = model_marginal
        self.model_joint = model_joint

        self.model_marginal.censored = censored
        self.censored = censored

        number_of_features: int = forecast_encoder.number_of_output_features

        # Marginal model decoder
        self.decoder_marginal = DecoderMarginal(number_of_features, marginal_internal_features,  model_marginal.number_of_required_parameters)

    @jit.ignore
    def freeze_encoder(self, freeze: bool = True):
        """
        Enable or disable gradient updates for the forecast encoder.

        Args:
            freeze: If True, encoder parameters are frozen.
        """

        for _, w in self.forecast_encoder.named_parameters():
            if w.requires_grad is not None:
                w.requires_grad = not freeze

    @jit.ignore
    def freeze_marginal(self, freeze: bool = True):
        """
        Enable or disable gradient updates for the marginal model and
        its decoder.

        Args:
            freeze: If True, marginal-model parameters are frozen.
        """

        for _, w in self.model_marginal.named_parameters():
            if w.requires_grad is not None:
                w.requires_grad = not freeze

        for name, w in self.decoder_marginal.named_parameters():
            if w.requires_grad is not None:
                w.requires_grad = not freeze

    @jit.ignore
    def freeze_joint(self, freeze: bool = True):
        """
        Enable or disable gradient updates for the joint model.

        Args:
            freeze: If True, joint-model parameters are frozen.
        """

        for _, w in self.model_joint.named_parameters():
            if w.requires_grad is not None:
                w.requires_grad = not freeze

    @jit.export
    def get_embeddings(self,
                       forecast: torch.Tensor,
                       day_of_year: torch.Tensor,
                       station_index: torch.Tensor) -> torch.Tensor:
        """
        Generate latent embeddings from forecast inputs.

        Args:
            forecast: Forecast feature tensor.
            day_of_year: Day-of-year indicator tensor.
            station_index: Station identifier tensor.

        Returns:
            Embedding tensor produced by the forecast encoder.
        """

        return self.forecast_encoder(forecast, day_of_year, station_index)

    @jit.export
    def init_marginal(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Convert embeddings into marginal distribution parameters.

        Args:
            embeddings: Encoder output embeddings.

        Returns:
            Predicted marginal distribution parameters.
        """
        return self.decoder_marginal(embeddings)

    @jit.ignore
    def loss_marginal(self,
                      embeddings: torch.Tensor,
                      observations: torch.Tensor) -> torch.Tensor:
        """
        Compute the marginal-model loss.

        Args:
            embeddings: Encoder embeddings.
            observations: Observed target values.

        Returns:
            Marginal loss tensor.
        """
        
        return self.model_marginal.loss(observations, self.init_marginal(embeddings))

    @jit.ignore
    def loss_joint(self,
                  embeddings: torch.Tensor,
                  t: torch.Tensor,
                  yt: torch.Tensor,
                  v: torch.Tensor,
                  y1: torch.Tensor) -> torch.Tensor:
        """
        Compute the joint-model training loss.

        For uncensored targets, returns a regression loss.
        For censored targets, returns both regression and classification
        losses.

        Args:
            embeddings: Encoder embeddings.
            t: Current time values.
            yt: State values at time t.
            v: Target velocity/drift values.
            y1: Target endpoint state.

        Returns:
            Regression loss, or a tuple of regression and classification
            losses when censoring is enabled.
        """

        p: torch.Tensor = self.model_joint(embeddings, t, yt)

        L_reg: torch.Tensor = (p[..., 0] - v[..., 0]).pow(2)

        if self.censored:
            
            I: torch.Tensor = torch.nn.functional.one_hot(y1[..., 0].type(torch.int64), 2).type(torch.float32)
            P: torch.Tensor = torch.stack([torch.nn.functional.logsigmoid(p[..., 1]), -p[..., 1] + torch.nn.functional.logsigmoid(p[..., 1])], dim = -1)

            L_cls: torch.Tensor = -(I*P).sum(dim = -1)
           
            return L_reg, L_cls

        return L_reg

    @jit.export
    def step_joint(self,
                   embeddings: torch.Tensor,
                   t0: torch.Tensor,
                   t1: torch.Tensor,
                   yt: torch.Tensor) -> torch.Tensor:
        """
        Advance the joint process by a single integration step.

        The continuous component is updated using the predicted drift.
        When censoring is enabled, the discrete component is updated
        using the predicted class probabilities.

        Args:
            embeddings: Encoder embeddings.
            t0: Start time of the step.
            t1: End time of the step.
            yt: Current state.

        Returns:
            Updated state after one step.
        """

        t0 = t0.view(1, 1, 1).expand(yt.shape[0], -1, -1)
        h: torch.Tensor = (t1 - t0)

        p: torch.Tensor = self.model_joint(embeddings, t0, yt)

        yt[..., 0] += h[..., 0]*p[..., 0]

        if self.censored:
            p_d: torch.Tensor = torch.stack([torch.nn.functional.sigmoid(p[..., 1]), 1 - torch.nn.functional.sigmoid(p[..., 1])], dim = -1)

            onehot: torch.Tensor = torch.nn.functional.one_hot(yt[..., 1].long(), 2).float()
            u: torch.Tensor = (p_d - onehot)/(1.0 - t0)

            yt[..., 1] = ((onehot + h*u).cumsum(dim = -1) < torch.rand_like(p_d)).sum(dim = -1).clamp(max = 1)

        return yt

    @jit.export
    def sample_joint(self,
                     number_of_samples: int,
                     number_of_steps: int,
                     embeddings: torch.Tensor,
                     y0_c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate samples from the joint model by iteratively applying
        the learned dynamics from time 0 to 1.

        Args:
            number_of_samples: Number of samples to generate per input.
            number_of_steps: Number of integration steps.
            embeddings: Encoder embeddings.
            y0_c: Optional initial continuous state. If not provided,
                standard normal noise is used.

        Returns:
            Sampled trajectories or terminal states in the model's
            output format.
        """
        
        batch: int = embeddings.shape[0]
        lead: int = embeddings.shape[1]

        time_steps: torch.Tensor = torch.linspace(0.0, 1.0, number_of_steps + 1, device = embeddings.device)

        if y0_c is None:
            y0_c = torch.randn((batch, lead, number_of_samples), dtype = torch.float32, device = embeddings.device)

        if self.censored:
            y0_d: torch.Tensor = (y0_c > 0.0).type(torch.float32)
            y0: torch.Tensor = torch.stack([y0_c, y0_d], dim = -1)
        else:
            y0: torch.Tensor = y0_c[..., None]

        y0 = y0.swapaxes(1, 2).flatten(end_dim = 1)
        embeddings = embeddings[:, None].expand(-1, number_of_samples, -1, -1).flatten(end_dim = 1)

        for k in range(number_of_steps):
            y0 = self.step_joint(embeddings, time_steps[k], time_steps[k + 1], y0)

        return y0.view(batch, number_of_samples, lead, -1)[..., None].swapaxes(1, -1).swapaxes(-2, -1)[:, 0]
