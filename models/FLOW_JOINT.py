import torch
import torch.nn as nn
import torch.jit as jit

from typing import Tuple

class Block(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: bool = True):
        """
        Convolutional processing block consisting of an optional SiLU
        activation followed by a 1D convolution.

        Parameters
        ----------
        in_features : int
            Number of input channels to the convolution.
        out_features : int
            Number of output channels produced by the convolution.
        activation : bool, optional
            If True, applies a SiLU activation before the convolution.
            If False, no activation is applied. Default is True.
        """

        super().__init__()

        self.f0: nn.Module = nn.Sequential(
            nn.SiLU() if activation else nn.Identity(),
            nn.Conv1d(in_features, out_features, 3, padding = 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolutional block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, channels, sequence_length].

        Returns
        -------
        torch.Tensor
            Output tensor after activation (if enabled) and convolution.
        """
        return self.f0(x)

class ModelJoint(nn.Module):

    def __init__(self,
                 incoming_features: int,
                 internal_features: int,
                 number_of_outputs: int,
                 features: int = 128,
                 input_vectors: int = 2):
        """
        Joint decoder model that combines learned embeddings, time
        information, and auxiliary input vectors to produce output
        predictions.

        Parameters
        ----------
        incoming_features : int
            Feature dimension of the input embeddings.
        internal_features : int
            Internal embedding dimension used within the model.
        number_of_outputs : int
            Number of output values produced for each lead time.
        features : int, optional
            Unused argument retained for compatibility. The effective
            feature count is computed from internal_features and
            input_vectors. Default is 128.
        input_vectors : int, optional
            Number of auxiliary input features concatenated with the
            embeddings. Default is 2.
        """

        super().__init__()
       
        features: int = internal_features + input_vectors

        self.decoder_joint = nn.Sequential(
            Block(features, features,
                  activation = incoming_features != internal_features),
            Block(features, features),
            Block(features, features)
        )

        self.i: nn.Module = (
            nn.Linear(incoming_features, internal_features)
            if incoming_features != internal_features
            else nn.Identity()
        )

        self.f: nn.Module = nn.Sequential(
            nn.SiLU(),
            nn.Linear(features + internal_features, features),
            nn.SiLU(),
            nn.Linear(features, number_of_outputs)
        )

    def forward(self,
                embeddings: torch.Tensor,
                t: torch.Tensor,
                yt: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions from embeddings and auxiliary inputs.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input embedding tensor of shape
            [batch, lead_times, incoming_features].
        t : torch.Tensor
            Time-dependent features. Expected to be broadcastable across
            the lead-time dimension and concatenated with the embeddings.
        yt : torch.Tensor
            Additional per-lead-time input features of shape compatible
            with the embedding tensor.

        Returns
        -------
        torch.Tensor
            Prediction tensor of shape
            [batch, lead_times, number_of_outputs].
        """

        batch, lead, features = embeddings.shape

        embeddings = self.i(embeddings)
       
        x: torch.Tensor = torch.cat(
            [embeddings, t.expand(-1, lead, -1), yt],
            dim = -1
        ).swapaxes(-1, -2)

        return self.f(
            torch.cat(
                [self.decoder_joint(x).swapaxes(-1, -2), embeddings],
                dim = -1
            )
        )
