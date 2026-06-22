import torch
import torch.nn as nn
import torch.jit as jit

from typing import Tuple

class Block(nn.Module):
    """
    Basic convolutional processing block consisting of an optional
    SiLU activation followed by a 1D convolution.

    This block is used as a building component within the joint model
    decoder to process sequential feature representations.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: bool = True):
        """
        Initialize the convolutional block.

        Args:
            in_features: Number of input channels.
            out_features: Number of output channels.
            activation: If True, apply a SiLU activation before the
                convolution. Otherwise use the identity transform.
        """

        super().__init__()

        self.f0: nn.Module = nn.Sequential(
            nn.SiLU() if activation else nn.Identity(),
            nn.Conv1d(in_features, out_features, 3, padding = 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolutional processing block.

        Args:
            x: Input tensor of shape [batch, channels, length].

        Returns:
            Processed tensor after activation and convolution.
        """
        return self.f0(x)

class ModelJoint(nn.Module):
    """
    Joint conditional dynamics model used to evolve state variables
    through time.

    The model combines forecast embeddings, time information, and
    current state values to predict the parameters governing the
    next state transition.
    """

    def __init__(self,
                 incoming_features: int,
                 internal_features: int,
                 number_of_outputs: int,
                 features: int = 128,
                 input_vectors: int = 2):
        """
        Initialize the joint model.

        Args:
            incoming_features: Number of features in the incoming
                embedding representation.
            internal_features: Internal embedding dimension used by
                the model.
            number_of_outputs: Number of output values predicted by
                the model.
            features: Intermediate feature dimension used within the
                convolutional decoder.
            input_vectors: Number of additional input state variables
                concatenated to the embeddings.
        """

        super().__init__()
       
        features: int = internal_features + input_vectors

        self.decoder_joint = nn.Sequential(
            Block(features, features,
                  activation = incoming_features != internal_features),
            Block(features, features),
            Block(features, features))

        self.i: nn.Module = (
            nn.Linear(incoming_features, internal_features)
            if incoming_features != internal_features
            else nn.Identity())

        self.f: nn.Module = nn.Sequential(
            nn.SiLU(),
            nn.Linear(features + internal_features, features),
            nn.SiLU(),
            nn.Linear(features, number_of_outputs))

    def forward(self,
                embeddings: torch.Tensor,
                t: torch.Tensor,
                yt: torch.Tensor) -> torch.Tensor:
        """
        Predict joint transition parameters for the current time step.

        Forecast embeddings are combined with the time step value
        and current state variables, processed through a
        convolutional decoder, and mapped to the requested output
        parameters.

        Args:
            embeddings: Forecast embedding tensor with shape
                [batch, lead, features].
            t: Current time step.
            yt: Current state tensor.

        Returns:
            Predicted transition parameters for the next update step.
        """

        batch, lead, features = embeddings.shape

        embeddings = self.i(embeddings)
       
        x: torch.Tensor = torch.cat([embeddings, t.expand(-1, lead, -1), yt], dim = -1).swapaxes(-1, -2)

        return self.f(torch.cat([self.decoder_joint(x).swapaxes(-1, -2), embeddings], dim = -1))
