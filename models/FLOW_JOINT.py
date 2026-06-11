import torch
import torch.nn as nn
import torch.jit as jit

from typing import Tuple

class Block(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: bool = True):

        super().__init__()

        self.f0: nn.Module = nn.Sequential(nn.SiLU() if activation else nn.Identity(), nn.Conv1d(in_features, out_features, 3, padding = 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f0(x)

class ModelJoint(nn.Module):

    def __init__(self,
                 incoming_features: int,
                 internal_features: int,
                 number_of_outputs: int,
                 features: int = 128,
                 input_vectors: int = 2):

        super().__init__()
       
        
        features: int = internal_features + input_vectors

        self.decoder_joint = nn.Sequential(Block(features, features, activation = incoming_features != internal_features), 
                                           Block(features, features),
                                           Block(features, features))

        self.i: nn.Module = nn.Linear(incoming_features, internal_features) if incoming_features != internal_features else nn.Identity()
        self.f: nn.Module = nn.Sequential(nn.SiLU(), nn.Linear(features + internal_features, features), nn.SiLU(), nn.Linear(features, number_of_outputs))

    def forward(self,
                embeddings: torch.Tensor,
                t: torch.Tensor,
                yt: torch.Tensor) -> torch.Tensor:

        batch, lead, features = embeddings.shape

        embeddings = self.i(embeddings)
       
        x: torch.Tensor = torch.cat([embeddings, t.expand(-1, lead, -1), yt], dim = -1).swapaxes(-1, -2)

        return self.f(torch.cat([self.decoder_joint(x).swapaxes(-1, -2), embeddings], dim = -1))

