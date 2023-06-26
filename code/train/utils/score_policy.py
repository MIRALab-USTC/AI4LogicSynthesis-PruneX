import torch
import numpy as np
from torch import nn
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)
import os
import sys
# sys.path.insert(0, os.getcwd())
from utils.mlp import MLP

ModuleType = Type[nn.Module]


class ScorePolicy(nn.Module):
    """
    Simple cut selection policy based on simple mlp.

    :param feature_size: the size of cut feature vector.
    :param out_size: the size of output score, default=1.
    :param hidden_sizes: the size of hidden layers of mlp.
    :param device: the device of the model and tensor. 
    """

    def __init__(
        self,
        feature_size: int = 13,
        out_size: int = 1,
        hidden_sizes: Sequence[int] = (),
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.Sigmoid,
        device: Union[str, int, torch.device] = "cpu",
        mean_max='mean'
    ) -> None:
        super().__init__()
        self.device = device
        self.mean_max = mean_max
        self.model = MLP(
            feature_size,
            out_size,
            hidden_sizes,
            activation=activation,
            device=self.device
        )

    def inference(
        self,
        obs: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.model(obs)
        scores = scores.cpu().detach().numpy()
        if self.mean_max is not None:
            if self.mean_max == 'mean':
                scores = np.mean(scores, axis=1, keepdims=True)
        ascending_indexes = np.argsort(
            scores.squeeze())  # default ascending order

        return ascending_indexes, scores

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.model(obs)
        return scores
