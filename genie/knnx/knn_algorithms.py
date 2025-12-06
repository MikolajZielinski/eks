from __future__ import annotations

import torch
import faiss
import numpy as np

from dataclasses import dataclass, field
from nerfstudio.configs.base_config import InstantiateConfig
from typing import Type

from genie.knnx import knnx


@dataclass
class BaseKNNConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: BaseKNN)
    """Base class for KNN configuration."""
    n_neighbours: int = 16
    """Number of nearest neighbours to consider."""
    device: str = 'cuda'
    """Device to run the KNN algorithm on."""


class BaseKNN:
    """Base class for KNN algorithms."""

    def __init__(self, config: BaseKNNConfig):
        super().__init__()
        self.config: BaseKNNConfig = config

    def fit(self, points: torch.Tensor):
        """Fit the KNN model to the given points."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_nearest_neighbours(self, query: torch.Tensor):
        """Get indices of nearest gaussians."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    

@dataclass
class OptixKNNConfig(BaseKNNConfig):

    _target: Type = field(default_factory=lambda: OptixKNN)
    """Configuration for OptiX KNN algorithm."""
    chi_squared_radius: float = 2.0
    """Chi-squared radius for KNN search."""

class OptixKNN(BaseKNN):
    """KNN algorithm using OptiX."""

    def __init__(self, config: OptixKNNConfig):
        super().__init__(config)

        self.knn = knnx.CPyOptiXKNN(config.chi_squared_radius)

    def fit(self, means: torch.Tensor, scales: torch.Tensor, quaternions: torch.Tensor):
        """
        Fit the KNN model to the given points.

        Parameters:
        - means: (M, D) torch tensor (on CUDA)
        - scales: (M, ...) torch tensor (on CUDA)
        - quaternions: (M, ...) torch tensor (on CUDA)
        """
        assert means.is_cuda, "Means must be on CUDA"

        self.knn.Fit(means, scales, quaternions)

    def get_nearest_neighbours(self, query: torch.Tensor) -> torch.Tensor:
        """
        Efficient KNN using OptiX.

        Parameters:
        - query: (N, D) torch tensor (on CUDA)
        - points: (M, D) torch tensor (on CUDA)

        Returns:
        - nearest_indices: (N, n_neighbors) torch tensor
        """
        # if query.shape[1] == 3:
        #     pad = torch.full((query.shape[0], 1), 0.0, device=query.device, dtype=query.dtype)
        #     self.pad_query = torch.cat([query, pad], dim=1)
        # else:
        #     self.pad_query = query

        # distances = torch.empty((self.config.n_neighbours, self.pad_query.shape[0]), dtype=torch.float32, device='cuda')
        # indices = torch.empty((self.config.n_neighbours, self.pad_query.shape[0]), dtype=torch.int32, device='cuda')

        indices, distances_squared = self.knn.KNeighbors(query, self.config.n_neighbours)
        distances = torch.sqrt(distances_squared)

        distances = distances.T
        indices = indices.T

        return indices, distances
