"""
Encoding functions
"""

from abc import abstractmethod
from typing import Optional, Callable, Union, Dict, List

import numpy as np
import torch
from torch import Tensor, nn

from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion

from genie.knn.knn_algorithms import BaseKNN


class SplashEncoding(nn.Module):
    def __init__(
        self,
        n_gausses: int = 10000,
        n_features_per_gauss: int = 32,
        knn_algorithm: Optional[BaseKNN] = None,
        means: Optional[Tensor] = None,
        densify: bool = True,
        prune: bool = True,
        unfreeze_means: bool = True,
        spatial_distortion: Optional[SpatialDistortion] = None,
        device: str = 'cuda',
        empty_as_hash: bool = False,
    ):
        """
        """
        super().__init__()
        assert knn_algorithm is not None, "KNN algorithm must be provided"
        
        self.n_features_per_gauss = n_features_per_gauss
        self.densify_gausses = densify
        self.prune_gausses = prune
        self.unfreeze_gausses = unfreeze_means
        self.device = device
        self.empty_as_hash = empty_as_hash
        
        if means is not None and isinstance(means, np.ndarray):
            means = torch.tensor(means, dtype=torch.float32, device=self.device)
        elif means is not None and isinstance(means, Tensor):
            means = means.to(device=self.device)
        else:
            means = self.init_mean(n_gausses)

        if spatial_distortion is not None:
            unifrom_means = self.init_mean_unifrom(1000000)
            means = torch.cat([means, unifrom_means], dim=0)

        self.hash_encoding = HashEncoding(max_res=8192, log2_hashmap_size=21)
        self.means_hash = HashEncoding(max_res=8192, log2_hashmap_size=21)

        # Initialize Gaussians
        self.total_gaus = means.shape[0]
        means = nn.Parameter(means, requires_grad=False)
        self.register_buffer("feats", self.means_hash(means))
        log_covs = nn.Parameter(torch.log(torch.ones(self.total_gaus, 3, device=self.device) * 0.0001))
        quats = nn.Parameter(torch.zeros(self.total_gaus, 4, device=self.device))
        quats.data[:, 0] = 1.0
        self.confidence = torch.ones_like(means[:, 0], device=self.device, requires_grad=False)
        self.gauss_params = torch.nn.ParameterDict({
            "means": means,
            "log_covs": log_covs,
            "quats": quats
        })
        
        # Initialize KNN algorithm
        self.knn = knn_algorithm

    def init_mean(self, N):
        print(f'Total number of gauss: {N}')
        pts = np.random.randn(N, 3)
        r = np.sqrt(np.random.rand(N, 1))
        pts = pts / np.linalg.norm(pts, axis=1)[:, None] * r
        pts = pts * 0.5 + 0.5 # [0.25 ... 0.75]
        
        return torch.tensor(pts, dtype=torch.float32, device=self.device)
    
    def init_mean_unifrom(self, N, thickness: float = 0.05, box_min: float = 0.1, box_max: float = 0.9):
        """Initialize means uniformly on the outer shell (thickness) of a box [box_min,box_min,box_min]-[box_max,box_max,box_max], excluding top and bottom faces."""
        print(f'Total number of gauss: {N}')
        pts = []
        batch = int(N * 1.5)  # oversample to ensure enough points
        box_size = box_max - box_min
        while sum(len(p) for p in pts) < N:
            samples = np.random.rand(batch, 3)
            samples = samples * box_size + box_min  # scale to [box_min, box_max]
            # Only select points near the sides (x or y near boundary), not z
            mask = ((samples[:, 0] <= box_min + thickness) | (samples[:, 0] >= box_max - thickness) |
                    (samples[:, 1] <= box_min + thickness) | (samples[:, 1] >= box_max - thickness))
            shell_pts = samples[mask]
            if len(shell_pts) > 0:
                pts.append(shell_pts)
        pts = np.concatenate(pts, axis=0)
        if pts.shape[0] > N:
            pts = pts[:N]
        return torch.tensor(pts, dtype=torch.float32, device=self.device)
    
    @torch.no_grad()
    def _update_param_with_optimizer(
        self,
        param_fn: Callable[[str, Tensor], Tensor],
        optimizer_fn: Callable[[str, Tensor], Tensor],
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        names: Union[List[str], None] = None,
    ):
        """Update the parameters and the state in the optimizers with defined functions.

        Args:
            param_fn: A function that takes the name of the parameter and the parameter itself,
                and returns the new parameter.
            optimizer_fn: A function that takes the key of the optimizer state and the state value,
                and returns the new state value.
            params: A dictionary of parameters.
            optimizers: A dictionary of optimizers, each corresponding to a parameter.
            names: A list of key names to update. If None, update all. Default: None.
        """
        if names is None:
            # If names is not provided, update all parameters
            names = list(params.keys())

        for name in names:
            param = params[name]
            new_param = param_fn(name, param)
            params[name] = new_param
            if name not in optimizers:
                assert not param.requires_grad, (
                    f"Optimizer for {name} is not found, but the parameter is trainable."
                    f"Got requires_grad={param.requires_grad}"
                )
                continue
            optimizer = optimizers[name]
            for i in range(len(optimizer.param_groups)):
                param_state = optimizer.state[param]
                del optimizer.state[param]
                for key in param_state.keys():
                    if key != "step":
                        v = param_state[key]
                        param_state[key] = optimizer_fn(key, v)
                optimizer.param_groups[i]["params"] = [new_param]
                optimizer.state[new_param] = param_state
    
    def densify(self, new_means: torch.Tensor, optimizers: Dict[str, torch.optim.Optimizer]) -> None:
        """
        Add new means, feats, log_covs, quats, and confidence entries, and refit KNN.
        """

        if self.densify_gausses:
            def param_fn(name: str, p: Tensor) -> Tensor:
                if name == 'means':
                    new_param = nn.Parameter(torch.cat([p, new_means], dim=0), requires_grad=self.means.requires_grad)
                elif name == 'log_covs':
                    new_covs = torch.log(torch.ones(new_means.shape[0], 3, device=new_means.device) * 0.0001)
                    new_param = nn.Parameter(torch.cat([p, new_covs], dim=0), requires_grad=self.log_covs.requires_grad)
                elif name == 'quats':
                    new_quats = torch.zeros(new_means.shape[0], 4, device=new_means.device)
                    new_quats[:, 0] = 1.0
                    new_param = nn.Parameter(torch.cat([p, new_quats], dim=0), requires_grad=self.quats.requires_grad)

                return new_param

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                return torch.cat([v, torch.zeros((len(new_means), *v.shape[1:]), device=self.device)])

            self._update_param_with_optimizer(param_fn, optimizer_fn, self.gauss_params, optimizers)
            
            print(f'Densifying with {new_means.shape[0]} new means')
            new_confidence = torch.ones(new_means.shape[0], device=new_means.device)
            self.confidence = torch.cat([self.confidence, new_confidence], dim=0)
            self.total_gaus = self.means.shape[0]
            self.feats = self.means_hash(self.means)
            print(f'New total number of gauss: {self.means.shape[0]}')

    def prune(self, optimizers: Dict[str, torch.optim.Optimizer], threshold: float=0.1):
        """
        Remove all means, feats, log_covs, quats, and confidence entries with confidence lower than threshold.
        """

        if self.prune_gausses:

            mask = self.confidence >= threshold
            def param_fn(name: str, p: Tensor) -> Tensor:
                return torch.nn.Parameter(p[mask], requires_grad=p.requires_grad)

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                return v[mask]

            self._update_param_with_optimizer(param_fn, optimizer_fn, self.gauss_params, optimizers)

            # Only keep entries where mask is True
            self.confidence = self.confidence[mask]
            self.total_gaus = self.means.shape[0]
            self.feats = self.means_hash(self.means)
            
            # Refit KNN with new means
            print(f"Pruned to {self.means.shape[0]} gaussians.")

    def reinitialize_params(self, n_gausses: int) -> None:
        """
        Reinitialize the means, feats, log_covs, and confidence with new random values, and refit KNN.
        """
        self.gauss_params["means"] = nn.Parameter(self.init_mean(n_gausses), requires_grad=False)
        self.feats = self.means_hash(self.means)
        self.gauss_params["log_covs"] = nn.Parameter(torch.log(torch.ones(n_gausses, 3, device=self.device) * 0.0001), requires_grad=self.log_covs.requires_grad)
        quats = torch.zeros(n_gausses, 4, device=self.device)
        quats[:, 0] = 1.0
        self.gauss_params["quats"] = nn.Parameter(quats, requires_grad=self.quats.requires_grad)
        self.confidence = torch.ones(n_gausses, device=self.device)
        self.total_gaus = n_gausses
        print(f"Reinitialized to {n_gausses} gaussians.")

    def unfreeze_means(self):
        if self.unfreeze_gausses:
            self.gauss_params["means"].requires_grad_(True)

    def freeze_means(self):
        if self.unfreeze_gausses:
            self.gauss_params["means"].requires_grad_(False)

    def get_out_dim(self) -> int:
        return self.n_features_per_gauss
    
    @property
    def means(self) -> Tensor:
        return self.gauss_params["means"]
    
    @property
    def log_covs(self) -> Tensor:
        return self.gauss_params["log_covs"]

    @property
    def quats(self) -> Tensor:
        return self.gauss_params["quats"]
        
    def quat_to_rotmat(self, quats):
        w, x, y, z = quats.unbind(-1)
        
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        
        row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
        row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
        row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
        
        return torch.stack([row0, row1, row2], dim=-2)

    def interpolate(self, coords, nearest_gausses_indicies):

        if self.training:
            self.feats = self.means_hash(self.means)
        nearest_features = self.feats[nearest_gausses_indicies]
        nearest_covs = torch.exp(self.log_covs[nearest_gausses_indicies])
        nearest_quats = self.quats[nearest_gausses_indicies]
        nearest_quats = nearest_quats / nearest_quats.norm(dim=-1, keepdim=True)
        R = self.quat_to_rotmat(nearest_quats)

        diff = coords[:, None, :] - self.means[nearest_gausses_indicies]
        diff_local = torch.matmul(R.transpose(-1, -2), diff.unsqueeze(-1)).squeeze(-1)
        mdist = (diff_local ** 2 / nearest_covs).sum(-1)

        # Normalization constant for diagonal Gaussian
        gau_weights = torch.exp(-0.5 * mdist)
        zeros = torch.zeros_like(gau_weights)
        gau_weights = torch.where(nearest_gausses_indicies != -1, gau_weights, zeros)
        weighted_features = nearest_features * gau_weights.unsqueeze(-1)

        return torch.sum(weighted_features, dim=1)

    def forward(self, coords):
        
        with torch.no_grad():
            sigma_max, _ = torch.sqrt(torch.exp(self.log_covs)).max(dim=-1)
            pad_means = torch.cat([self.means, sigma_max.unsqueeze(-1)], dim=1)
            self.knn.fit(pad_means)
            nearest_gausses_indicies, self.distances = self.knn.get_nearest_neighbours(coords)
            max_idx = self.means.shape[0] - 1
            nearest_gausses_indicies = torch.clamp(nearest_gausses_indicies, min=0, max=max_idx)

        slpash_feats = self.interpolate(coords, nearest_gausses_indicies)

        if self.empty_as_hash:
            hash_feats = self.hash_encoding(coords)
            mask = (nearest_gausses_indicies > 0).all(dim=1)
            feats = torch.where(mask.unsqueeze(-1), slpash_feats, hash_feats)
        else:
            feats = slpash_feats

        if self.training:
            self.confidence -= 0.001
            self.confidence[nearest_gausses_indicies] += 0.01
            self.confidence.clamp_(min=0.0, max=1.0)

        return feats
