import torch

from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Optional, Type
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig, DynamicBatchPipeline
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager

@dataclass 
class GENIEPipelineConfig(DynamicBatchPipelineConfig):
    """Configuration for the GENIEPipeline."""

    _target: Type = field(default_factory=lambda: GENIEPipeline)
    """target class to instantiate"""


class GENIEPipeline(DynamicBatchPipeline):

    config: GENIEPipelineConfig
    datamanager: VanillaDataManager

    def __init__(
        self,
        config: DynamicBatchPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }

        means_size = None
        for key, value in state.items():
            if key == "_model.field.mlp_base.encoder.gauss_params.means":
                means_size = value.shape[0]
                break
        
        self.model.field.mlp_base.encoder.reinitialize_params(means_size)

        self.model.update_to_step(step)
        self.load_state_dict(state)
        means = self.model.field.mlp_base.encoder.means
        scales = torch.sqrt(torch.exp(self.model.field.mlp_base.encoder.log_covs))
        quats = self.model.field.mlp_base.encoder.quats
        self.model.field.mlp_base.encoder.knn.fit(means, scales, quats)

    