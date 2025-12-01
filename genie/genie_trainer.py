import csv
import time
import torch
import open3d as o3d
import numpy as np
from dataclasses import dataclass, field
from typing import Type
from pathlib import Path

from nerfstudio.engine.trainer import TrainerConfig, Trainer
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.engine.callbacks import TrainingCallbackLocation
from nerfstudio.utils import writer
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.decorators import check_main_thread


@dataclass
class GENIETrainerConfig(TrainerConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: GENIETrainer)
    """target class to instantiate"""
    densification_start_step: int = 1000
    """Step at which to start densifying points"""
    denisification_interval: int = 200
    """Interval at which to densify points"""
    densification_stop_step: int = 8000
    """Step at which to stop densifying points"""
    pruning_start_step: int = 1000
    """Step at which to start pruning the MLP encoder"""
    pruning_interval: int = 1000
    """Interval at which to prune the MLP encoder"""
    pruning_stop_step: int = 14000
    """Step at which to stop pruning the MLP encoder"""
    unfreeze_means_step: int = 500
    """Step at which to unfreeze the means of the MLP encoder"""
    freeze_means_step: int = 15000
    """Step at which to freeze the means of the MLP encoder"""


class GENIETrainer(Trainer):
    """Trainer for GENIE"""

    config: GENIETrainerConfig

    def __init__(self, config: GENIETrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        super().__init__(config=config, local_rank=local_rank, world_size=world_size)
        self.total_time = 0.0

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"
        if hasattr(self.pipeline.datamanager, "train_dataparser_outputs"):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )

        # Create metrics.csv file and write header
        time_path = self.base_dir / "metrics.csv"
        with time_path.open("w", newline="") as f:
            pass

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations
            step = 0
            self.stop_training = False
            for step in range(self._start_step, self._start_step + num_iterations):
                self.step = step
                
                if step >= self.config.densification_start_step and step % self.config.denisification_interval == 0 and step <= self.config.densification_stop_step:
                    self.pipeline.model.densify_points(self.optimizers.optimizers)

                if step >= self.config.pruning_start_step and step % self.config.pruning_interval == 0 and step <= self.config.pruning_stop_step:
                    self.pipeline.model.field.mlp_base.encoder.prune(optimizers=self.optimizers.optimizers)

                if step == self.config.unfreeze_means_step:
                    self.pipeline.model.field.mlp_base.encoder.unfreeze_means()

                if step == self.config.freeze_means_step:
                    self.pipeline.model.field.mlp_base.encoder.freeze_means()

                if self.stop_training:
                    break
                while self.training_state == "paused":
                    if self.stop_training:
                        self._after_train()
                        return
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass and record duration to time.csv in the output dir
                        start_time = time.perf_counter()
                        loss, loss_dict, metrics_dict = self.train_iteration(step)
                        iter_time = time.perf_counter() - start_time
                        self.total_time += iter_time

                        if step % 100 == 0 or step == self.config.max_num_iterations - 1:
                            time_path = self.base_dir / "metrics.csv"
                            with time_path.open("a", newline="") as f:
                                writer_csv = csv.writer(f)
                                writer_csv.writerow([step, float(self.total_time), metrics_dict["psnr"].item()])

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    with self.train_lock:
                        self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

        # save checkpoint at the end of training, and write out any remaining events
        self._after_train()

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers, and encoder means as .ply

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # Save encoder means as .ply
        means = self.pipeline.model.field.mlp_base.encoder.means.detach().cpu().numpy()
        ply_path = self.checkpoint_dir.parent / f"step-{step:09d}_means.ply"
        self._save_means_as_ply(means, ply_path)
        # Possibly delete old .ply files from parent dir and old .ckpt files from checkpoint dir
        if self.config.save_only_latest_checkpoint:
            # Remove old .ply files from parent dir
            for f in self.checkpoint_dir.parent.glob("*_means.ply"):
                if f != ply_path and f.is_file():
                    f.unlink()
            # Remove old .ckpt files from checkpoint dir
            for f in self.checkpoint_dir.glob("*.ckpt"):
                if f != ckpt_path and f.is_file():
                    f.unlink()

    def _save_means_as_ply(self, means: np.ndarray, ply_path: Path):
        """Save means as a .ply point cloud file using open3d."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(means)
        o3d.io.write_point_cloud(str(ply_path), pcd)