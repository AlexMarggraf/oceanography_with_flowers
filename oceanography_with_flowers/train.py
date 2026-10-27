import argparse
import sys
import logging
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR
)
from omegaconf import OmegaConf
from hydra.utils import instantiate

from oceanography_with_flowers.model.fno import Pactcs30FNO
from oceanography_with_flowers.data.dataset import Pactcs30Dataset
from oceanography_with_flowers.trainer import Pactcs30Trainer

class RealTimeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream=stream or sys.stdout)

    def emit(self, record):
        super().emit(record)
        self.flush()

def main(model_path, data_path):
    """
    Build model from config file, load metadata from config file, and start training loop.
    Fully modular, config file dictates hyper parameters and what model will be used in training.

    Args:
        model_path (str): Path to model config file
        data_path (str): Path to data config file
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = OmegaConf.load(model_path)
    data_cfg = OmegaConf.load(data_path)
    logging.info(f"model config:\n{OmegaConf.to_yaml(model_cfg)}")
    logging.info(f"data config:\n{OmegaConf.to_yaml(data_cfg)}")

    path = data_cfg.path
    data_vars = data_cfg.data_vars
    pred_vars = data_cfg.pred_vars
    split = data_cfg.split
    weight_path = data_cfg.weight_path
    wandb_path = data_cfg.wandb_path
    delta = data_cfg.delta
    rollout = data_cfg.rollout

    batch_size = model_cfg.batch_size
    epochs = model_cfg.epochs
    rollout_epochs = max(1, int(model_cfg.rollout_fraction * epochs))
    project_name = model_cfg.project_name

    model = instantiate(model_cfg.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"model architecure:\n{model}")
    logging.info(f"total parameters: {total_params:,}")

    loss_fn = instantiate(model_cfg.loss_fn)
    optimizer = instantiate(model_cfg.optimizer, params=model.parameters())
    rollout_optimizer = instantiate(model_cfg.rollout_optimizer, params=model.parameters())

    timespans = []
    for i in range(len(split) - 1):
        timespans.append([year for year in range(split[i], split[i+1])])

    dls = []
    for i, ts in enumerate(timespans):
        logging.info(f"timespan for {i+1}. training dataloader: {ts}")
        dls.append(
            DataLoader(
                Pactcs30Dataset(
                    path, 
                    data_vars, 
                    delta=delta, 
                    timespan=ts
                ), 
                batch_size=batch_size, 
                shuffle=True
            )
        )
    
    rollout_dls = []
    for i, ts in enumerate(timespans):
        logging.info(f"timespan for {i+1}. rollout dataloader: {ts}")
        rollout_dls.append(
            DataLoader(
                Pactcs30Dataset(
                    path, 
                    data_vars, 
                    delta=delta,
                    rollout=rollout,
                    timespan=ts
                ), 
                batch_size=batch_size, 
                shuffle=True
            )
        )

    total_steps = len(dls[0]) * epochs
    rollout_steps = len(rollout_dls[0]) * rollout_epochs
    warmup_steps = int(total_steps * 0.1)
    decay_steps = total_steps - warmup_steps
    logging.info(f"scheduler steps: {warmup_steps} / {decay_steps}")
    logging.info(f"rollout steps: {rollout_steps}")

    warmup_scheduler = instantiate(
        model_cfg.warmup_scheduler,
        optimizer=optimizer,
        total_iters=warmup_steps
    )
    
    decay_scheduler = instantiate(
        model_cfg.decay_scheduler,
        optimizer=optimizer,
        T_max=decay_steps
    )

    scheduler = instantiate(
        model_cfg.scheduler,
        optimizer=optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps]
    )
    
    rollout_scheduler = instantiate(
        model_cfg.rollout_decay_scheduler,
        optimizer=rollout_optimizer,
        T_max=rollout_steps
    )

    config = OmegaConf.to_container(model_cfg, resolve=True)
        
    trainer = Pactcs30Trainer(
        model,
        dls,
        rollout_dls,
        loss_fn,
        optimizer,
        rollout_optimizer,
        scheduler,
        rollout_scheduler,
        epochs,
        rollout_epochs,
        device, 
        pred_vars, 
        rollout
    )

    trainer.train(
	project_name,
	config,
	weight_path,
	wandb_path
    )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            RealTimeStreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Train and validate models on pactcs30 data")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model config file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to data config file",
    )

    args = parser.parse_args()
    main(args.model_path, args.data_path)
