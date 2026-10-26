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
    use_sin_enc = data_cfg.use_sin_enc

    batch_size = model_cfg.batch_size
    epochs = model_cfg.epochs
    project_name = model_cfg.project_name

    #model = instantiate(model_cfg.model, use_film=(not use_sin_enc)).to(device)
    model = instantiate(model_cfg.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"model architecure:\n{model}")
    logging.info(f"total parameters: {total_params:,}")

    loss_fn = instantiate(model_cfg.loss_fn)
    optimizer = instantiate(model_cfg.optimizer, params=model.parameters())

    timespans = []
    for i in range(len(split) - 1):
        timespans.append([year for year in range(split[i], split[i+1])])

    dls = []
    for i, ts in enumerate(timespans):
        logging.info(f"timespan for {i+1}. dataloader: {ts}")
        dls.append(
            DataLoader(Pactcs30Dataset(path, data_vars, delta=delta, timespan=ts, use_sin_enc=use_sin_enc), batch_size=batch_size, shuffle=True)
        )

    total_steps = len(dls[0]) * epochs
    warmup_steps = int(total_steps * 0.1)
    decay_steps = total_steps - warmup_steps
    logging.info(f"scheduler steps: {warmup_steps} / {decay_steps}")

    warmup_scheduler = instantiate(
        model_cfg.warmup_scheduler,
        optimizer=optimizer,
        total_iters=warmup_steps)
    
    decay_scheduler = instantiate(
        model_cfg.decay_scheduler,
        optimizer=optimizer,
        T_max=decay_steps)

    scheduler = instantiate(
        model_cfg.scheduler,
        optimizer=optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps]
        )

    config = OmegaConf.to_container(model_cfg, resolve=True)
        
    trainer = Pactcs30Trainer(model, dls, loss_fn, optimizer, scheduler, epochs, device, pred_vars, use_sin_enc=use_sin_enc)
    trainer.train(project_name, config, weight_path, wandb_path)

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
