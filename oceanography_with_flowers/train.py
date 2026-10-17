import argparse
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

from oceanography_with_flowers.model.fno import Pactcs30Fno
from oceanography_with_flowers.data.dataset import Pactcs30Dataset
from oceanography_with_flowers.trainer import Pactcs30Trainer

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

    path = data_cfg.path
    data_vars = data_cfg.data_vars
    pred_vars = data_cfg.pred_vars
    train_range = data_cfg.train_range
    val_range = data_cfg.val_range
    weight_path = data_cfg.weight_path
    wandb_path = data_cfg.wandb_path

    train_timespan = [i for i in range(train_range[0], train_range[1])]
    val_timespan = [i for i in range(val_range[0], val_range[1])]

    train_dl = DataLoader(Pactcs30Dataset(path, data_vars, timespan=train_timespan), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(Pactcs30Dataset(path, data_vars, timespan=val_timespan, batch_size=batch_size, shuffle=True))

    batch_size = model_cfg.batch_size
    epochs = model_cfg.epochs
    project_name = model_cfg.project_name

    model = instantiate(model_cfg.model).to(device)
    loss_fn = instantiate(model_cfg.loss)
    optimizer = instantiate(model_cfg.optimizer, params=model.parameters())

    total_steps = len(train_dl) * epochs
    warmup_steps = int(total_steps * 0.1)
    decay_steps = total_steps - warmup_steps

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
        
    trainer = Pactcs30Trainer(model, train_dl, val_dl, batch_size, loss_fn, optimizer, scheduler, epochs, device, pred_vars)
    trainer.train(project_name, config, weight_path, wandb_path)

if __name__ == "__main__":
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
