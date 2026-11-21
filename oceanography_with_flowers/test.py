import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from hydra.utils import instantiate
from pathlib import Path

from oceanography_with_flowers.model import (
    Pactcs30FNO,
    Flower,
    CNUnet,
    ScOT,
    GAOT
)

from oceanography_with_flowers.data.dataset import Pactcs30Dataset
from oceanography_with_flowers.trainer import Pactcs30Trainer
from oceanography_with_flowers.gaot_config import ModelConfig

class RealTimeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream=stream or sys.stdout)

    def emit(self, record):
        super().emit(record)
        self.flush()

class NRMSE(nn.Module):
    def __init__(
        self,
        reduce_dims=-1,
        eps=1.0e-6
    ):
        super().__init__()

        self.reduce_dims = reduce_dims
        self.eps = eps
        self.name = "nrmse"

    def get_name(self):
        return self.name

    def forward(self, pred, label):
        y_max = torch.amax(label, self.reduce_dims)
        y_min = torch.amin(label, self.reduce_dims)

        mse = torch.mean((pred - label) ** 2, dim=self.reduce_dims)
        rmse = torch.sqrt(mse)
        nrmse = rmse / (y_max - y_min + self.eps)

        return torch.mean(nrmse)

class VRMSE(nn.Module):
    def __init__(
        self,
        reduce_dims=-1,
        eps=1.0e-6
    ):
        super().__init__()

        self.reduce_dims = reduce_dims
        self.eps = eps
        self.name = "vrmse"

    def get_name(self):
        return self.name

    def forward(self, pred, label):
        var = torch.var(label, dim=self.reduce_dims, unbiased=False)

        mse = torch.mean((pred - label) ** 2, dim=self.reduce_dims)
        vrmse = torch.sqrt(mse / (var + self.eps))

        return torch.mean(vrmse)

class Tester:
    def __init__(
        self,
        model,
        dl,
        loss_fns,
        device,
        rollout,
        sample_days,
        img_path,
        gaot=False
    ):
        self.model = model
        self.dl = dl
        self.mask = dl.dataset.get_mask().squeeze()
        self.loss_fns = loss_fns
        self.device = device
        self.rollout = rollout
        self.sample_days = sample_days
        self.img_path = img_path
        self.gaot = gaot
        self.first_pass = True

        if gaot:
            self.latent_grid = dl.dataset.get_latent_grid().to(device, non_blocking=True)
            self.real_grid = dl.dataset.get_real_grid().to(device, non_blocking=True)

    def test_model(self):
        samples = len(self.dl)
        total_loss = dict()
        step_loss = dict()
        rollout_loss = dict()

        for loss in self.loss_fns:
            name = loss.get_name()

            total_loss[name] = []
            step_loss[name] = []
            rollout_loss[name] = []

        sample_dict = dict()

        for day in self.sample_days:
            # batch where the idx is in and the actual idx relative to the batch
            sample_dict[math.floor(day / self.dl.batch_size)] = day % self.dl.batch_size

        one_step_pred = []
        one_step_label = []
        rollout_pred = []
        rollout_label = []

        self.model.eval()

        with torch.no_grad():
            for i, (curr_state, label, metadata) in enumerate(self.dl):
                curr_state = curr_state.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                metadata = metadata.to(self.device, non_blocking=True)

                # one-step prediction
                if self.gaot:
                    curr_state = torch.cat([curr_state, metadata[:, 0]], dim=-1)
                    curr_state = self.model(self.latent_grid, self.real_grid, curr_state)

                    # mask actual values since loss can't be masked
                    # also transpose such that shape is [B, C, N]
                    pred_valid = curr_state[:, self.mask, :].transpose(1, 2)
                    label_valid = label[:, 0][:, self.mask, :].transpose(1, 2)
                else:
                    curr_state = torch.cat([curr_state, metadata[:, 0]], dim=1)
                    curr_state = self.model(curr_state)
                    
                    # mask actual values since loss can't be masked
                    # shape is [B, C, N] afterwards
                    pred_valid = curr_state[:, :, self.mask]
                    label_valid = label[:, 0][:, :, self.mask]
                    
                    self.first_pass = False

                for loss_fn in self.loss_fns:
                    loss = loss_fn(pred_valid, label_valid)

                    name = loss_fn.get_name()
                    total_loss[name].append(loss.item())
                    step_loss[name].append(loss.item())

                if i in sample_dict.keys():
                    one_step_pred.append(curr_state[sample_dict[i]].clone().detach().cpu())
                    one_step_label.append(label[sample_dict[i], 0].clone().detach().cpu())

                for j in range(1, self.dl.dataset.rollout):
                    if self.gaot:
                        curr_state = torch.cat([curr_state, metadata[:, j]], dim=-1)
                        curr_state = self.model(self.latent_grid, self.real_grid, curr_state)
                    else:
                        curr_state = torch.cat([curr_state, metadata[:, j]], dim=1)
                        curr_state = self.model(curr_state)

                if self.gaot:
                    # mask actual values since loss can't be masked
                    # also transpose such that shape is [B, C, N]
                    pred_valid = curr_state[:, self.mask, :].transpose(1, 2)
                    label_valid = label[:, 0][:, self.mask, :].transpose(1, 2)
                else:
                    # mask actual values since loss can't be masked
                    # shape is [B, C, N] afterwards
                    pred_valid = curr_state[:, :, self.mask]
                    label_valid = label[:, 0][:, :, self.mask]

                for loss_fn in self.loss_fns:
                    loss = loss_fn(pred_valid, label_valid)
                    
                    name = loss_fn.get_name()
                    total_loss[name][-1] += loss.item()
                    rollout_loss[name].append(loss.item())

                if i in sample_dict.keys():
                    rollout_pred.append(curr_state[sample_dict[i]].clone().detach().cpu())
                    rollout_label.append(label[sample_dict[i], 1].clone().detach().cpu())

            step = [self.dl.dataset.tensor_to_dataset(t) for t in one_step_pred]
            step_label = [self.dl.dataset.tensor_to_dataset(t) for t in one_step_label]

            rollout = [self.dl.dataset.tensor_to_dataset(t) for t in rollout_pred]
            rollout_label = [self.dl.dataset.tensor_to_dataset(t) for t in rollout_label]

            step_scales = []
            rollout_scales = []

            for (date, label) in zip(self.sample_days, step_label):
                step_scales.append(
                    self.dl.dataset.plot_dataset(
                        label,
                        f"{self.img_path}/day_{date}/step_label",
                        return_scales=True
                    )
                )

            for (date, step, scales) in zip(self.sample_days, step, step_scales):
                self.dl.dataset.plot_dataset(
                    step,
                    f"{self.img_path}/day_{date}/step_pred",
                    scales=scales
                )

            for (date, label) in zip(self.sample_days, rollout_label):
                rollout_scales.append(
                    self.dl.dataset.plot_dataset(
                        label, 
                        f"{self.img_path}/day_{date}/rollout_label", 
                        return_scales=True
                    )
                )

            for (date, rollout, scales) in zip(self.sample_days, rollout, rollout_scales):
                self.dl.dataset.plot_dataset(
                    rollout, 
                    f"{self.img_path}/day_{date}/rollout_pred", 
                    scales=scales
                )

        return step_loss, rollout_loss, total_loss

def plot_loss(loss_data, loss_titles, img_path):
    img_path = Path(img_path)
    img_path.mkdir(parents=True, exist_ok=True)

    for (title, data) in zip(loss_titles, loss_data):
        plt.figure(figsize=(10, 5))

        for name, values in data.items():
            values = np.array(values)
            steps = np.arange(1, len(values) + 1)

            plt.plot(steps, values, label=f"{name}")
        
        plt.xlabel("Batch")
        plt.ylabel("Loss Value")
        plt.title(f"Loss Values over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{img_path}/{title}", dpi=300, bbox_inches="tight")
        plt.close()

def main(model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = OmegaConf.load(model_path)
    data_cfg = OmegaConf.load(data_path)
    logging.info(f"device: {device}")
    logging.info(f"model config:\n{OmegaConf.to_yaml(model_cfg)}")
    logging.info(f"data config:\n{OmegaConf.to_yaml(data_cfg)}")

    path = data_cfg.path
    weight_path = data_cfg.weight_path
    test_path = data_cfg.test_path
    run_id = data_cfg.run_id
    meta_vars = data_cfg.meta_vars
    pred_vars = data_cfg.pred_vars
    split = data_cfg.split
    sample_days = data_cfg.sample_days
    delta = data_cfg.delta
    rollout = data_cfg.rollout
    width = data_cfg.width
    height = data_cfg.height

    project_name = model_cfg.project_name
    model_name = model_cfg.model_name
    batch_size = model_cfg.batch_size

    img_path = f"{test_path}/img/{model_name}/{run_id}"
    weight_path = f"{weight_path}{run_id}/best_loss.pt"
    loss_fns = [instantiate(data_cfg.nrmse), instantiate(data_cfg.vrmse)]

    pad = None
    grid = True
    gaot = False
    latent_dim = None

    match model_name:
        case "flower" | "scot":
            pad_x = model_cfg.pad_x
            pad_y = model_cfg.pad_y
            model = instantiate(model_cfg.model, spatial_resolution=(width + 2*pad_x, height + 2*pad_y)).to(device)
            pad = (pad_x, pad_y)
        case "cnunet" :
            pad_x = model_cfg.pad_x
            pad_y = model_cfg.pad_y
            model = instantiate(model_cfg.model).to(device)
            pad = (pad_x, pad_y)
        case "gaot":
            # adhereing to build pipeline of original GAOT code
            default = OmegaConf.structured(ModelConfig)
            merged = OmegaConf.merge(default, model_cfg.model)
            config = OmegaConf.to_object(merged)

            model = GAOT(config=config).to(device)
            grid = False
            gaot = True
            latent_dim = (model_cfg.model.latent_tokens_size[0], model_cfg.model.latent_tokens_size[1])
        case _:
            model = instantiate(model_cfg.model).to(device)

    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    total_params = sum(p.numel() for p in model.parameters())
    # Uncomment if you want to see the model architecture, output can be very long
    # logging.info(f"model architecure:\n{model}")
    logging.info(f"total parameters: {total_params:,}")

    timespan = [year for year in range(split[0], split[1])]
    logging.info(f"timespan for dataloader: {timespan}")

    dl = DataLoader(
        Pactcs30Dataset(
            path,
            pred_vars,
            meta_vars,
            pad,
            delta=delta,
            rollout=rollout,
            timespan=timespan,
            grid=grid,
            latent_dim=latent_dim
        ),
        batch_size=batch_size,
        shuffle=False
    )

    tester = Tester(
        model,
        dl,
        loss_fns,
        device,
        rollout,
        sample_days,
        img_path,
        gaot=gaot
    )

    step_loss, rollout_loss, total_loss = tester.test_model()

    loss_data = [step_loss, rollout_loss, total_loss]
    titles = ["step_loss", "rollout_loss", "total_loss"]

    plot_loss(loss_data, titles, img_path)
    for (title, data) in zip(titles, loss_data):
        print(f"{title}:")

        for (name, values) in data.items():
            print(f"{name}: {sum(values) / len(values)}")
        
        print("")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            RealTimeStreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Test models on pactcs30 data")
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