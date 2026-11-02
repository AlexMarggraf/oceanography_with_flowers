import numpy as np
import logging
import torch
import wandb
from pathlib import Path

class Pactcs30Trainer():
    """
    A trainer for the pactcs30 dataset which trains and validates the data.

    Args:
        model (nn.Module): Model used for training
        train_dl (Dataloader): Dataloader of the training data
        val_dl (Dataloader): Dataloader of the validation data
        loss_fn (Callable): Loss function to be used in training and validation
        optimizer: Optimizer used during training
        scheduler: Scheduler used during training
        epochs: Epochs
        device: The device used for training
        pred_vars: The variables that are predicted by the model
    """

    def __init__(
        self, 
        model, 
        dls,
        loss_fn, 
        optimizer, 
        scheduler, 
        epochs, 
        device, 
        pred_vars, 
        rollout
    ):
        self.model = model
        self.dls = dls
        self.mask = dls[0].dataset.get_mask()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.pred_vars = pred_vars
        self.rollout = rollout
        self.first_pass = True

    def train_model(self):
        total_loss = 0
        step_loss = 0
        rollout_loss = 0
        no_batches = len(self.dls[0])
        self.model.train()

        for i, (curr_state, label, metadata) in enumerate(self.dls[0]):
            self.optimizer.zero_grad(set_to_none=True)
            combined_loss = 0
            
            curr_state = curr_state.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            metadata = metadata.to(self.device, non_blocking=True)
            
            # calculate one step loss
            curr_state = torch.cat([curr_state, metadata[:, 0]], dim=1)
            curr_state = self.model(curr_state)

            v_mask = self.mask.expand_as(label[:, 0])
            loss = self.loss_fn(curr_state[v_mask], label[:, 0][v_mask])

            combined_loss += loss.item()
            step_loss += loss.item()

            loss.backward()

            if self.first_pass:
                logging.info(f"rollout state shape: {curr_state.shape}")
                logging.info(f"rollout label shape: {label.shape}")
                logging.info(f"rollout metadata shape: {metadata.shape}")
                self.first_pass = False

            # calculate rollout loss
            with torch.no_grad():
                for j in range(1, self.dls[0].dataset.rollout - 1):
                    curr_state = torch.cat([curr_state, metadata[:, j]], dim=1)
                    curr_state = self.model(curr_state)

            curr_state = torch.cat([curr_state, metadata[:, -1]], dim=1)
            curr_state = self.model(curr_state)

            v_mask = self.mask.expand_as(label[:, 1])
            loss = self.loss_fn(curr_state[v_mask], label[:, 1][v_mask])

            combined_loss += loss.item()
            rollout_loss += loss.item()
            total_loss += combined_loss
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return step_loss / no_batches, rollout_loss / no_batches, total_loss / no_batches
    
    def validate_model(self, best_loss, img_path=None):
        total_loss = 0
        step_loss = 0
        rollout_loss = 0
        no_batches = len(self.dls[0])
        self.model.eval()

        with torch.no_grad():
            for i, (curr_state, label, metadata) in enumerate(self.dls[0]):
                combined_loss = 0
                
                curr_state = curr_state.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                metadata = metadata.to(self.device, non_blocking=True)
                
                # calculate one step loss
                curr_state = torch.cat([curr_state, metadata[:, 0]], dim=1)
                curr_state = self.model(curr_state)

                v_mask = self.mask.expand_as(label[:, 0])
                loss = self.loss_fn(curr_state[v_mask], label[:, 0][v_mask])

                combined_loss += loss.item()
                step_loss += loss.item()

                # calculate rollout loss
                with torch.no_grad():
                    for j in range(1, self.dls[0].dataset.rollout - 1):
                        curr_state = torch.cat([curr_state, metadata[:, j]], dim=1)
                        curr_state = self.model(curr_state)

                curr_state = torch.cat([curr_state, metadata[:, -1]], dim=1)
                curr_state = self.model(curr_state)

                v_mask = self.mask.expand_as(label[:, 1])
                loss = self.loss_fn(curr_state[v_mask], label[:, 1][v_mask])

                combined_loss += loss.item()
                rollout_loss += loss.item()
                total_loss += combined_loss

            step_loss /= no_batches
            rollout_loss /= no_batches
            total_loss /= no_batches

            if total_loss < best_loss:
                curr_state, label, metadata = next(iter(self.dls[1]))
                curr_state = curr_state.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                metadata = metadata.to(self.device,non_blocking=True)

                step = torch.cat([curr_state, metadata[:, 0]], dim=1)
                step = self.model(step)
                rollout = step.clone()

                for i in range(1, self.dls[1].dataset.rollout):
                    rollout = torch.cat([rollout, metadata[:, i]], dim=1)
                    rollout = self.model(rollout)

                step = self.dls[0].dataset.tensor_to_dataset(step[0].detach().cpu(), self.pred_vars)
                step_label = self.dls[0].dataset.tensor_to_dataset(label[0, 0].detach().cpu(), self.pred_vars)

                rollout = self.dls[0].dataset.tensor_to_dataset(rollout[0].detach().cpu(), self.pred_vars)
                rollout_label = self.dls[0].dataset.tensor_to_dataset(label[0, 1].detach().cpu(), self.pred_vars)

                step_scales = self.dls[0].dataset.plot_dataset(
                    step_label,
                    f"{img_path}/step_label",
                    self.pred_vars,
                    return_scales=True
                )

                self.dls[0].dataset.plot_dataset(
                    step,
                    f"{img_path}/step_pred",
                    self.pred_vars,
                    scales=step_scales
                )

                rollout_scales = self.dls[0].dataset.plot_dataset(
                    rollout_label, 
                    f"{img_path}/rollout_label", 
                    self.pred_vars, 
                    return_scales=True
                )

                self.dls[0].dataset.plot_dataset(
                    rollout, 
                    f"{img_path}/rollout_pred", 
                    self.pred_vars, 
                    scales=rollout_scales
                )

        return step_loss, rollout_loss, total_loss

    def train(self, project_name, config, weight_path, wandb_path):
        """
        Train and validate the model, as well as log loss in weights and biases.

        Args:
            project_name (str): Name of the weights and biases project
            config (dict()}: Config used for weights and biases
            weight_path (str): Path to folder where weights are stored
            wandb_path (str): Path to folder where weights and biases data is stored
        """

        run = wandb.init(project=project_name, config=config, dir=wandb_path, reinit=True)
        weight_path = Path(f"{weight_path}/run_{run.id}")
        weight_path.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")
        
        for epoch in range(self.epochs):
            step_loss, rollout_loss, total_loss = self.train_model() 
            step_val, rollout_val, total_val = self.validate_model(best_loss, img_path=f"{wandb_path}/img/{run.id}")

            if epoch == 0:
                max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                logging.info(f"max GPU memory allocated: {max_mem_gb:.2f} GB")
            
            wandb.log({
                "step_loss": step_loss,
                "rollout_loss": rollout_loss,
                "total_loss": total_loss,
                "step_val": step_val,
                "rollout_val": rollout_val,
                "total_val": total_val
            })

            if total_val < best_loss:
                best_loss = total_val
                torch.save(self.model.state_dict(), weight_path / "best_loss.pt")

        wandb.finish()
