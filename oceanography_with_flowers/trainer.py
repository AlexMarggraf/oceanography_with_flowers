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
    ):
        self.model = model
        self.dls = dls
        self.rollout_dls = rollout_dls
        self.mask = dls[0].dataset.get_mask()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.rollout_optimizer = rollout_optimizer
        self.scheduler = scheduler
        self.rollout_scheduler = rollout_scheduler
        self.epochs = epochs
        self.rollout_epochs = rollout_epochs
        self.device = device
        self.pred_vars = pred_vars
        self.rollout = rollout
        self.first_pass = True
    
def train_model(self):
    """
    One iteration of the training loop.

    Returns:
        float: Average 1-step training loss of all batches
    """
    self.model.train()
    train_loss = 0
    no_batches = len(self.dls[0])
    
    for data in self.dls[0]:
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0
        
        feature, label = data
        feature = feature.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)
        
        pred = self.model(feature) 

        if self.first_pass:
            logging.info(f"feature shape: {feature.shape}")
            logging.info(f"label shape: {label.shape}")
            logging.info(f"prediction shape: {pred.shape}")
            self.first_pass = False

        v_mask = self.mask.expand_as(pred)

        # Loss is masked to ignore regions not in ocean
        loss = self.loss_fn(pred[v_mask], label[v_mask])
        train_loss += loss.item()

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    return train_loss / no_batches
    
    def validate_model(self, best_loss, img_path=None):
        """
        One iteration of the validation loop.

        Returns:
            int: Average validation loss of all batches
        """
                
        self.model.eval()
        val_loss = 0
        no_batches = len(self.dls[1])
    
        with torch.no_grad():
            for data in self.dls[1]:
                feature, label = data
                feature = feature.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                pred = self.model(feature) 

                v_mask = self.mask.expand_as(pred)

                # loss is masked to ignore regions not in ocean
                loss = self.loss_fn(pred[v_mask], label[v_mask])
                val_loss += loss.item()

            val_loss /= no_batches
        
            if val_loss < best_loss:
                feature, label = next(iter(self.dls[1]))
                feature = feature.to(self.device, non_blocking=True)
                pred = self.model(feature)

                pred = self.dls[0].dataset.tensor_to_dataset(pred[0].detach().cpu(), self.pred_vars)
                label = self.dls[0].dataset.tensor_to_dataset(label[0].detach().cpu(), self.pred_vars)

                scales = self.dls[0].dataset.plot_dataset(
                    label, 
                    f"{img_path}/label", 
                    self.pred_vars, 
                    return_scales=True
                )
                self.dls[0].dataset.plot_dataset(
                    pred, 
                    f"{img_path}/prediction", 
                    self.pred_vars, 
                    scales=scales
                )

        return val_loss

    def train_rollout(self):
        self.model.train()
        rollout_loss = 0
        no_batches = len(self.rollout_dls[0])

        for data in self.rollout_dls[0]:
            total_loss = 0
            self.rollout_optimizer.zero_grad(set_to_none=True)

            curr_state, labels, metadata = data
            curr_state = curr_state.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            metadata = metadata.to(device,non_blocking=True)

            if self.first_pass:
                logging.info(f"rollout start shape: {curr_state.shape}")
                logging.info(f"rollout label shape: {labels.shape}")
                logging.info(f"rollout metadata shape: {metadata.shape}")

            for i in range(self.rollout):
                curr_state = torch.cat(
                    [
                        curr_state,
                        metadata[:, i]
                    ], dim=1
                )

                next_state = self.model(curr_state)
                v_mask = self.mask.expand_as(labels[:, i])
                total_loss += self.loss_fn(next_state[v_mask], labels[:, i][v_mask])

                if i < self.rollout - 1:
                    curr_state = next_state.detach()

            if self.first_pass:
                logging.info(f"rollout prediction shape: {curr_state.shape}")
                self.first_pass = False

            rollout_loss += total_loss.item()
            total_loss.backward()
            self.rollout_optimizer.step()
            self.rollout_scheduler.step()
        
        return rollout_loss / no_batches
    
    def validate_rollout(self, best_loss, img_path=None):
        self.model.eval()
        val_loss = 0
        no_batches = len(self.rollout_dls[1])

        with torch.no_grad():
            for data in self.rollout_dls[1]:
                curr_state, labels, metadata = data
                curr_state = curr_state.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                metadata = metadata.to(device,non_blocking=True)

                for i in range(self.rollout):
                    curr_state = torch.cat(
                        [
                            curr_state,
                            metadata[:, i]
                        ], dim=1
                    )

                    curr_state = self.model(curr_state)
                    v_mask = self.mask.expand_as(labels[:, i])
                    total_loss += self.loss_fn(curr_state[v_mask], labels[:, i][v_mask])
                
                val_loss += total_loss.item()
        
            val_loss /= no_batches

            if val_loss < best_loss:
                curr_state, labels, metadata = next(iter(self.rollout_dls[1]))
                curr_state = curr_state.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                metadata = metadata.to(device,non_blocking=True)

                for i in range(self.rollout):
                    curr_state = torch.cat(
                        [
                            curr_state,
                            metadata[:, i]
                        ], dim=1
                    )

                    curr_state = self.model(curr_state)

                curr_state = self.rollout_dls[0].dataset.tensor_to_dataset(curr_state[0].detach().cpu(), self.pred_vars)
                label = self.rollout_dls[0].dataset.tensor_to_dataset(labels[0, -1].detach().cpu(), self.pred_vars)

                scales = self.rollout_dls[0].dataset.plot_dataset(
                    label, 
                    f"{img_path}/rollout_label", 
                    self.pred_vars, 
                    return_scales=True
                )
                self.rollout_dls[0].dataset.plot_dataset(
                    curr_state, 
                    f"{img_path}/rollout_prediction", 
                    self.pred_vars, 
                    scales=scales
                )

        return val_loss

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
        """ weight_path = Path(f"{weight_path}/run_{run.id}")
        weight_path.mkdir(parents=True, exist_ok=True) """

        best_loss = float("inf")
        
        for epoch in range(self.epochs):
            train_loss = self.train_model() 
            val_loss = self.validate_model(best_loss, img_path=f"{wandb_path}/img/{run.id}")

            """ logging.info(f"train loss: {train_loss}")
            logging.info(f"validation loss: {val_loss}")
            logging.info(f"rollout loss: {rollout_loss}") """

            if epoch == 0:
                max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                logging.info(f"max GPU memory allocated: {max_mem_gb:.2f} GB")
            
            wandb.log({
                "train_loss": train_loss,
                "validation_loss": val_loss
            })

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), weight_path / "best_loss.pt")

        self.first_pass = True
        best_loss = float("inf")
        state_dict = torch.load(weight_path / "best_loss.pt", weights_only=True)
        self.model.load_state_dict(state_dict)

        for epoch in range(self.rollout_epochs):
            rollout_loss = self.train_rollout() 
            val_loss = self.validate_rollout(best_loss, img_path=f"{wandb_path}/img/{run.id}")

            """ logging.info(f"rollout loss: {rollout_loss}")
            logging.info(f"rollout validation loss: {val_loss}") """

            if epoch == 0:
                max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                logging.info(f"max GPU memory allocated (rollout): {max_mem_gb:.2f} GB")
            
            wandb.log({
                "rollout_loss": rollout_loss,
                "rollout_validation": val_loss
            })

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), weight_path / f"best_loss.pt")

        wandb.finish()
        #logging.info(f"best validation loss: {best_loss}")
