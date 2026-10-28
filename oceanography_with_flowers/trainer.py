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
        
        for i, (feature, label) in enumerate(self.dls[0]):
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
            self.optimizer.zero_grad(set_to_none=True)

        return train_loss / no_batches
    
    def validate_model(self, plot=False, img_path=None):
        """
        One iteration of the validation loop.

        Returns:
            int: Average validation loss of all batches
        """
                
        self.model.eval()
        val_loss = 0
        no_batches = len(self.dls[1])
    
        with torch.no_grad():
            for feature, label in self.dls[1]:
                feature = feature.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                pred = self.model(feature) 

                v_mask = self.mask.expand_as(pred)

                # loss is masked to ignore regions not in ocean
                loss = self.loss_fn(pred[v_mask], label[v_mask])
                val_loss += loss.item()

            val_loss /= no_batches
        
            if plot:
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
        rollout_loss = 0
        no_batches = len(self.dls[0])

        for i, (curr_state, label, metadata) in enumerate(self.dls[0]):
            curr_state = curr_state.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            metadata = metadata.to(self.device, non_blocking=True)

            with torch.no_grad():
                for j in range(self.dls[0].dataset.rollout):
                    curr_state = torch.cat([curr_state, metadata[:, j]], dim=1)
                    curr_state = self.model(curr_state)

            curr_state = torch.cat([curr_state, metadata[:, -1]], dim=1)
            curr_state = self.model(curr_state)

            if self.first_pass:
                logging.info(f"rollout state shape: {curr_state.shape}")
                logging.info(f"rollout label shape: {label.shape}")
                logging.info(f"rollout metadata shape: {metadata.shape}")
                self.first_pass = False

            v_mask = self.mask.expand_as(label)
            loss = self.loss_fn(curr_state[v_mask], label[v_mask])
            rollout_loss += loss.item()
            loss.backward()

            self.rollout_optimizer.step()
            self.rollout_scheduler.step()
            self.rollout_optimizer.zero_grad(set_to_none=True)

        return rollout_loss / no_batches
    
    def validate_rollout(self, best_loss, img_path=None):
        self.model.eval()
        val_loss = 0
        no_batches = len(self.dls[1])

        with torch.no_grad():
            for curr_state, label, metadata in self.dls[1]:
                curr_state = curr_state.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                metadata = metadata.to(self.device,non_blocking=True)

                for i in range(self.dls[1].dataset.rollout):
                    curr_state = torch.cat([curr_state, metadata[:, i]], dim=1)
                    curr_state = self.model(curr_state)

                v_mask = self.mask.expand_as(label)
                loss = self.loss_fn(curr_state[v_mask], label[v_mask])

                val_loss += loss.item()
        
            val_loss /= no_batches

            if val_loss < best_loss:
                curr_state, label, metadata = next(iter(self.dls[1]))
                curr_state = curr_state.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                metadata = metadata.to(self.device,non_blocking=True)

                for i in range(self.dls[1].dataset.rollout):
                    curr_state = torch.cat([curr_state, metadata[:, i]], dim=1)
                    curr_state = self.model(curr_state)

                curr_state = self.dls[0].dataset.tensor_to_dataset(curr_state[0].detach().cpu(), self.pred_vars)
                label = self.dls[0].dataset.tensor_to_dataset(label[0].detach().cpu(), self.pred_vars)

                scales = self.dls[0].dataset.plot_dataset(
                    label, 
                    f"{img_path}/rollout_label", 
                    self.pred_vars, 
                    return_scales=True
                )
                self.dls[0].dataset.plot_dataset(
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
        weight_path = Path(f"{weight_path}/run_{run.id}")
        weight_path.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        
        for epoch in range(self.epochs):
            train_loss = self.train_model() 
            val_loss = self.validate_model()

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
        state_dict = torch.load(weight_path / "best_loss.pt", weights_only=True)
        self.model.load_state_dict(state_dict)

        # Assumption is, that the rollout will get better no matter what
        best_loss = float("inf")
        self.dls[0].dataset.rollout = 2
        self.dls[1].dataset.rollout = self.rollout

        for epoch in range(self.rollout_epochs):
            rollout_loss = self.train_rollout()
            val_loss = self.validate_rollout(
                best_loss, img_path=f"{wandb_path}/img/{run.id}"
            )

            if epoch == 0:
                max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                logging.info(f"max GPU memory allocated during rollout: {max_mem_gb:.2f} GB")

            wandb.log({
                "rollout_loss": rollout_loss, 
                "rollout_validation": val_loss
            })

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), weight_path / "best_loss.pt")
        
        self.dls[1].dataset.rollout = 1
        val_loss = self.validate_model(plot=True, img_path=f"{wandb_path}/img/{run.id}")
        logging.info(f"final one step prediction loss: {val_loss}")
        wandb.finish()
