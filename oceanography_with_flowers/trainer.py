import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from pathlib import Path

class Pactcs30Trainer():
    """
    A trainer for the pactcs30 dataset which trains and validates the data.

    Args:
        model (nn.Module): Model used for training
        train_ds (Pactcs30Dataset): Dataset of the training data
        val_ds (Pactcs30Dataset): Dataset of the validation data
        batch_size (int): Batch size
        loss_fn (Callable): Loss function to be used in training and validation
        optimizer: Optimizer used during training
        scheduler: Scheduler used during training
        epochs: Epochs
        device: The device used for training
        pred_vars: The variables that are predicted by the model
    """

    def __init__(self, model, train_ds, val_ds, batch_size, loss_fn, optimizer, scheduler, epochs, device, pred_vars):
        self.model = model
        self.train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        self.mask = train_ds.get_mask()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.pred_vars = pred_vars
    
    def train_model(self):
        """
        One iteration of the training loop.

        Returns:
            int: Average training loss of all batches
        """

        self.model.train()
        train_loss = 0
        no_batches = len(self.train_dl)
        
        for _, (feature, label) in enumerate(self.train_dl):
            feature = feature.to(self.device)
            label = label.to(self.device)
            pred = self.model(feature)
            v_mask = self.mask.expand_as(pred)

            # loss is masked to ignore regions not in ocean
            loss = self.loss_fn(pred[v_mask], label[v_mask])
            train_loss += loss.item()
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return train_loss / no_batches
    
    def validate_model(self):
        """
        One iteration of the validation loop.

        Returns:
            int: Average validation loss of all batches
        """
                
        self.model.eval()
        val_loss = 0
        no_batches = len(self.val_dl)
    
        with torch.no_grad():
            for feature, label in self.val_dl:
                feature = feature.to(self.device)
                label = label.to(self.device)
                pred = self.model(feature)
                v_mask = self.mask.expand_as(pred)

                # loss is masked to ignore regions not in ocean
                loss = self.loss_fn(pred[v_mask], label[v_mask])
                val_loss += loss.item()
        
        #feature, label = next(iter(self.val_dl))
        #feature = feature.to(device)
        #pred = self.model(feature)

        #pred = self.train_dl.dataset.tensor_to_dataset(pred[0], self.pred_vars)
        #label = self.train_dl.dataset.tensor_to_dataset(label[0], self.pred_vars)

        #self.train_dl.dataset.plot_dataset(label)
        #self.train_dl.dataset.plot_dataset(pred)

        return val_loss / no_batches

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
        
        for i in range(self.epochs):
            train_loss = self.train_model()
            val_loss = self.validate_model()
            
            wandb.log({
                "epoch": i,
                "train_loss": train_loss,
                "validation_loss": val_loss,
            })

            torch.save(self.model.state_dict(), weight_path / f"model_{i}.pt")

        wandb.finish()