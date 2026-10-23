import numpy as np
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

    def __init__(self, model, dls, loss_fn, optimizer, scheduler, epochs, device, pred_vars, use_sin_enc=False):
        self.model = model
        self.dls = dls
        self.mask = dls[0].dataset.get_mask()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.pred_vars = pred_vars
        self.use_sin_enc = use_sin_enc
    
    def train_model(self):
        """
        One iteration of the training loop.

        Returns:
            int: Average training loss of all batches
        """

        self.model.train()
        train_loss = 0
        no_batches = len(self.dls[0])
        
        for data in self.dls[0]:
            if self.use_sin_enc:
                feature, label = data
                feature = feature.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                pred = self.model(feature) 
            else:
                feature, time, label = data
                feature = feature.to(self.device, non_blocking=True)
                time = time.to(self.device)
                label = label.to(self.device, non_blocking=True)
                pred = self.model(feature, param=time)

            v_mask = self.mask.expand_as(pred)

            # loss is masked to ignore regions not in ocean
            loss = self.loss_fn(pred[v_mask], label[v_mask])
            train_loss += loss.item()
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return train_loss / no_batches
    
    def validate_model(self, img_path=None, save_plot=False):
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
                if self.use_sin_enc:
                    feature, label = data
                    feature = feature.to(self.device, non_blocking=True)
                    label = label.to(self.device, non_blocking=True)
                    pred = self.model(feature) 
                else:
                    feature, time, label = data
                    feature = feature.to(self.device, non_blocking=True)
                    time = time.to(self.device)
                    label = label.to(self.device, non_blocking=True)
                    pred = self.model(feature, param=time)

                v_mask = self.mask.expand_as(pred)

                # loss is masked to ignore regions not in ocean
                loss = self.loss_fn(pred[v_mask], label[v_mask])
                val_loss += loss.item()
        
        if save_plot:
            with torch.no_grad():
                if self.use_sin_enc:
                    feature, label = next(iter(self.dls[1]))
                    feature = feature.to(self.device)
                    pred = self.model(feature)
                else:
                    feature, time, label = next(iter(self.dls[1]))
                    feature = feature.to(self.device)
                    time = time.to(self.device)
                    pred = self.model(feature, time)

                pred = self.dls[0].dataset.tensor_to_dataset(pred[0].detach().cpu(), self.pred_vars)
                label = self.dls[0].dataset.tensor_to_dataset(label[0].detach().cpu(), self.pred_vars)

                self.dls[0].dataset.plot_dataset(label, f"{img_path}/label", self.pred_vars)
                self.dls[0].dataset.plot_dataset(pred, f"{img_path}/prediction", self.pred_vars)

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

        best_loss = float("inf")
        
        for epoch in range(self.epochs):
            train_loss = self.train_model()
            if epoch == 0 or epoch == self.epochs - 1:
                val_loss = self.validate_model(img_path=f"{wandb_path}/img/run_{run.id}/epoch_{epoch}", save_plot=True)
            else:
                val_loss = self.validate_model()

            if epoch == 0:
                max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"Max GPU Memory Allocated: {max_mem_gb:.2f} GB", flush=True)
            
            wandb.log({
                "train_loss": train_loss,
                "validation_loss": val_loss,
            })

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), weight_path / f"best_loss.pt")

        wandb.finish()
