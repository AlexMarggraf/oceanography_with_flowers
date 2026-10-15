import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from pathlib import Path

class Pactcs30Trainer():
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
        self.model.train()
        train_loss = 0
        no_batches = len(self.train_dl)
        
        for batch, (feature, label) in enumerate(self.train_dl):
            feature = feature.to(self.device)
            label = label.to(self.device)
            pred = self.model(feature)
            v_mask = self.mask.expand_as(pred)
            
            loss = self.loss_fn(pred[v_mask], label[v_mask])
            train_loss += loss.item()
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return train_loss / no_batches
    
    def validate_model(self):
        self.model.eval()
        val_loss = 0
        no_batches = len(self.val_dl)
    
        with torch.no_grad():
            for feature, label in self.val_dl:
                feature = feature.to(self.device)
                label = label.to(self.device)
                pred = self.model(feature)
                v_mask = self.mask.expand_as(pred)
                
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