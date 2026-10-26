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
        self.first_pass = True
    
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

                if self.first_pass:
                    logging.info(f"feature shape: {feature.shape}")
                    logging.info(f"label shape: {label.shape}")
                    logging.info(f"prediction shape: {pred.shape}")
                    self.first_pass = False
            else:
                feature, time, label = data
                feature = feature.to(self.device, non_blocking=True)
                time = time.to(self.device)
                label = label.to(self.device, non_blocking=True)
                pred = self.model(feature, param=time)

                if self.first_pass:
                    logging.info(f"feature shape: {feature.shape}")
                    logging.infe(f"time shape: {time.shape}")
                    logging.info(f"label shape: {label.shape}")
                    logging.info(f"prediction shape: {pred.shape}")
                    self.first_pass = False

            v_mask = self.mask.expand_as(pred)

            # loss is masked to ignore regions not in ocean
            loss = self.loss_fn(pred[v_mask], label[v_mask])
            train_loss += loss.item()
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

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

            val_loss /= no_batches
        
            if val_loss < best_loss:
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

                scales = self.dls[0].dataset.plot_dataset(label, f"{img_path}/label", self.pred_vars, return_scales=True)
                self.dls[0].dataset.plot_dataset(pred, f"{img_path}/prediction", self.pred_vars, scales=scales)

            if self.use_sin_enc:
                #TODO unhardcode length of rollout
                curr_day = np.random.randint(0, len(self.dls[1].dataset) - 1 - (self.dls[1].dataset.delta * 10)) 
                curr_rollout, _ = self.dls[1].dataset[curr_day]
                _, rollout_label = self.dls[1].dataset[curr_day + (self.dls[1].dataset.delta * 10)]
                logging.info(f"rollout start day: {curr_day}")

                curr_rollout = curr_rollout[7:]
                curr_rollout = curr_rollout.unsqueeze(0)
                rollout_label = rollout_label.unsqueeze(0)

                curr_rollout = curr_rollout.to(self.device)
                rollout_label = rollout_label.to(self.device)

                for _ in range(10):
                    metadata, _ = self.dls[1].dataset[curr_day]
                    metadata = metadata[:7]
                    metadata = metadata.unsqueeze(0)
                    metadata = metadata.to(self.device)

                    curr_rollout = torch.cat(
                        [
                            metadata,
                            curr_rollout
                        ], dim=1
                    )

                    curr_rollout = self.model(curr_rollout)
                    curr_day += self.dls[1].dataset.delta

            else:
                #TODO unhardcode length of rollout
                curr_day = np.random.randint(0, len(self.dls[1].dataset) - 1 - (self.dls[1].dataset.delta * 10)) 
                curr_rollout, _, _ = self.dls[1].dataset[curr_day]
                _, _, rollout_label = self.dls[1].dataset[curr_day + (self.dls[1].dataset.delta * 10)]
                logging.info(f"rollout start day: {curr_day}")

                curr_rollout = curr_rollout[5:]
                curr_rollout = curr_rollout.unsqueeze(0)
                rollout_label = rollout_label.unsqueeze(0)

                curr_rollout = curr_rollout.to(self.device)
                rollout_label = rollout_label.to(self.device)

                for _ in range(10):
                    metadata, time, _ = self.dls[1].dataset[curr_day]
                    time = time.to(self.device)
                    #TODO unhardcode this lowk
                    metadata = metadata[:5]
                    metadata = metadata.unsqueeze(0)
                    metadata = metadata.to(self.device)

                    curr_rollout = torch.cat(
                        [
                            metadata,
                            curr_rollout
                        ], dim=1
                    )

                    curr_rollout = self.model(curr_rollout, time)
                    curr_day += self.dls[1].dataset.delta
                
            v_mask = self.mask.expand_as(curr_rollout)
            rollout_loss = self.loss_fn(curr_rollout[v_mask], rollout_label[v_mask])

        return val_loss, rollout_loss

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
            val_loss, rollout_loss = self.validate_model(best_loss, img_path=f"{wandb_path}/img/{run.id}")

            """ logging.info(f"train loss: {train_loss}")
            logging.info(f"validation loss: {val_loss}")
            logging.info(f"rollout loss: {rollout_loss}") """

            if epoch == 0:
                max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                logging.info(f"max GPU memory allocated: {max_mem_gb:.2f} GB")
            
            wandb.log({
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "rollout_loss" : rollout_loss
            })

            if val_loss < best_loss:
                best_loss = val_loss
                #torch.save(self.model.state_dict(), weight_path / f"best_loss.pt")

        wandb.finish()
        #logging.info(f"best validation loss: {best_loss}")
