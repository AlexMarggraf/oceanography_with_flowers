import xarray as xr
import xroms
import xcmocean
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

class Pactcs30Dataset(Dataset):
    """
    Represents pactcs30 data in a torch dataset which can be used for a torch dataloader.

    Attributes:
        data ([torch.Tensor]): Datapoints of dataset stored as torch tensors.
        transform (None): Nessecary because of inheritance, not used here.
        target_transform (None): Nessecary because of inheritance, not used here.
        len (int): length of data.
    """
    def __init__(self, dataset):
        """
        Initialize dataset.

        Args:
            dataset (xarray.Dataset): Data to be used in Dataset.
        """

        self.data
        self.transform
        self.target_transform
        self.len
        
        # -1 because last entry has no next simulation step
        self.len = dataset.sizes["time"] - 1
        self.data = [None] * (self.len + 1)
        
        row, col = dataset.sizes["eta_rho"], dataset.sizes["xi_rho"]
        base_time = torch.ones((row, col), dtype=torch.float32)
        
        mask_rho = torch.from_numpy(dataset["mask_rho"].values)
        
        data_vars = list(dataset.data_vars)
        data_vars.remove("mask_rho")

        def store_tensor(idx):
            datapoint = dataset.isel(time=idx)
            # represent date as day of year
            time = datapoint.time.dt.dayofyear.values.item() * base_time
            features = [time, mask_rho]

            for var in data_vars:
                features.append(torch.from_numpy(datapoint[var].values))

            self.data[idx] = torch.stack(features)

        jobs = [i for i in range(self.len + 1)]
        
        # used to load data quicker
        with ThreadPoolExecutor((os.cpu_count() or 1)) as executor:
            list(executor.map(store_tensor, jobs))
            
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        feature = self.data[idx]
        label = self.data[idx+1][2:]
        return feature, label