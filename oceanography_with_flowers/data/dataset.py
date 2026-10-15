import xarray as xr
import xroms
import xcmocean
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

class Pactcs30Dataset(Dataset):
    def __init__(self, path, data_vars, keep_raw=["mask_rho", "angle"], timespan=[2010], depth=1):
        self.data = None
        self.len = 0

        self.mean = dict()
        self.std = dict()

        self.base_time = None
        self.coast_mask = None
        self.positional = None

        self.mask_rho = None
        self.eta_rho = None
        self.xi_rho = None
        self.lat_rho = None
        self.lon_rho = None

        dataset = self.load_pactcs30(path, timespan)
        dataset = self.clean_pactcs30(dataset, data_vars, keep_raw)

        self.mask_rho = dataset["mask_rho"]
        self.eta_rho = dataset["eta_rho"]
        self.xi_rho = dataset["xi_rho"]
        self.lat_rho = dataset["lat_rho"]
        self.lon_rho = dataset["lon_rho"]
        
        # -1 because last entry has no next simulation step
        self.len = dataset.sizes["time"] - 1
        self.data = [None] * (self.len + 1)

        # i chose xi here so i don't have to transpose
        row, col = dataset.sizes["eta_rho"], dataset.sizes["xi_rho"]
        time_vec = torch.linspace(0, 2*np.pi, col)
        time_mat = time_vec.repeat(row, 1)

        self.base_time = torch.stack(
            [
                torch.sin(time_mat),
                torch.cos(time_mat)
            ]
        )
        
        self.coast_mask = torch.from_numpy(dataset["mask_rho"].values).unsqueeze(0)

        h_xi = torch.from_numpy(dataset["dx"].values)
        h_eta = torch.from_numpy(dataset["dy"].values)
        
        cos_angle = torch.cos(torch.from_numpy(dataset["angle"].values))
        sin_angle = torch.sin(torch.from_numpy(dataset["angle"].values))

        self.positional = torch.stack(
            [
                h_xi * cos_angle,
                h_xi * sin_angle,
                -h_eta * sin_angle,
                h_eta * cos_angle,
            ]
        )

        # dx and dy shouldn't be in keep_raw since they should be normalized, which keep_raw prevents
        exclude = keep_raw + ["dx", "dy"]
        feature_vars = [v for v in data_vars if v not in exclude]
        
        def store_tensor(idx):
            datapoint = dataset.isel(time=idx)
            time = datapoint["time"].item() * self.base_time
            features = [time]

            for v in feature_vars:
                features.append(torch.from_numpy(datapoint[v].values).unsqueeze(0))

            self.data[idx] = torch.cat(features, dim=0)

        jobs = [i for i in range(self.len + 1)]
        
        with ThreadPoolExecutor((os.cpu_count() or 1)) as executor:
            list(executor.map(store_tensor, jobs))
            
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        feature = self.data[idx]
        feature = torch.cat(
            [
                self.coast_mask,
                self.positional,
                feature
            ], dim=0
        )
        label = self.data[idx+1][2:]
        return feature, label

    def get_mask(self):
        return self.coast_mask.bool()

    def open_pactcs30(self, path, year=2010):
        """
        Open single pactcs30.zarr file.
    
        Args:
            year (int): Year of the simulation data.
        
        Returns:
            xarray.Dataset: Data as a dataset.
        """
    
        file_path = path + "pactcs30_" + str(year) + "_avg.zarr"
        dataset = xr.open_zarr(file_path, decode_times=False)
        # convert time to correct format
        dataset['time'].attrs['units'] = "seconds since 1979-01-01 00:00:00"
        dataset['time'].attrs['calendar'] = "gregorian"
        dataset = xr.decode_cf(dataset)
        # read data with roms package for easy plotting
        dataset, _ = xroms.roms_dataset(dataset, include_3D_metrics=False)
        return dataset.load()
    
    def load_pactcs30(self, path, timespan=[2010]):
        """
        Load multiple pactcs30.zarr files into one dataset
    
        Args:
            timespan ([int]): List of all years the data should cover.
    
        Returns:
            xarray.Dataset: All data in one dataset.
        """
    
        all_data = [self.open_pactcs30(path, year) for year in timespan]
        dataset = xr.concat(all_data, dim='time', data_vars="minimal", coords="minimal")
        return dataset
    
    def clean_pactcs30(self, dataset, data_vars, keep_raw=["mask_rho", "angle"]):
        """
        Clean and prepare data for usage in training.
    
        Args:
            dataset (xarray.Dataset): Dataset that's supposed to be cleaned.
            mean ({string : float}): Dictionary of the mean values of all data variables.
            std ({string : float}): Dictionary of the standard deviation of all data variables.
    
        Returns:
            xarray.Dataset: Cleaned data in dataset.
        """
    
        dataset["time"] = xr.where(
            dataset["time"].dt.is_leap_year,
            (dataset["time"].dt.dayofyear.values - 1) / 365,
            (dataset["time"].dt.dayofyear.values - 1) / 364
        )
        
        # linear interpolation to go from u to rho grid
        u_first = dataset["u"].isel(xi_u=slice(0,-1))
        u_last = dataset["u"].isel(xi_u=slice(1,None))
        u_first["xi_u"] = u_first["xi_u"] + 1
        
        u_rho = 0.5 * (u_first + u_last)
        u_rho = u_rho.pad(xi_u=(1,1), mode="edge")
        u_rho = u_rho.rename({"xi_u" : "xi_rho"})
        u_rho = u_rho.assign_coords(xi_rho=dataset.xi_rho)
    
        # linear interploation to go from v to rho grid
        v_first = dataset["v"].isel(eta_v=slice(0,-1))
        v_last = dataset["v"].isel(eta_v=slice(1,None))
        v_first["eta_v"] = v_first["eta_v"] + 1
        
        v_rho = 0.5 * (v_first + v_last)
        v_rho = v_rho.pad(eta_v=(1,1), mode="edge")
        v_rho = v_rho.rename({"eta_v" : "eta_rho"})
        v_rho = v_rho.assign_coords(eta_rho=dataset.eta_rho)
    
        var_norm = [v for v in data_vars if v not in keep_raw]
    
        for v in var_norm:
            match v:
                case "u":
                    self.mean[v] = np.nanmean(u_rho)
                    self.std[v] = np.nanstd(u_rho)
                case "v":
                    self.mean[v] = np.nanmean(v_rho)
                    self.std[v] = np.nanstd(v_rho)
                case _:
                    self.mean[v] = np.nanmean(dataset[v])
                    self.std[v] = np.nanstd(dataset[v])
    
        new = xr.Dataset()
    
        for v in data_vars:
            match v:
                case "u":
                    new[v] = u_rho.squeeze("s_rho")
                case "v":
                    new[v] = v_rho.squeeze("s_rho")
                case _:
                    if "s_rho" in dataset[v].dims:
                        new[v] = dataset[v].squeeze("s_rho")
                    else:
                        new[v] = dataset[v]
        
        # used to make data behave more like normal distribution
        for v in var_norm:
            new[v] = (new[v] - self.mean[v]) / self.std[v]
    
        new = new.fillna(0)
        return new

    def tensor_to_dataset(self, tensor, pred_vars):
        """
        Turn tensor into dataset.
    
        Args:
            tensor (torch.Tensor): The tensor that will be turned into a dataset.
            mean ({string : float}): Dictionary of the mean values of all data variables.
            std ({string : float}): Dictionary of the standard deviation of all data variables.
            mask (xarray.DataArray): The mask values for land/ocean.
            eta_rho (xarray.DataArray): The rho points in y direction.
            xi_rho (xarray.DataArray): The rho points in x direction.
            lat_rho (xarray.DataArray): Latitude of rho points.
            lon_rho (xarray.DataArray): Longitude of rho points.
    
        Returns:
            xarray.Dataset: Tensor represented as dataset.
        """
    
        tensor_data = tensor.detach().cpu().numpy()
        data_array = []
    
        for t in tensor_data:
            data_array.append(
                xr.DataArray(
                    data=t,
                    dims=["eta_rho", "xi_rho"],
                    coords={
                        "eta_rho": self.eta_rho,
                        "xi_rho": self.xi_rho,
                    },
                )
            )

        dataset = xr.Dataset(dict(zip(pred_vars, data_array)))
        dataset["mask_rho"] = self.mask_rho
        
        for v in pred_vars:
            # reverse normalization and turn 0 back to nan for better plotting
            dataset[v] = (dataset[v] * self.std[v]) + self.mean[v]
            dataset[v] =  dataset[v].where(dataset["mask_rho"] == 1)
        
        dataset = dataset.assign_coords(
            lat_rho = self.lat_rho,
            lon_rho = self.lon_rho,
        )
    
        return dataset
    
    def plot_dataset(self, dataset, vars=["temp"]):
        """
        Plot pactcs30 dataset.
    
        Args:
            dataset (xarray.Dataset): Dataset to be visualized.
            vars ([string]): Data variables to be visualized.
        """
    
        for v in vars:
            dataset[v].cf.plot(
                x="longitude",
                y="latitude",
                robust=True,
                cmap="cmo.thermal",
            )
            plt.title(f"{v}")
            plt.show()