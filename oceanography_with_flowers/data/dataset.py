import xarray as xr
import xroms
import xcmocean
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from einops import repeat, rearrange
import logging

class Pactcs30Dataset(Dataset):
    """
    Dataset for the ROMS Pactcs30 20 year hindcast.

    Args:
        path (str): Path to the stored data
        data_vars ([str]): All variables needed from the dataset
        pred_vars ([str]): All variables that should be predicted
        pad ((int, int)) | None: How much the image needs to be padded
        delta int: How big one prediction step is
        rollout: How many rollout steps are made
        keep_raw ([str]): All variables that shouldn't be normalized
        timespan ([int]): What years should be loaded
        depth (int): Amount of depth levels
        grid (bool): Whether the data should be stored as grid or graph
        latent_dim ((int, int)): Dimensions of the latent grid when gaot is used
    """

    def __init__(
        self,
        path,
        pred_vars,
        meta_vars,
        pad,
        delta=5,
        rollout=1,
        keep_raw=["mask_rho", "angle"],
        timespan=[2010],
        depth=1,
        grid=True,
        latent_dim=(256, 256)
    ):
        self.pred_vars = pred_vars
        self.delta = delta
        self.rollout = rollout
        self.pad = pad
        self.grid = grid
        self.latent_dim = latent_dim
        logging.info(f"delta: {delta}")
        logging.info(f"rollout: {rollout}")

        self.mean = dict()
        self.std = dict()

        data_vars = pred_vars + meta_vars
        dataset = self.load_pactcs30(path, timespan)
        dataset = self.clean_pactcs30(dataset, data_vars, pad, keep_raw=keep_raw)
        logging.info(f"data variables: {list(dataset.data_vars)}")
        logging.info(f"data dimensions: {list(dataset.dims)}")
        logging.info(f"dataset shape: {dataset.dims}")
        logging.info(f"types of data arrays: {dataset.dtypes}")

        # stored for translation from tensor to dataset
        self.mask_rho = dataset["mask_rho"]
        self.eta_rho = dataset["eta_rho"]
        self.xi_rho = dataset["xi_rho"]
        self.lat_rho = dataset["lat_rho"]
        self.lon_rho = dataset["lon_rho"]

        self.len = dataset.sizes["time"]
        coast_mask = torch.from_numpy(dataset["mask_rho"].values)
        meta_vars = [v for v in meta_vars if v != "mask_rho"] + ["lat_rho", "lon_rho"]

        if grid:
            self.bool_coast_mask = coast_mask.bool()
            # time and data are stored seperately to prevent slicing
            self.data = torch.empty((self.len, len(pred_vars), *coast_mask.shape))
            self.time = torch.empty((self.len, 2, *coast_mask.shape))

            metadata = [coast_mask]

            for v in meta_vars:
                tensor = torch.from_numpy(dataset[v].values)
                metadata.append(tensor)

            self.metadata = torch.stack(metadata)
            # shape should be [Rollout, metadata.shape]
            self.metadata = self.metadata.expand(rollout, *self.metadata.shape)
        else:
            coast_mask = coast_mask.flatten()
            self.bool_coast_mask = coast_mask.bool()

            # time and data are stored seperately to prevent slicing
            self.data = torch.empty((self.len, *coast_mask.shape, len(pred_vars),))
            self.time = torch.empty((self.len, *coast_mask.shape, 2))

            metadata = [coast_mask]

            for v in meta_vars:
                metadata.append(torch.from_numpy(dataset[v].values).flatten())

            self.metadata = torch.stack(metadata)
            # shape should be [Rollout, metadata.shape]
            self.metadata = self.metadata.expand(rollout, *self.metadata.shape)

            # latent gridpoints for gaot
            x_latent_steps = torch.linspace(-1, 1, latent_dim[0])
            y_latent_steps = torch.linspace(-1, 1, latent_dim[1])

            x_latent_grid, y_latent_grid = torch.meshgrid(x_latent_steps, y_latent_steps, indexing="ij")
            x_latent_grid = x_latent_grid.flatten()
            y_latent_grid = y_latent_grid.flatten()

            self.latent_grid = torch.stack([x_latent_grid, y_latent_grid], dim=-1)

            # longitude and latitude grid for geometric embeddings
            lon_min = np.min(dataset["lon_rho"])
            lon_max = np.max(dataset["lon_rho"])
            lat_min = np.min(dataset["lat_rho"])
            lat_max = np.max(dataset["lat_rho"])

            # mapping lat / lon onto [-1, 1]^2
            lon_normal = 2 * (dataset["lon_rho"] - lon_min) / (lon_max - lon_min) - 1
            lat_normal = 2 * (dataset["lat_rho"] - lat_min) / (lat_max - lat_min) - 1
            x_real_grid = torch.from_numpy(lon_normal.values)
            y_real_grid = torch.from_numpy(lat_normal.values)
            x_real_grid = x_real_grid.flatten()
            y_real_grid = y_real_grid.flatten()

            self.real_grid = torch.stack([x_real_grid, y_real_grid], dim=-1)

            logging.info(f"latent grid shape: {self.latent_grid.shape}")
            logging.info(f"real grid shape: {self.real_grid.shape}")

        logging.info(f"bool coast mask shape: {self.bool_coast_mask.shape}")
        logging.info(f"metadata shape: {self.metadata.shape}")

        row = dataset.sizes["eta_rho"]
        col = dataset.sizes["xi_rho"]

        if not self.grid:
            self.original_dim = (row, col)

        def store_tensor(idx):
            datapoint = dataset.isel(time=idx)
            base_time = torch.full((row, col), 2*torch.pi*datapoint["time"].item())

            # sinusoidal encodings such that jan 1. is the same as dec 31.
            if grid:
                self.time[idx] = torch.stack(
                    [
                        torch.sin(base_time),
                        torch.cos(base_time)
                    ]
                )
            if not grid:
                base_time = base_time.flatten()
                self.time[idx] = torch.stack(
                    [
                        torch.sin(base_time),
                        torch.cos(base_time)
                    ], dim=-1
                )

            if idx == 0:
                logging.info(f"time encoding shape: {self.time[0].shape}")

            features = []
            
            if grid:
                for v in pred_vars:
                    features.append(torch.from_numpy(datapoint[v].values))

                self.data[idx] = torch.stack(features)
            else:
                for v in pred_vars:
                    features.append(torch.from_numpy(datapoint[v].values).flatten())
                
                self.data[idx] = torch.stack(features, dim=-1)

            if idx == 0:
                logging.info(f"First entry of dataset shape: {self.data[idx].shape}")

        with ThreadPoolExecutor((os.cpu_count() or 1)) as executor:
            list(executor.map(store_tensor, [i for i in range(self.len)]))
            
    def __len__(self):
        return self.len - (self.rollout * self.delta)
        
    def __getitem__(self, idx):
        goal_idx = idx + self.delta * self.rollout

        # start point of prediction
        feature = self.data[idx]
        # label for first and last prediction
        label = torch.stack([self.data[idx + self.delta], self.data[goal_idx]])
        time_steps = torch.arange(idx, goal_idx, self.delta)

        # nessecary time encodings needed during rollout prediction
        time = torch.stack([self.time[i] for i in time_steps])

        if self.grid:
            metadata = torch.cat(
                [
                    self.metadata,
                    time
                ], dim=1
            )
        else:
            metadata = torch.cat(
                [
                    self.metadata,
                    time
                ], dim=-1
            )
                
        return feature, label, metadata

    def get_mask(self):
        return self.bool_coast_mask
    
    def get_latent_grid(self):
        return self.latent_grid
    
    def get_real_grid(self):
        return self.real_grid

    def open_pactcs30(self, path, year=2010):
        """
        Open single pactcs30.zarr file.
    
        Args:
            path (str): Path to the data file
            year (int): Year of the simulation data
        
        Returns:
            xarray.Dataset: Data as a dataset
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
        Load multiple pactcs30.zarr files into one dataset.
    
        Args:
            path (str): Path to the data files
            timespan ([int]): List of all years the data should cover
    
        Returns:
            xarray.Dataset: All data in one dataset
        """
    
        all_data = [self.open_pactcs30(path, year) for year in timespan]
        dataset = xr.concat(
            all_data, 
            dim='time', 
            data_vars="minimal", 
            coords="minimal"
        )
        return dataset
    
    def clean_pactcs30(
        self,
        dataset,
        data_vars,
        pad,
        keep_raw=["mask_rho", "angle"]
    ):
        """
        Clean and prepare data for usage in training. Normalization happens during
        this step.
    
        Args:
            dataset (xarray.Dataset): Dataset that's supposed to be cleaned
            data_vars ([str]): All variables to include from dataset
            pad ((int, int)) | None: The amount of rows / collumns to pad the data to
            keep_raw ([str]): All variables that should not be normalized

        Returns:
            xarray.Dataset: Cleaned data in dataset.
        """

        # normalize time
        dataset["time"] = xr.where(
            dataset["time"].dt.is_leap_year,
            (dataset["time"].dt.dayofyear.values - 1) / 365,
            (dataset["time"].dt.dayofyear.values - 1) / 364
        )
        logging.info(f"first and last time value {dataset["time"][0].values} / {dataset["time"][-1].values}")

        # clip values below 0 to avoid plotting errors
        dataset["NO3"] = dataset["NO3"].clip(min=0)
        dataset["TOT_CHL"] = dataset["TOT_CHL"].clip(min=0)
        logging.info(f"min NO3 and TOT_CHL values: {np.nanmin(dataset["NO3"])} / {np.nanmin(dataset["TOT_CHL"])}")
        
        # linear interpolation to go from u to rho grid
        u_first = dataset["u"].isel(xi_u=slice(0, -1))
        u_last = dataset["u"].isel(xi_u=slice(1, None))
        u_first["xi_u"] = u_first["xi_u"] + 1
        
        u_rho = 0.5 * (u_first + u_last)
        u_rho = u_rho.pad(xi_u=(1,1), mode="edge")
        u_rho = u_rho.rename({"xi_u" : "xi_rho"})
        u_rho = u_rho.assign_coords(xi_rho=dataset["xi_rho"])
        logging.info(f"interpolated u shape: {u_rho.shape}")
    
        # linear interploation to go from v to rho grid
        v_first = dataset["v"].isel(eta_v=slice(0, -1))
        v_last = dataset["v"].isel(eta_v=slice(1, None))
        v_first["eta_v"] = v_first["eta_v"] + 1
        
        v_rho = 0.5 * (v_first + v_last)
        v_rho = v_rho.pad(eta_v=(1,1), mode="edge")
        v_rho = v_rho.rename({"eta_v" : "eta_rho"})
        v_rho = v_rho.assign_coords(eta_rho=dataset["eta_rho"])
        logging.info(f"interpolated v shape: {v_rho.shape}")
    
        var_norm = [v for v in data_vars if v not in keep_raw]
    
        # get mean and standard deviation for normalization
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
    
        # select data that is needed for prediction
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
        
        # normalization of all data
        for v in var_norm:
            new[v] = (new[v] - self.mean[v]) / self.std[v]

        # Pad data if model reduces spatial dimensions during forward pass
        if pad is not None:
            x_pad, y_pad = pad

            # needed for plotting later
            lat_attrs = new["lat_rho"].attrs
            lon_attrs = new["lon_rho"].attrs

            lat_pad = new["lat_rho"].pad(
                eta_rho=(x_pad, x_pad),
                xi_rho=(y_pad, y_pad),
                mode="constant",
                constant_values=0
            )

            lon_pad = new["lon_rho"].pad(
                eta_rho=(x_pad, x_pad),
                xi_rho=(y_pad, y_pad),
                mode="constant",
                constant_values=0
            )

            new = new.pad(
                eta_rho=(x_pad, x_pad),
                xi_rho=(y_pad, y_pad),
                mode="constant",
                constant_values=0
            )

            new = new.assign_coords(
                lat_rho=(lat_pad.dims, lat_pad.values, lat_attrs),
                lon_rho=(lon_pad.dims, lon_pad.values, lon_attrs),
            )

        new = new.fillna(0)
        return new

    def tensor_to_dataset(self, tensor, pred_vars=None):
        """
        Turn tensor into dataset.
    
        Args:
            tensor (torch.Tensor): The tensor that will be turned into a dataset
            pred_vars ([str]): The variables that were predicted during training
    
        Returns:
            xarray.Dataset: Tensor represented as dataset
        """

        if pred_vars is None:
            pred_vars = self.pred_vars

        if not self.grid:
            # store data in grid rather than graph
            tensor = rearrange(tensor, "(w h) c -> c w h", w=self.original_dim[0], h=self.original_dim[1])
    
        tensor_data = tensor.numpy()
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

        if len(pred_vars) != len(data_array):
            logging.error(f"variable list of length {len(pred_vars)} and tensor of length {len(data_array)} cannot be combined into dataset!")

        dataset = xr.Dataset(dict(zip(pred_vars, data_array)))
        dataset["mask_rho"] = self.mask_rho
        
        for v in pred_vars:
            # reverse normalization and turn 0 back to nan for better plotting
            dataset[v] = (dataset[v] * self.std[v]) + self.mean[v]
            dataset[v] =  dataset[v].where(dataset["mask_rho"] == 1)
        
        dataset = dataset.assign_coords(
            lat_rho=self.lat_rho,
            lon_rho=self.lon_rho,
        )
    
        return dataset
    
    def plot_dataset(
        self,
        dataset,
        path,
        data_vars=None,
        scales=None,
        return_scales=False,
    ):
        """Plot pactcs30 dataset with consistent color scaling.

        Args:
            dataset (xarray.Dataset): Dataset to be visualized.
            path (str | Path): Where to store data.
            data_vars (list[str]): Data variables to be visualized.
            scales (dict, optional): Mapping of {var_name: (vmin, vmax)}.
                If provided, these limits override automatic scaling.
            return_scales (bool): If True, computes and returns the dictionary
                of scales used across all variables. Defaults to False.

        Returns:
            dict: {var_name: (vmin, vmax)} if return_scales is True, else None.
        """

        if data_vars is None:
            data_vars = self.pred_vars

        create_dir = Path(path)
        create_dir.mkdir(parents=True, exist_ok=True)

        if return_scales:
            used_scales = dict()

        for v in data_vars:
            fig, ax = plt.subplots(figsize=(8, 6))
            da = dataset[v]

            if self.pad is not None:
                # reverse padding if applied for forward pass
                x_pad, y_pad = self.pad
                da = da.isel(eta_rho=slice(x_pad, -x_pad), xi_rho=slice(y_pad, -y_pad))

            if scales is not None and v in scales:
                vmin, vmax = scales[v]
            else:
                # calculate vmin and vmax to keep same scales between lable and prediction plots
                vmin, vmax = np.nanpercentile(da, 2), np.nanpercentile(da, 98)

            if return_scales:
                used_scales[v] = (vmin, vmax)

            plot_kwargs = {
                "ax": ax,
                "x": "longitude",
                "y": "latitude",
                "center": False,
                "cmap": "cmo.thermal",
                "robust": True,
                "vmin": vmin,
                "vmax": vmax
            }

            da.cf.plot(**plot_kwargs)

            ax.set_title(f"{v}")
            fig.savefig(create_dir / f"{v}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        if return_scales:
            return used_scales
