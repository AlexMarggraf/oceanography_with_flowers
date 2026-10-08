import xarray as xr
import xroms
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Pactcs30Dataset(Dataset):
    def __init__(self, timespan):
        self.data = []
        self.transform= None
        self.target_transform = None
        self.len = 0

        # load all data
        all_data = []
        for year in timespan:
            all_data.append(clean_pactcs30(open_pactcs30(year)))
            
        dataset = xr.concat(all_data, dim='time', data_vars='all')
        self.len = dataset.sizes["time"] - 1

        row, col = dataset.sizes["eta_rho"], dataset.sizes["xi_rho"]

        # store as tensor once
        for i in range(self.len+1):
            datapoint = dataset.isel(time=i)
            time = torch.full((row, col), datapoint.time.dt.dayofyear.values.item())
            features = [time]

            for var in datapoint.data_vars.keys():
                features.append(torch.from_numpy(datapoint[var].values))

            self.data.append(torch.stack(features))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        feature = self.data[idx]
        label = self.data[idx+1]
        return feature, label

def open_pactcs30(year):
    path = "../../../GROUP/pactcs30/pactcs30_" + str(year) + "_avg.zarr"
    # dates on the original ROMS output have the wrong time
    ds = xr.open_zarr(path, decode_times=False)
    # we have to set the units and calendar from 0000-01-01 00:00:00 and 365noleap
    ds['time'].attrs['units'] = "seconds since 1979-01-01 00:00:00"
    ds['time'].attrs['calendar'] = "gregorian"
    ds = xr.decode_cf(ds)  # now, interpret the calendar
    # read in the data with ROMS package - allows for easy plotting
    ds, _ = xroms.roms_dataset(ds, include_3D_metrics=False)
    return ds

def clean_pactcs30(old_dataset):
    # interpolate u velocity onto rho grid
    u_rho = 0.5 * (old_dataset.u.isel(xi_u=slice(0,-1)) + old_dataset.u.isel(xi_u=slice(1, None)))
    u_rho["xi_u"] = u_rho["xi_u"] + 1
    u_rho = u_rho.rename({"xi_u" : "xi_rho"})

    # interpolate v velocity onto rho grid
    v_rho = 0.5 * (old_dataset.v.isel(eta_v=slice(0,-1)) + old_dataset.v.isel(eta_v=slice(1, None)))
    v_rho["eta_v"] = v_rho["eta_v"] + 1
    v_rho = v_rho.rename({"eta_v" : "eta_rho"})
    
    new = xr.Dataset({
        "u" : u_rho.squeeze("s_rho"),
        "v" : v_rho.squeeze("s_rho"),
        "w" : old_dataset["w"].squeeze("s_rho"),
        "temp" : old_dataset["temp"].squeeze("s_rho"),
        "salt" : old_dataset["salt"].squeeze("s_rho"),
        "zeta" : old_dataset["zeta"],
        "DIC" : old_dataset["DIC"].squeeze("s_rho"),
        "PCO2OC" : old_dataset["PCO2OC"],
        "mask_rho" : old_dataset["mask_rho"]
    })

    # pad u and v to match grid size
    new.pad(
        u=1,
        v=1
    )

    new = new.fillna(0)
    return new