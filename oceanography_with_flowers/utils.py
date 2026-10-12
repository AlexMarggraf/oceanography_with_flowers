import xarray as xr
import xroms
import xcmocean
import matplotlib.pyplot as plt
import numpy as np

def open_pactcs30(year=2010):
    """
    Open single pactcs30.zarr file.

    Args:
        year (int): Year of the simulation data.
    
    Returns:
        xarray.Dataset: Data as a dataset.
    """

    path = "../../../GROUP/pactcs30/pactcs30_" + str(year) + "_avg.zarr"
    ds = xr.open_zarr(path, decode_times=False)
    # convert time to correct format
    ds['time'].attrs['units'] = "seconds since 1979-01-01 00:00:00"
    ds['time'].attrs['calendar'] = "gregorian"
    ds = xr.decode_cf(ds)
    # read data with roms package for easy plotting
    ds, _ = xroms.roms_dataset(ds, include_3D_metrics=False)
    return ds

def load_pactcs30(timespan=[2010]):
    """
    Load multiple pactcs30.zarr files into one dataset

    Args:
        timespan ([int]): List of all years the data should cover.

    Returns:
        xarray.Dataset: All data in one dataset.
        {string : float}: Dictionary of the mean values of all data variables.
        {string : float}: Dictionary of  the standard deviation of all data variables.
    """

    all_data = [open_pactcs30(year) for year in timespan]
    dataset = xr.concat(all_data, dim='time', data_vars="minimal", coords="minimal").compute()
    mean = get_mean(dataset)
    std = get_std(dataset)

    return dataset, mean, std

def clean_pactcs30(dataset, mean, std):
    """
    Clean and prepare data for usage in training.

    Args:
        dataset (xarray.Dataset): Dataset that's supposed to be cleaned.
        mean ({string : float}): Dictionary of the mean values of all data variables.
        std ({string : float}): Dictionary of the standard deviation of all data variables.

    Returns:
        xarray.Dataset: Cleaned data in dataset.
    """

    # linear interpolation to go from u to rho grid
    u_rho = 0.5 * (dataset.u.isel(xi_u=slice(0,-1)) + dataset.u.isel(xi_u=slice(1, None)))
    u_rho["xi_u"] = u_rho["xi_u"] + 1
    u_rho = u_rho.rename({"xi_u" : "xi_rho"})

    # linear interploation to gro from v to rho grid
    v_rho = 0.5 * (dataset.v.isel(eta_v=slice(0,-1)) + dataset.v.isel(eta_v=slice(1, None)))
    v_rho["eta_v"] = v_rho["eta_v"] + 1
    v_rho = v_rho.rename({"eta_v" : "eta_rho"})
    
    new = xr.Dataset({
        "u" : u_rho.squeeze("s_rho"),
        "v" : v_rho.squeeze("s_rho"),
        "w" : dataset["w"].squeeze("s_rho"),
        "temp" : dataset["temp"].squeeze("s_rho"),
        "salt" : dataset["salt"].squeeze("s_rho"),
        "zeta" : dataset["zeta"],
        "DIC" : dataset["DIC"].squeeze("s_rho"),
        "PCO2OC" : dataset["PCO2OC"],
        "mask_rho" : dataset["mask_rho"]
    })

    # needed after interpolation
    new.pad(
        u=1,
        v=1,
    )

    new = new.fillna(0)

    data_vars = list(new.data_vars)
    data_vars.remove("mask_rho")
    
    # used to make data behave more like normal distribution
    for v in data_vars:
        new[v] = (new[v] - mean[v]) / std[v]
    
    return new

def get_mean(dataset):
    """
    Calculate mean of all data variables of dataset.

    Args:
        dataset (xarray.Dataset): Dataset that is used to calculate the mean.
    
    Returns:
        {string : float}: Dictionary of the mean values of all data variables.
    """

    mean = dict()
    data_vars = ["u", "v", "w", "temp", "salt", "zeta", "DIC", "PCO2OC"]

    for v in data_vars:
        mean[v] = np.nanmean(dataset[v])

    return mean

def get_std(dataset):
    """
    Calculate standard deviation of all data variables of dataset.

    Args:
        dataset (xarray.Dataset): Dataset that is used to calculate the standard deviation.
    
    Returns:
        {string : float}: Dictionary of the standard deviation of all data variables.
    """

    std = dict()
    data_vars = ["u", "v", "w", "temp", "salt", "zeta", "DIC", "PCO2OC"]

    for v in data_vars:
        std[v] = np.nanstd(dataset[v])

    return std
        
def tensor_to_dataset(tensor, mean, std, mask, eta_rho, xi_rho, lat_rho, lon_rho):
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

    tensor_data = tensor.detach().numpy()
    
    data_vars = ["u", "v", "w", "temp", "salt", "zeta", "DIC", "PCO2OC", "mask_rho"]
    data_array = []

    for t in tensor_data:
        data_array.append(
            xr.DataArray(
                data=t,
                dims=["eta_rho", "xi_rho"],
                coords={
                    "eta_rho": eta_rho,
                    "xi_rho": xi_rho,
                },
            )
        )

    data_array.append(mask)
    dataset = xr.Dataset(dict(zip(data_vars, data_array)))

    data_vars.remove("mask_rho")
    for v in data_vars:
        # reverse normalization and turn 0 back to nan for better plotting
        dataset[v] = (dataset[v] * std[v]) + mean[v]
        dataset[v] =  dataset[v].where(dataset["mask_rho"] == 1)
    
    dataset = dataset.assign_coords(
        lat_rho = lat_rho,
        lon_rho = lon_rho,
    )

    return dataset

def plot_dataset(dataset, vars=["temp"]):
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