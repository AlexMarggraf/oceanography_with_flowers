import xarray as xr
import xroms
import xcmocean
import os
import numpy as np
import pandas as np

pactcs30_data = []
relev_quant = ["u", "v", "w", "temp", "salt", "zeta", "NO3", "DIC", "PCO2OC", "TOT_CHL"]
statistics = []

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

for i in range(2000, 2020):
    pactcs30_data.append(open_pactcs30(i))

for i, data in enumerate(pactcs30_data):
    for quant in relev_quant:
        stat = dict()

        stat["year"] = 2000 + i
        stat["variable"] = quant
        stat["mean"] = np.nanmean(data[quant])
        stat["median"] = np.nanmedian(data[quant])
        stat["min"] = np.nanmin(data[quant])
        stat["max"] = np.nanmax(data[quant])
        stat["standard deviation"] = np.nanstd(data[quant])
        stat["Q1"] = np.nanpercentile(data[quant], 25)
        stat["Q3"] = np.nanpercentile(data[quant], 75)
        stat["Interquartile"] = stat["Q3"] - stat["Q1"]
    
        statistics.append(stat)

df = pd.DataFrame(statistics)
df.to_csv("../notebooks/statistics.csv", index=False)
