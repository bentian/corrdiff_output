import sys
from pathlib import Path
import xarray as xr

LANDMASK_NC = "./data/wrf_208x208_grid_coords.nc"  # Path to the landmask NetCDF file

def apply_landmask(truth, pred):
    """
    Apply landmask to truth and prediction datasets.

    Parameters:
        truth (xarray.Dataset): The truth dataset.
        pred (xarray.Dataset): The prediction dataset.

    Returns:
        tuple: Masked truth and pred datasets.
    """
    grid = xr.open_dataset(LANDMASK_NC, engine='netcdf4')
    landmask = grid.LANDMASK.rename({"south_north": "y", "west_east": "x"})

    # Expand landmask dimensions to match truth and pred
    landmask_expanded = landmask.expand_dims(dim={"time": truth.sizes["time"]})
    truth = truth.where(landmask_expanded == 1, 0)

    if "ensemble" in pred.dims:
        landmask_expanded = landmask_expanded.expand_dims(dim={"ensemble": pred.sizes["ensemble"]})
    pred = pred.where(landmask_expanded == 1, 0)

    return truth, pred

def save_masked_samples(input_file, output_file):
    """
    Open prediction and truth samples from a dataset file, apply the landmask, 
    and save them back to the file.

    Parameters:
        f (str): Path to the dataset file.
        masked (bool): Whether to apply the landmask.
    """
    with xr.open_dataset(input_file, engine="netcdf4", mode="r") as ds:
        # Extract time coordinate to preserve it
        time_coord = ds.coords["time"]

        with xr.open_dataset(input_file, group="truth", engine="netcdf4") as truth_ds, \
             xr.open_dataset(input_file, group="prediction", engine="netcdf4") as pred_ds:
            # Apply landmask
            masked_truth, masked_pred = apply_landmask(truth_ds, pred_ds)

            new_ds = xr.Dataset(coords={"time": time_coord})  # Preserves time

            # Copy non-masked root group variables from original dataset
            for var in ds.data_vars:
                new_ds[var] = ds[var]

            # Save the modified dataset with the same structure
            new_ds.to_netcdf(output_file, mode="w")  # Write root dataset
            masked_truth.to_netcdf(output_file, mode="a", group="truth")  # Append to 'truth' group
            masked_pred.to_netcdf(output_file, mode="a", group="prediction")  # Append to 'prediction' group

    print(f"Masked dataset saved to {output_file}")       
