from pathlib import Path
import xarray as xr
from typing import Tuple

LANDMASK_NC = "./data/wrf_208x208_grid_coords.nc"  # Path to the landmask NetCDF file

def apply_landmask(truth: xr.Dataset, pred: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Apply landmask to truth and prediction datasets.

    Parameters:
        truth (xr.Dataset): The truth dataset.
        pred (xr.Dataset): The prediction dataset.

    Returns:
        Tuple[xr.Dataset, xr.Dataset]: Masked truth and prediction datasets.
    """
    grid = xr.open_dataset(LANDMASK_NC, engine='netcdf4')
    landmask = grid.LANDMASK.rename({"south_north": "y", "west_east": "x"})

    # Expand landmask dimensions to match truth and pred
    landmask_expanded = landmask.expand_dims(dim={"time": truth.sizes["time"]})
    truth_masked = truth.where(landmask_expanded == 1, 0)

    if "ensemble" in pred.dims:
        landmask_expanded = landmask_expanded.expand_dims(dim={"ensemble": pred.sizes["ensemble"]})
    pred_masked = pred.where(landmask_expanded == 1, 0)

    return truth_masked, pred_masked

def save_masked_samples(input_file: Path, output_file: Path) -> None:
    """
    Open prediction and truth samples from a dataset file, apply the landmask, 
    and save them back to a new NetCDF file while preserving metadata.

    Parameters:
        input_file (Path): Path to the input dataset file.
        output_file (Path): Path to save the output dataset file.
    """
    with xr.open_dataset(input_file, engine="netcdf4", mode="r") as ds:
        # Extract time coordinate to preserve it
        time_coord = ds.coords["time"]

        with xr.open_dataset(input_file, group="truth", engine="netcdf4") as truth_ds, \
             xr.open_dataset(input_file, group="prediction", engine="netcdf4") as pred_ds:
            # Apply landmask
            truth_masked, pred_masked = apply_landmask(truth_ds, pred_ds)

            new_ds = xr.Dataset(coords={"time": time_coord})  # Preserves time

            # Copy non-masked root group variables from original dataset
            for var in ds.data_vars:
                new_ds[var] = ds[var]

            # Save the modified dataset with the same structure
            new_ds.to_netcdf(output_file, mode="w")
            truth_masked.to_netcdf(output_file, mode="a", group="truth")
            pred_masked.to_netcdf(output_file, mode="a", group="prediction")

    print(f"Masked dataset saved to {output_file}")       
