"""
Configuration for BCSD processing pipelines.

This module provides helpers to:
- Detect whether the code is running in a local testing environment or on the BIG server.
- Construct consistent file paths for BCSD inputs, model outputs, and processed results
  across different environments and SSP scenarios.

Key components:
- TIME_SLICE: Defines the time range used in processing (2015-01-01 to 2080-12-31).
- is_local_testing(): Determines environment based on filesystem availability.
- build_paths(): Generates input/output paths for BCSD and model datasets.

The goal is to centralize environment-specific logic and path construction so that
other modules (e.g., regridding, merging, scoring) can remain environment-agnostic.
"""

from pathlib import Path

TIME_SLICE = slice(0, 24090)  # 2015-01-01 to 2080-12-31


def is_local_testing() -> bool:
    """
    Determines if the current environment is for local testing or BIG.

    Returns:
        bool: True if the environment is for local testing; False otherwise.
    """
    return not Path("/lfs/archive/Reanalysis/").exists()


def build_paths(ssp: str, w1: str) -> tuple[list[str], str, str]:
    """
    Build paths for BCSD input files, grouped input file, and output file.

    Args:
        ssp (str): SSP scenario (e.g., "ssp126").
        w1 (str): W1 identifier (e.g., "W1-1").

    Returns:
        tuple[list[str], str, str]: Tuple containing:
            - List of BCSD input file paths (pr and tas).
            - Grouped input file path.
            - Output file path.
    """
    if is_local_testing():
        bcsd_nc = [
            "../../data/bcsd/ssp126/pr_QDM_TaiESM1.nc",
            "../../data/bcsd/ssp126/tas_QDM_TaiESM1.nc",
        ]
        input_nc = "../../data/bcsd/W1-1/output_0_reg_masked.nc"
        out_nc = "./bcsd_masked.nc"
    else:
        bcsd_nc = [
            f"/lfs/archive/TCCIP_data/CMIP6_QDM/pr/{ssp}/TaiESM1/r1i1p1f1/pr_QDM_TaiESM1.nc",
            f"/lfs/archive/TCCIP_data/CMIP6_QDM/tas/{ssp}/TaiESM1/r1i1p1f1/tas_QDM_TaiESM1.nc",
        ]
        input_nc = f"~/SSP_result/W1/{w1}/netcdf/output_0_reg_masked.nc"
        out_nc = f"./B-{w1.split('-')[1]}/bcsd_masked.nc"

    return bcsd_nc, input_nc, out_nc
