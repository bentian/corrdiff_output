from pathlib import Path

TIME_SLICE = slice(0, 365)


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
            "./ssp126/pr_QDM_TaiESM1.nc",
            "./ssp126/tas_QDM_TaiESM1.nc",
        ]
        input_nc = "./W1-1/output_0_reg_masked.nc"
        out_nc = "./bcsd_masked.nc"
    else:
        bcsd_nc = [
            f"/lfs/archive/TCCIP_data/CMIP6_QDM/pr/{ssp}/TaiESM1/r1i1p1f1/pr_QDM_TaiESM1.nc",
            f"/lfs/archive/TCCIP_data/CMIP6_QDM/tas/{ssp}/TaiESM1/r1i1p1f1/tas_QDM_TaiESM1.nc",
        ]
        input_nc = f"../SSP_result/W1/{w1}/netcdf/output_0_reg_masked.nc"
        out_nc = f"./B-{w1.split('-')[1]}/netcdf/bcsd_masked.nc"

    return bcsd_nc, input_nc, out_nc
