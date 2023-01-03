import requests
import os

from subprocess import call

from interpolate import interpolate_climate_model_data_to_ico_grid

import argparse


def download_file(url, filename):
    """
    Download a file from a given URL and save it under the provided filename.
    @param url:
    @param filename:
    @return:
    """
    # https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_required_files(output_directory="Datasets/"):
    filenames = {"ECHAM5": [
        "https://zenodo.org/record/6610684/files/ECHAM5-wiso_d18O_850-1849.nc",
        "https://zenodo.org/record/6610684/files/ECHAM5-wiso_prec_850-1849.nc",
        "https://zenodo.org/record/6610684/files/ECHAM5-wiso_tsurf_850-1849.nc"
    ],
        "GISS": [
            "https://zenodo.org/record/6610684/files/GISS-E2-R_d18O_850-1849.nc",
            "https://zenodo.org/record/6610684/files/GISS-E2-R_prec_850-1849.nc",
            "https://zenodo.org/record/6610684/files/GISS-E2-R_tsurf_850-1849.nc"
    ],

        "iCESM": [
            "https://zenodo.org/record/6610684/files/iCESM_d18O_850-1849.nc",
            "https://zenodo.org/record/6610684/files/iCESM_prec_850-1849.nc",
            "https://zenodo.org/record/6610684/files/iCESM_tsurf_850-1849.nc"
    ],
        "iHadCM3": [
            "https://zenodo.org/record/6610684/files/iHadCM3_d18O_850-1849.nc",
            "https://zenodo.org/record/6610684/files/iHadCM3_prec_850-1849.nc",
            "https://zenodo.org/record/6610684/files/iHadCM3_tsurf_850-1849.nc"
    ],
        "isoGSM": [
            "https://zenodo.org/record/6610684/files/isoGSM_d18O_850-1849.nc",
            "https://zenodo.org/record/6610684/files/isoGSM_prec_850-1849.nc",
            "https://zenodo.org/record/6610684/files/isoGSM_tsurf_850-1849.nc"]
    }
    for climate_model in filenames.keys():
        tmp_folder = os.path.join(output_directory, climate_model, "Original/")
        if os.path.exists(os.path.join(output_directory, climate_model)):
            print("Data for climate model {} seems to have been downloaded already (Folder exists).".format(
                climate_model))
        else:
            print("Downloading required data for {} climate model.".format(
                climate_model))
            os.makedirs(tmp_folder)
            os.makedirs(os.path.join(output_directory,
                        climate_model, "Interpolated/"))
            download_file(filenames[climate_model][0],
                          os.path.join(tmp_folder, "isotopes_raw.nc"))
            download_file(filenames[climate_model][1],
                          os.path.join(tmp_folder, "prec_raw.nc"))
            download_file(filenames[climate_model][2],
                          os.path.join(tmp_folder, "tsurf_raw.nc"))
    print("Done")


def preprocess_required_files(output_directory="Datasets/", low=-100, high=100, grid=None):
    """
    Apply preprocessing to the downloaded datasets.

    @param output_directory: Dataset we downloaded the files to.
    @param low: Lower limit for the d18O range.
    @param high: Upper limit for the d18O range.
    @param grid: Target grid we want to regrid to.
    """
    for m in ["ECHAM5", "GISS", "iCESM", "iHadCM3", "isoGSM"]:
        print("Preprocessing {} climate model data.".format(m))
        for f in ["isotopes_raw.nc", "prec_raw.nc", "tsurf_raw.nc"]:
            file = os.path.join(output_directory, m, "Original", f)
            script = os.path.join("./preprocess.sh")
            # if no grid is specified, use the iHadCM3 grid.
            if grid is None:
                call([script, "-f", file, "-u", str(high), "-l", str(low), "-g",
                     os.path.join(output_directory, "iHadCM3/Original/tsurf_raw.nc")])
            else:
                call([script, "-f", file, "-u", str(high),
                     "-l", str(low), "-g", grid])


def main(output_directory="Datasets/", low=-100, high=100, grid=None, create_ico=False, interpolation="cons1", resolution=5):
    download_required_files(output_directory=output_directory)
    preprocess_required_files(
        output_directory=output_directory, low=low, high=high, grid=grid)
    if create_ico:
        for cm in ["iCESM", "iHadCM3", "isoGSM", "GISS", "ECHAM5"]:
            for var_name in ["isotopes", "tsurf", "prec"]:
                print("Interpolating: ", cm, var_name)
                interpolate_climate_model_data_to_ico_grid(model_name=cm, variable_name=var_name, script_folder="Scripts/",
                                                           dataset_folder=output_directory, resolution=resolution, interpolation=interpolation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download and precprocess files required for reproducing results')
    parser.add_argument('-d', dest='directory', action='store',
                        default="Datasets/", required=False,
                        help='Directory to store the files in (default: ./Datasets)')
    parser.add_argument('-g', dest='grid', action='store', required=False,
                        help='Directory the grids are stored in (default: ./Datasets)')
    parser.add_argument('-dmin', dest='l', action='store',
                        default=-100, required=False,
                        help='Minimum of desired d18O range (default: -100)')
    parser.add_argument('-dmax', dest='h', action='store',
                        default=100, required=False,
                        help='Maximum of desired d18O range (default: 100)')
    parser.add_argument('--createico', dest='create_ico', action='store_true',
                        help='If flag set, interpolate to create files required for Icosahedral UNet.')
    parser.add_argument('-ico_resolution', dest='res', action='store',
                        default=5, required=False,
                        help='Resolution level of the icosahedral grid (default: 5)')
    parser.add_argument('-ico_interpolation', dest='interpolation', action='store',
                        default="cons1", required=False,
                        help='Interpolation type for interpolating to icosahedral grid (default: cons1)')
    args = parser.parse_args()
    main(output_directory=args.directory, low=args.l, high=args.h, grid=args.grid,
         create_ico=args.create_ico, interpolation=args.interpolation, resolution=args.res)
