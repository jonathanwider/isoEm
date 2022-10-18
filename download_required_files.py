import requests
import os


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
            print("Data for climate model {} seems to have been downloaded already (Folder exists).".format(climate_model))
        else:
            print("Downloading required data for {} climate model.".format(climate_model))
            os.makedirs(tmp_folder)
            download_file(filenames[climate_model][0], os.path.join(tmp_folder, "isotopes.nc"))
            download_file(filenames[climate_model][1], os.path.join(tmp_folder, "temp.nc"))
            download_file(filenames[climate_model][2], os.path.join(tmp_folder, "precip.nc"))