

from evaluate import get_rescaled_predictions_and_gt
import netCDF4 as nc
import numpy as np
import os
from icosahedron import Icosahedron

from subprocess import call


def interpolate_predictions(descriptions, predictions, script_folder="Scripts/"):
    """
    Provide functions to interpolate between grids.
    Ideally the function would proceed in the following steps:
    1) load predictions, undo the standardization
    2) create netcdf4 (temporary)
    3) do the interpolation by calling the script files
    4) create a gz file from the interpolated file or alternatively, write a loader that can handle .nc files.
    @param descriptions:
    @param predictions:
    @param script_folder: Folder in which the shell scripts to do the cdo interpolations with are stored in. This
    is the place where we need to store files to be interpolated.
    @return:
    """
    assert len(descriptions["DATASET_DESCRIPTION"]["TARGET_VARIABLES"].keys()) == 1
    # load the predictions, undo the scaling
    rescaled_predictions, _ = get_rescaled_predictions_and_gt(descriptions, predictions)

    netcdf_from_rescaled_predictions(descriptions, rescaled_predictions,
                                     descriptions["DATASET_DESCRIPTION"]["TIMESTEPS_TEST"], script_folder)
    run_script(descriptions, script_folder)


def netcdf_from_rescaled_predictions(descriptions, rescaled_predictions, t_test, output_folder):
    dataset_description = descriptions["DATASET_DESCRIPTION"]
    model_training_description = descriptions["MODEL_TRAINING_DESCRIPTION"]
    assert len(dataset_description[
                   "TARGET_VARIABLES"].keys()) == 1, "Interpolation only implemented for single target variable"

    if dataset_description["GRID_TYPE"] == "Flat":
        tocopy = ['longitude', 'latitude', ]
        dimscopy = ['t', 'bnds', 'longitude', 'latitude']
        output_file = os.path.join(output_folder, "tmp.nc")

        if dataset_description["TIMESCALE"] == "YEARLY":
            filename = os.path.join("Datasets", dataset_description["CLIMATE_MODEL"], "Original",
                                    "{}.nc".format(list(dataset_description["TARGET_VARIABLES"].keys())[0]))
        elif dataset_description["TIMESCALE"] == "MONTHLY":
            filename = os.path.join("Datasets", dataset_description["CLIMATE_MODEL"], "Original",
                                    "{}_monthly.nc".format(list(dataset_description["TARGET_VARIABLES"].keys())[0]))
        else:
            raise NotImplementedError("Invalid timescale")
        original_dimensions = nc.Dataset(filename).variables[
            "{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0])].dimensions
        necessary_dimensions = (original_dimensions[0], original_dimensions[2], original_dimensions[3])
        original_dataype = nc.Dataset(filename).variables["{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0])].datatype

        src = nc.Dataset(filename)
        dst = nc.Dataset(output_file, "w")
        dst.setncatts(src.__dict__)

        # copy dimensions
        for name, dimension in src.dimensions.items():
            if name in dimscopy:
                dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
        # copy all file data except for the excluded
        for name, variable in src.variables.items():
            if name in tocopy:
                x = dst.createVariable(name, variable.datatype, variable.dimensions)
                dst[name][:] = src[name][:]
                # copy variable attributes all at once via dictionary
                dst[name].setncatts(src[name].__dict__)
        target_var_attribute_dict = nc.Dataset(filename).variables[
            "{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0])].__dict__
        dst.createVariable(
            "{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0]), original_dataype, necessary_dimensions)
        dst.variables["{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0])].setncatts(target_var_attribute_dict)
        dst.createVariable("t", "float64", "t")
        dst.createVariable("t_bnds", "float64", ("t", "bnds"))
        dst.variables["t"][:] = t_test

        # extract t_bnds from source file
        src_t = src.variables["t"][:].data
        t_bnds = []
        for t_c in t_test:
            t_bnds.append([])
            i = np.where(src_t == t_c)[0][0]
            t_bnds[-1] = src.variables["t_bnds"][i].data
        dst.variables["t_bnds"][:] = np.array(t_bnds)
        # pad the numpy by the amount the we trimmed of when loading the data
        tmp = np.pad(np.squeeze(rescaled_predictions), (
            (0, 0), (dataset_description["LATITUDES_SLICE"][0], -dataset_description["LATITUDES_SLICE"][1]), (0, 0)),
                     'constant',
                     constant_values=target_var_attribute_dict["missing_value"])
        dst.variables["dO18"][:] = tmp
        dst.close()
        src.close()

    elif dataset_description["GRID_TYPE"] == "Ico":
        assert dataset_description["TIMESCALE"] == "YEARLY"
        tocopy = ['lon', 'lon_bnds', 'lat', 'lat_bnds']
        dimscopy = ['t', 'bnds', 'ncells', 'vertices']
        output_file_5_nbs = os.path.join(output_folder, "tmp_5_nbs.nc")
        output_file_6_nbs = os.path.join(output_folder, "tmp_6_nbs.nc")

        filename_5_nbs = os.path.join("Datasets", dataset_description["CLIMATE_MODEL"],
                                      "Interpolated",
                                      "{}_r_{}_nbs_5_{}.nc".format(
                                          list(dataset_description["TARGET_VARIABLES"].keys())[0],
                                          dataset_description["RESOLUTION"],
                                          dataset_description["INTERPOLATION"]))
        filename_6_nbs = os.path.join("Datasets", dataset_description["CLIMATE_MODEL"],
                                      "Interpolated",
                                      "{}_r_{}_nbs_6_{}.nc".format(
                                          list(dataset_description["TARGET_VARIABLES"].keys())[0],
                                          dataset_description["RESOLUTION"],
                                          dataset_description["INTERPOLATION"]))

        original_dimensions_5_nbs = nc.Dataset(filename_5_nbs).variables[
            "{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0])].dimensions
        original_dimensions_6_nbs = nc.Dataset(filename_6_nbs).variables[
            "{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0])].dimensions
        necessary_dimensions_5_nbs = (original_dimensions_5_nbs[0], original_dimensions_5_nbs[2])
        necessary_dimensions_6_nbs = (original_dimensions_6_nbs[0], original_dimensions_6_nbs[2])
        original_dataype_5_nbs = nc.Dataset(filename_5_nbs).variables[
            "{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0])].datatype
        original_dataype_6_nbs = nc.Dataset(filename_6_nbs).variables[
            "{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0])].datatype

        src_5_nbs = nc.Dataset(filename_5_nbs)
        src_6_nbs = nc.Dataset(filename_6_nbs)
        dst_5_nbs = nc.Dataset(output_file_5_nbs, "w")
        dst_6_nbs = nc.Dataset(output_file_6_nbs, "w")
        dst_5_nbs.setncatts(src_5_nbs.__dict__)
        dst_6_nbs.setncatts(src_6_nbs.__dict__)

        for name, dimension in src_5_nbs.dimensions.items():
            if name in dimscopy:
                dst_5_nbs.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
        for name, dimension in src_6_nbs.dimensions.items():
            if name in dimscopy:
                dst_6_nbs.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

        # copy all file data except for the excluded
        for name, variable in src_5_nbs.variables.items():
            if name in tocopy:
                x = dst_5_nbs.createVariable(name, variable.datatype, variable.dimensions)
                dst_5_nbs[name][:] = src_5_nbs[name][:]
                # copy variable attributes all at once via dictionary
                dst_5_nbs[name].setncatts(src_5_nbs[name].__dict__)
        for name, variable in src_6_nbs.variables.items():
            if name in tocopy:
                x = dst_6_nbs.createVariable(name, variable.datatype, variable.dimensions)
                dst_6_nbs[name][:] = src_6_nbs[name][:]
                # copy variable attributes all at once via dictionary
                dst_6_nbs[name].setncatts(src_6_nbs[name].__dict__)
        target_var_attribute_dict_6_nbs = nc.Dataset(filename_6_nbs).variables[
            "{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0])].__dict__
        dst_6_nbs.createVariable(
            "{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0]), original_dataype_6_nbs, necessary_dimensions_6_nbs)
        dst_6_nbs.variables["{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0])].setncatts(target_var_attribute_dict_6_nbs)
        dst_6_nbs.createVariable("t", "float64", "t")
        dst_6_nbs.createVariable("t_bnds", "float64", ("t", "bnds"))
        dst_6_nbs.variables["t"][:] = t_test

        target_var_attribute_dict_5_nbs = nc.Dataset(filename_5_nbs).variables[
            "{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0])].__dict__
        dst_5_nbs.createVariable(
            "{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0]), original_dataype_5_nbs, necessary_dimensions_5_nbs)
        dst_5_nbs.variables["{}".format(list(dataset_description["TARGET_VARIABLES"].values())[0][0])].setncatts(target_var_attribute_dict_5_nbs)
        dst_5_nbs.createVariable("t", "float64", "t")
        dst_5_nbs.createVariable("t_bnds", "float64", ("t", "bnds"))
        dst_5_nbs.variables["t"][:] = t_test

        rescaled_predictions_5_nbs, rescaled_predictions_6_nbs = split_5_nbs_6_nbs(rescaled_predictions, dataset_description)

        dst_5_nbs.variables["dO18"][:] = rescaled_predictions_5_nbs
        dst_5_nbs.close()
        src_5_nbs.close()

        dst_5_nbs.variables["dO18"][:] = rescaled_predictions_5_nbs
        dst_5_nbs.close()
        src_5_nbs.close()
    else:
        raise NotImplementedError("Invalid grid type")


def split_5_nbs_6_nbs(rescaled_predictions, dataset_description):
    assert dataset_description["GRID_TYPE"] == "Ico"
    ico = Icosahedron(r=dataset_description["RESOLUTION"])
    regions, vertices = ico.get_voronoi_regions_vertices()
    charts = ico.get_charts_cut()
    indices_six_nb = []
    indices_five_nb = []
    for i in range(len(regions)):
        if len(regions[i]) > 5:
            indices_six_nb.append(i)
        else:
            indices_five_nb.append(i)
    # create numpy arrays
    indices_6_nbs = np.array(indices_six_nb)
    indices_5_nbs = np.array(indices_five_nb)
    rescaled_predictions_5_nbs = np.squeeze(rescaled_predictions[..., indices_5_nbs])
    rescaled_predictions_6_nbs = np.squeeze(rescaled_predictions[..., indices_6_nbs])
    return rescaled_predictions_5_nbs, rescaled_predictions_6_nbs


def run_script(descriptions, script_folder, interpolation_type="cons1", resolution=5):
    """

    @param descriptions: Descriptions of dataset and (model and training)
    @param script_folder: Should contain the two scripts: ico_to_model.sh and model_to_ico.sh as
    well as grid description files for the icosahedral grid and model grids.
    @return:
    """
    if descriptions["DATASET_DESCRIPTION"]["GRID_TYPE"] == "Flat":
        script = os.path.join(script_folder, "model_to_ico.sh")
        print(os.listdir(script_folder))
        print(script_folder, script)
        files = os.path.join(script_folder, "tmp.nc")
        i_arg = interpolation_type
        f_arg = files
        g_arg = "grid_description_r_{}_nbs_6_ico.txt grid_description_r_{}_nbs_5_ico.txt".format(resolution, resolution)
        call([script, "-f", f_arg, "-g", g_arg, "-i", i_arg])

    elif descriptions["DATASET_DESCRIPTION"]["GRID_TYPE"] == "Ico":
        files_5_nb = "tmp_5_nbs"
        files_6_nb = "tmp_6_nbs"
        script = os.path.join(script_folder, "ico_to_model.sh")
        print(script_folder, script)
        o_arg = "default_grid.txt"
        i_arg = interpolation_type

        g_arg = "grid_description_r_{}_nbs_6_ico.txt".format(descriptions["DATASET_DESCRIPTION"]["RESOLUTION"])
        call([script, "-f", files_6_nb, "-g", g_arg, "-o", o_arg, "-i", i_arg])

        g_arg = "grid_description_r_{}_nbs_5_ico.txt".format(descriptions["DATASET_DESCRIPTION"]["RESOLUTION"])
        call([script, "-f", files_5_nb, "-g", g_arg, "-o", o_arg, "-i", i_arg])

    else:
        raise NotImplementedError("Invalid grid type")
