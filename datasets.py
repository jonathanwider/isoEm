# Notebooks containing code to create datasets we can do training on from raw climate data.

import os.path
import netCDF4

import gzip
import pickle

import numpy as np

import icosahedron
import util


def get_required_datasets(description, dataset_folder):
    """
    Compose the paths of all the files that are required to create the model data set.
    """

    assert "DATASETS_USED" in description.keys()
    assert "CLIMATE_MODEL" in description.keys()
    assert "GRID_TYPE" in description.keys()
    assert "TIMESCALE" in description.keys()
    if description["TIMESCALE"] == "MONTHLY":
        if description["GRID_TYPE"] == "Ico":
            raise NotImplementedError("Monthly timescale was only investigated on Flat grid.")

    datasets = {}

    if description["GRID_TYPE"] == "Ico":
        assert "RESOLUTION" in description.keys()
        assert "INTERPOLATE_CORNERS" in description.keys()
        assert "INTERPOLATION" in description.keys()

    for dst in description["DATASETS_USED"]:
        if description["GRID_TYPE"] == "Flat":
            if description["TIMESCALE"] == "YEARLY":
                d_path = os.path.join(dataset_folder, description["CLIMATE_MODEL"], "Original", "{}_yearly.nc".format(dst))
                datasets[dst] = netCDF4.Dataset(d_path, "a")
            elif description["TIMESCALE"] == "MONTHLY":
                d_path = os.path.join(dataset_folder, description["CLIMATE_MODEL"], "Original", "{}.nc".format(dst))
                datasets[dst] = netCDF4.Dataset(d_path, "a")
            else:
                raise NotImplementedError("Only MONTHLY and YEARLY timescales implemented.")
        elif description["GRID_TYPE"] == "Ico":
            if description["INTERPOLATE_CORNERS"]:
                name6 = os.path.join(dataset_folder, description["CLIMATE_MODEL"], "Interpolated",
                                     "{}_yearly_r_{}_nbs_6_{}.nc".format(dst, description["RESOLUTION"], description["INTERPOLATION"]))
                name5 = os.path.join(dataset_folder, description["CLIMATE_MODEL"], "Interpolated",
                                     "{}_yearly_r_{}_nbs_5_{}.nc".format(dst, description["RESOLUTION"], description["INTERPOLATION"]))
                datasets[dst] = {}
                datasets[dst]["6_nb"] = netCDF4.Dataset(name6, "a")
                datasets[dst]["5_nb"] = netCDF4.Dataset(name5, "a")
            else:
                name6 = os.path.join(dataset_folder, description["CLIMATE_MODEL"], "Interpolated",
                                     "{}_yearly_r_{}_nbs_6_{}.nc".format(dst, description["RESOLUTION"], description["INTERPOLATION"]))
                datasets[dst] = {}
                datasets[dst]["6_nb"] = netCDF4.Dataset(name6, "a")
        else:
            raise NotImplementedError("Only Ico and Flat grids were implemented.")
    return datasets


def get_available_variables(description, basefolder):
    """
    Of all the raw datasets selected in description["DATASETS_USED"], display details on available variables.
    """
    assert "GRID_TYPE" in description.keys()

    datasets = get_required_datasets(description, basefolder)
    if description["GRID_TYPE"] == "Flat":
        for d_name, dset in datasets.items():
            print(d_name)
            for name, var in dset.variables.items():
                print(f'\tName: {var.name}')
                if hasattr(var, 'long_name'):
                    print(f'\tLong name: {var.long_name}')
                print(f'\tDimensions: {var.dimensions}')
                if hasattr(var, 'units'):
                    print(f'\tUnits: {var.units}')
                print('\n')
    elif description["GRID_TYPE"] == "Ico":
        for d_name, dset in datasets.items():
            print(d_name)
            for name, var in dset["6_nb"].variables.items():
                print()
                print(f'\tName: {var.name}')
                if hasattr(var, 'long_name'):
                    print(f'\tLong name: {var.long_name}')
                print(f'\tDimensions: {var.dimensions}')
                if hasattr(var, 'units'):
                    print(f'\tUnits: {var.units}')
                print('\n')
    else:
        raise NotImplementedError("Only Ico and Flat grids were implemented.")


def get_shared_timesteps(description, dataset_folder):
    """
    load the datasets for which we require that variables are present at each timestep.
    Then extract the shared timesteps.
    Timesteps are considered shared if year and month match in between all necessary files.
    Works with missing timesteps problems might occur if timesteps occur twice.
    """
    from functools import reduce

    assert "CLIMATE_MODEL" in description.keys()
    assert "DATASETS_NO_GAPS" in description.keys()

    datasets = {}

    if description["TIMESCALE"] == "YEARLY":
        for dst in description["DATASETS_NO_GAPS"]:
            d_path = os.path.join(dataset_folder, description["CLIMATE_MODEL"], "Original", "{}_yearly.nc".format(dst))
            datasets[dst] = netCDF4.Dataset(d_path, "a")
    elif description["TIMESCALE"] == "MONTHLY":
        for dst in description["DATASETS_NO_GAPS"]:
            d_path = os.path.join(dataset_folder, description["CLIMATE_MODEL"], "Original", "{}.nc".format(dst))
            datasets[dst] = netCDF4.Dataset(d_path, "a")
    # assume that the time variable always has the same name for a given climate model... otherwise would need to
    # rewrite next lines
    try:
        ts = tuple([dset.variables["t"][:].data for dset in datasets.values()])
        units = tuple([dset.variables["t"].units for dset in datasets.values()])
        cals = tuple([dset.variables["t"].calendar for dset in datasets.values()])
    except KeyError:
        ts = tuple([dset.variables["time"][:].data for dset in datasets.values()])
        units = tuple([dset.variables["time"].units for dset in datasets.values()])
        cals = tuple([dset.variables["time"].calendar for dset in datasets.values()])

    if description["TIMESCALE"] == "YEARLY":
        ys = [set(util.get_years_months(t, units[i], cals[i])[0]) for (i, t) in enumerate(ts)]
        common_dates = set.intersection(*ys)
    elif description["TIMESCALE"] == "MONTHLY":
        years_l = [util.get_years_months(t, units[i], cals[i])[0] for (i, t) in enumerate(ts)]
        months_l = [util.get_years_months(t, units[i], cals[i])[1] for (i, t) in enumerate(ts)]
        c_dates = []
        for i in range(len(years_l)):
            dates = set()
            for j in range(len(years_l[i])):
                dates.add((years_l[i][j], months_l[i][j]))
            c_dates.append(dates)
        common_dates = set.intersection(*c_dates)
    return list(common_dates)


def load_variables_and_timesteps(description, dataset_folder):
    """import variables, reshape and remove the points at north and south pole"""
    assert "START_YEAR" in description.keys()
    assert "END_YEAR" in description.keys()
    assert "PREDICTOR_VARIABLES" in description.keys()
    assert "TARGET_VARIABLES" in description.keys()
    assert "LATITUDES_SLICE" in description.keys()
    assert "TIMESCALE" in description.keys()
    assert description["PRECIP_WEIGHTING"] is False
    assert description["TIMESCALE"] == "YEARLY"

    variables = {}
    masks = {}
    datasets = get_required_datasets(description, dataset_folder)
    # make sure that all datasets that have a non-trivial time axis share the same calendar and units.
    if description["GRID_TYPE"] == "Flat":
        description["T_UNITS"], description["CALENDAR"] = util.load_units_cals(description, list(datasets.values()))
    elif description["GRID_TYPE"] == "Ico":
        ds_6nb = [ds["6_nb"] for ds in list(datasets.values())]
        description["T_UNITS"], description["CALENDAR"] = util.load_units_cals(description, list(ds_6nb))
    else:
        raise NotImplementedError("Invalid grid type")

    c_years = np.array(get_shared_timesteps(description, dataset_folder))
    c_mask = np.logical_and(c_years >= description["START_YEAR"],
                            c_years < description["END_YEAR"])
    c_dates = c_years[c_mask]

    for dataset_name, dataset in datasets.items():  # loop over all used datasets
        variables[dataset_name] = {}
        masks[dataset_name] = {}
        # loop over all variables we want to use from this dataset
        if description["GRID_TYPE"] == "Flat":
            try:
                years = util.get_years_months(dataset.variables["t"][:].data, dataset.variables["t"].units,
                                              dataset.variables["t"].calendar)[0]
            except KeyError:
                years = util.get_years_months(dataset.variables["time"][:].data, dataset.variables["time"].units,
                                              dataset.variables["time"].calendar)[0]
            # get the corresponding indices:
            indices = []
            if years is not None:
                for i, t in enumerate(years):
                    if t in c_dates:
                        indices.append(i)
                indices = np.array(indices, dtype=int)
        elif description["GRID_TYPE"] == "Ico":
            try:
                years = util.get_years_months(dataset["6_nb"].variables["t"][:].data,
                                              dataset["6_nb"].variables["t"].units,
                                              dataset["6_nb"].variables["t"].calendar)[0]
            except KeyError:
                years = util.get_years_months(dataset["6_nb"].variables["time"][:].data,
                                              dataset["6_nb"].variables["time"].units,
                                              dataset["6_nb"].variables["time"].calendar)[0]
            # get the corresponding indices:
            indices = []
            if years is not None:
                for i, t in enumerate(years):
                    if t in c_dates:
                        indices.append(i)
                indices = np.array(indices, dtype=int)
        else:
            raise NotImplementedError("Invalid grid type")

        for variable_name in dict(description["PREDICTOR_VARIABLES"], **description["TARGET_VARIABLES"])[dataset_name]:
            if description["GRID_TYPE"] == "Flat":
                description["LATITUDES"], description["LONGITUDES"] = util.load_longitudes_latitudes(description, dataset)

                if dataset.variables[variable_name][:].data.shape[0] > 1:  # only if time dimension is not trivial
                    variables[dataset_name][variable_name] = np.squeeze(dataset.variables[variable_name][:])[indices,
                                                             description["LATITUDES_SLICE"][0]:description["LATITUDES_SLICE"][1], :]
                else:
                    variables[dataset_name][variable_name] = np.squeeze(
                        np.repeat(dataset.variables[variable_name][:][..., 1:-1, :], repeats=len(c_dates), axis=0))

                # load the marker for the missing value in the dataset
                try:
                    missing_value = dataset.variables[variable_name].missing_value
                    variables[dataset_name][variable_name].data[variables[dataset_name][variable_name].data == missing_value] = np.nan
                except AttributeError:
                    pass
                # if there are missing values in any of the extracted data for this var, store the masks too.
                if dataset_name in description["TARGET_VARIABLES"].keys():
                    masks[dataset_name][variable_name] = np.ma.getmaskarray(variables[dataset_name][variable_name])

            elif description["GRID_TYPE"] == "Ico":
                variables[dataset_name][variable_name] = {}
                for subdataset_name, subdataset in dataset.items():  # loop over subfiles: containing points with 5 and 6 nbs
                    if subdataset.variables[variable_name][:].shape[0] > 1:  # only if time dimenion is not trivial
                        variables[dataset_name][variable_name][subdataset_name] = np.squeeze(
                            subdataset.variables[variable_name][:])[indices, :]
                    else:
                        variables[dataset_name][variable_name][subdataset_name] = np.squeeze(
                            np.repeat(subdataset.variables[variable_name][:], repeats=len(c_dates), axis=0))
                    try:
                        missing_value = dataset.variables[variable_name][subdataset_name].missing_value
                        variables[dataset_name][variable_name][subdataset_name].data[variables[dataset_name][variable_name][subdataset_name].data == missing_value] = np.nan
                    except AttributeError:
                        pass
                    print(variable_name)
                    print("contains nans", (variables[dataset_name][variable_name][subdataset_name] == np.nan).any())
                    print("mask non-zero:", (variables[dataset_name][variable_name][subdataset_name].mask == True).any())                    
                variables[dataset_name][variable_name] = combine_variables(description, variables[dataset_name][variable_name])
            else:
                raise NotImplementedError("Only Ico and Flat grids implemented")

    res_masks = {}
    res_variables = {}

    for dataset_name, dataset in variables.items():  # loop over all used datasets
        v = dict(description["PREDICTOR_VARIABLES"], **description["TARGET_VARIABLES"])
        for variable_name in v[dataset_name]:  # loop over all variables we want to use from this dataset
            res_variables[variable_name] = dataset[variable_name]
            if description["GRID_TYPE"] == "Flat":
                res_variables[variable_name] = res_variables[variable_name].reshape(res_variables[variable_name].shape[0], 1,
                                                                                    res_variables[variable_name].shape[-2],
                                                                                    res_variables[variable_name].shape[-1])
            elif description["GRID_TYPE"] == "Ico":
                res_variables[variable_name] = res_variables[variable_name].reshape(res_variables[variable_name].shape[0], 1,
                                                                                    res_variables[variable_name].shape[-1])
            else:
                raise NotImplementedError("Only Ico and Flat grids implemented")

    # exclude timesteps with missing values in predictor variables.
    masked_timesteps = np.zeros(list(res_variables.values())[0].shape[0], dtype=bool)
    for vs in description["PREDICTOR_VARIABLES"].values():
        for v in vs:
            if (np.ma.getmaskarray(res_variables[v]) != False).any():
                m = np.where(np.mean(np.ma.getmaskarray(res_variables[v]), axis=(-1, -2, -3)) > 0)[0]
                masked_timesteps[m] = True
    print("exclude {} timestep".format(np.where(masked_timesteps==True)[0]))
    c_dates = c_dates[~masked_timesteps]

    for dataset_name, dataset_masks in masks.items():  # loop over all used datasets
        v = dict(description["PREDICTOR_VARIABLES"], **description["TARGET_VARIABLES"])
        for variable_name in v[dataset_name]:  # loop over all variables we want to use from this dataset
            if variable_name in dataset_masks.keys():
                res_masks[variable_name] = dataset_masks[variable_name][~masked_timesteps, ...]

    for key, v in res_variables.items():
        res_variables[key] = v[~masked_timesteps, ...].data
    return res_variables, res_masks, c_dates


def combine_variables(description, dataset_dict):
    """
    For the icosahedral grid we need to combine the corner pixels (5nbs) and the other pixels (6nb),
    that were interpolated separately into one variable.
    """
    assert description["GRID_TYPE"] == "Ico"
    assert "6_nb" in dataset_dict.keys()
    # figure out the indices of the corner and non-corner points in the format we require later.
    ico = icosahedron.Icosahedron(r=description["RESOLUTION"])
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
    indices_six_nb = np.array(indices_six_nb)
    indices_five_nb = np.array(indices_five_nb)

    # there are 12 corner pixels, North and south pole are not included in charts and thus don't need to be added
    combined_data = np.zeros(dataset_dict["6_nb"].shape[:-1] + (dataset_dict["6_nb"].shape[-1] + 10,))
    if "5_nb" in dataset_dict.keys():
        combined_data[:, indices_six_nb] = dataset_dict["6_nb"]
        combined_data[:, indices_five_nb] = dataset_dict["5_nb"]
    else:
        combined_data[:, indices_six_nb] = dataset_dict["6_nb"]
        combined_data[:, indices_five_nb] = 0
    return combined_data


def create_yearly_dataset(description, dataset_folder, output_folder):
    """
    create a dataset that we can do ML on from climate model output file in folder dataset_folder and given a dict of
    decisions in the creation process (description).
    """
    assert "DO_SHUFFLE" in description.keys()
    assert np.logical_xor("TEST_FRACTION" in description.keys(), "SPLIT_YEAR" in description.keys())
    assert description["TIMESCALE"] == "YEARLY"

    print("loading variables")
    # load the selected climate variables.
    variables, masks, c_dates = load_variables_and_timesteps(description, dataset_folder)

    # split the variables into predictors and targets.
    pvars = util.flatten(description["PREDICTOR_VARIABLES"].values())
    tvars = util.flatten(description["TARGET_VARIABLES"].values())
    predictors = np.concatenate(tuple([variables[p_var] for p_var in pvars]), axis=1)
    targets = np.concatenate(tuple([variables[t_var] for t_var in tvars]), axis=1)

    if description["GRID_TYPE"] == "Flat":
        masks = np.concatenate(tuple([masks[t_var] for t_var in tvars], ))

    indices = np.arange(predictors.shape[0])
    # if we want to reload a dataset with a specific configuration of indices for training and testing
    if "INDICES_TEST" in description.keys():
        x_train, x_test = util.train_test_split_by_indices(description, predictors)
        y_train, y_test = util.train_test_split_by_indices(description, targets)
        indices_train, indices_test = util.train_test_split_by_indices(description, indices)
        timesteps_train, timesteps_test = util.train_test_split_by_indices(description, c_dates)
        if description["GRID_TYPE"] == "Flat":
            masks_train, masks_test = util.train_test_split_by_indices(description, masks)
    elif "TEST_FRACTION" in description.keys():
        seed = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max)
        x_train, x_test = util.train_test_split_by_proportion(description, predictors, seed)
        y_train, y_test = util.train_test_split_by_proportion(description, targets, seed)
        indices_train, indices_test = util.train_test_split_by_proportion(description, indices, seed)
        timesteps_train, timesteps_test = util.train_test_split_by_proportion(description, c_dates, seed)
        if description["GRID_TYPE"] == "Flat":
            masks_train, masks_test = util.train_test_split_by_proportion(description, masks, seed)
    else:
        test_mask = c_dates > description["SPLIT_YEAR"]
        x_train = predictors[~test_mask, ...]
        x_test = predictors[test_mask, ...]
        y_train = targets[~test_mask, ...]
        y_test = targets[test_mask, ...]
        indices_train = indices[~test_mask, ...]
        indices_test = indices[test_mask, ...]
        timesteps_train = c_dates[~test_mask, ...]
        timesteps_test = c_dates[test_mask, ...]
        if description["GRID_TYPE"] == "Flat":
            masks_train = masks[~test_mask, ...]
            masks_test = masks[test_mask, ...]
    if description["GRID_TYPE"] == "Flat":
        x_train = x_train.reshape(x_train.shape[0], len(pvars), -1, x_train.shape[-1])
        x_test = x_test.reshape(x_test.shape[0], len(pvars), -1, x_test.shape[-1])
        y_train = y_train.reshape(y_train.shape[0], len(tvars), -1, y_train.shape[-1])
        y_test = y_test.reshape(y_test.shape[0], len(tvars), -1, y_test.shape[-1])
        masks_train = masks_train.reshape(masks_train.shape[0], len(tvars), -1, masks_train.shape[-1])
        masks_test = masks_test.reshape(masks_test.shape[0], len(tvars), -1, masks_test.shape[-1])
    elif description["GRID_TYPE"] == "Ico":
        ico = icosahedron.Icosahedron(r=description["RESOLUTION"])
        charts = ico.get_charts_cut()
        x_train = x_train.reshape(x_train.shape[0], len(pvars), -1, charts.shape[-2])
        x_test = x_test.reshape(x_test.shape[0], len(pvars), -1, charts.shape[-2])
        y_train = y_train.reshape(y_train.shape[0], len(tvars), -1, charts.shape[-2])
        y_test = y_test.reshape(y_test.shape[0], len(tvars), -1, charts.shape[-2])
    else:
        raise NotImplementedError("Only Flat and Ico grid implemented")

    dataset = {}
    if description["GRID_TYPE"] == "Ico":
        dataset['test'] = {
            'predictors': x_test,
            'targets': y_test
        }
        dataset['train'] = {
            'predictors': x_train,
            'targets': y_train
        }
    elif description["GRID_TYPE"] == "Flat":
        dataset['test'] = {
            'predictors': x_test,
            'targets': y_test,
            'masks': masks_test
        }
        dataset['train'] = {
            'predictors': x_train,
            'targets': y_train,
            'masks': masks_train
        }
    else:
        raise NotImplementedError("Invalid grid type")
    name = util.create_hash_from_description(description)
    folder_name = os.path.join(output_folder, "dset_{}".format(name))

    if util.test_if_folder_exists(folder_name):
        raise FileExistsError("Specified dataset already exists.")
    else:
        os.makedirs(folder_name)
    dataset_file = os.path.join(folder_name, "dataset.gz")
    description_file = os.path.join(folder_name, "description.gz")

    print("writing pickle")
    with gzip.open(dataset_file, 'wb') as f:
        pickle.dump(dataset, f)
    print("done")

    dataset_dict = dict(description, **{
        "INDICES_TRAIN": indices_train,
        "INDICES_TEST": indices_test,
        "TIMESTEPS_TEST": timesteps_test,
        "GRID_SHAPE": x_test.shape[-2:]
    })

    print("writing dataset description")
    with gzip.open(description_file, 'wb') as f:
        pickle.dump(dataset_dict, f)
    print("done")


def load_variables_and_timesteps_months(description, dataset_folder):
    """
    Similar to load variables, now for monthly timescale
    @param description: Description of dataset
    @param dataset_folder: Directory to seach for fitting datasets.
    @return:
    """
    assert "START_YEAR" in description.keys()
    assert "END_YEAR" in description.keys()
    assert "PREDICTOR_VARIABLES" in description.keys()
    assert "TARGET_VARIABLES" in description.keys()
    assert "LATITUDES_SLICE" in description.keys()
    assert description["TIMESCALE"] == "MONTHLY"
    assert description["PRECIP_WEIGHTING"] is False
    assert description["GRID_TYPE"] == "Flat"

    variables = {}
    masks = {}

    datasets = get_required_datasets(description, dataset_folder)

    # make sure that all datasets that have a non-trivial time axis share the same calendar and units.
    description["CALENDAR"], description["T_UNITS"] = util.load_units_cals(description, list(datasets.values()))

    c_years = np.array(get_shared_timesteps(description, dataset_folder))
    c_mask = np.logical_and(c_years[:, 0] >= description["START_YEAR"],
                            c_years[:, 0] < description["END_YEAR"])
    c_dates = c_years[c_mask]
    c_dates = c_dates[c_dates[:, 1].argsort(kind="stable")]
    c_dates = c_dates[c_dates[:, 0].argsort(kind="stable")]

    for dataset_name, dataset in datasets.items():  # loop over all used datasets
        description["LATITUDES"], description["LONGITUDES"] = util.load_longitudes_latitudes(description, dataset)

        variables[dataset_name] = {}
        masks[dataset_name] = {}
        variables[dataset_name] = {}


        # loop over all variables we want to use from this dataset
        for variable_name in dict(description["PREDICTOR_VARIABLES"], **description["TARGET_VARIABLES"])[dataset_name]:
            variables[dataset_name][variable_name] = {}

            if dataset.variables[variable_name][:].data.shape[0] > 1:  # only if time dimension is not trivial

                try:
                    years, months = util.get_years_months(dataset.variables["t"][:].data, dataset.variables["t"].units,
                                                          dataset.variables["t"].calendar)
                except KeyError:
                    years, months = util.get_years_months(dataset.variables["time"][:].data, dataset.variables["time"].units,
                                                          dataset.variables["time"].calendar)
                # print(years, months)
                # get the corresponding indices:
                indices = []
                for i, (y, m) in enumerate(zip(years, months)):
                    # print(y,m)
                    if np.array([y, m]) in c_dates:
                        indices.append(i)
                indices = np.array(indices, dtype=int)

                months_indices = []
                for i in range(12):
                    # print(np.where(np.array(months) == i), i, months)
                    months_indices.append(np.where(np.array(months) == i))
                indices_selected_months = [np.squeeze(months_indices[i]) for i in description["MONTHS_USED"]]
                # we need to sort because otherwise dataset is not ordered chronologically but one sorted by months.
                indices_selected_months = np.sort(np.concatenate(tuple(indices_selected_months)))
                sel_i = indices[indices_selected_months]
                # we need to make sure that the other months used in the prediction are not missing from the dataset.
                all_needed_months_contained = np.array([np.array([util.add_dates(years[i], months[i], 0, dm) in c_dates
                                                                  for dm in description["MONTHS_USED_IN_PREDICTION"]]).all()
                                                        for i in sel_i])
                # filter out all months that are not possible as predictable target.
                sel_i = sel_i[all_needed_months_contained]
                data = np.squeeze(dataset.variables[variable_name][:][..., description["LATITUDES_SLICE"][0]:
                                                                           description["LATITUDES_SLICE"][1], :])
            else:
                data = np.squeeze(np.repeat(dataset.variables[variable_name][:][..., description["LATITUDES_SLICE"][0]:
                                            description["LATITUDES_SLICE"][1], :], repeats=len(sel_i), axis=0))


            # if there are missing values in any of the extracted data for this var, store the masks too.
            if dataset_name in description["TARGET_VARIABLES"].keys():
                masks[dataset_name][variable_name] = np.ma.getmaskarray(data[sel_i])
            # load the marker for the missing value in the dataset
            try:
                missing_value = dataset.variables[variable_name].missing_value
                for d_m in description["MONTHS_USED_IN_PREDICTION"]:
                    variables[dataset_name][variable_name][d_m] = data[sel_i + d_m, ...]
                    variables[dataset_name][variable_name][d_m].data[(variables[dataset_name][variable_name][d_m].data == missing_value)] = np.nan
            except AttributeError:
                for d_m in description["MONTHS_USED_IN_PREDICTION"]:
                    variables[dataset_name][variable_name][d_m] = data[sel_i + d_m, ...]

    res_variables = {}
    res_masks = {}
    for dataset_name, dataset in variables.items():  # loop over all used datasets
        v = dict(description["PREDICTOR_VARIABLES"], **description["TARGET_VARIABLES"])
        for variable_name in v[dataset_name]:  # loop over all variables we want to use from this dataset
            res_variables[variable_name] = {}
            for dm in description["MONTHS_USED_IN_PREDICTION"]:
                res_variables[variable_name][dm] = dataset[variable_name][dm]

    # exclude timesteps with missing values in predictor variables. Assume that the shape of all the elements in the res_variables dict is identical.
    masked_timesteps = np.zeros(len(sel_i), dtype=bool)
    for vs in description["PREDICTOR_VARIABLES"].values():
        for v in vs:
            for dm in description["MONTHS_USED_IN_PREDICTION"]:
                if (np.ma.getmaskarray(res_variables[v][dm]) != False).any():
                    m = np.where(np.mean(np.ma.getmaskarray(res_variables[v][dm]), axis=(-1, -2)) > 0)[0]
                    masked_timesteps[m] = True
    c_dates = c_dates[sel_i][~masked_timesteps]
    for key, vs in res_variables.items():
        for dm, v in vs.items():
            res_variables[key][dm] = v[~masked_timesteps, ...].data
    for dataset_name, dataset_masks in masks.items():  # loop over all used datasets
        v = dict(description["PREDICTOR_VARIABLES"], **description["TARGET_VARIABLES"])
        for variable_name in v[dataset_name]:  # loop over all variables we want to use from this dataset
            if variable_name in dataset_masks.keys():
                res_masks[variable_name] = dataset_masks[variable_name][~masked_timesteps, ...]
    # Run some tests that make sure, that we in fact created masks for the target variables and no masks for
    # the predictor variables
    for key in res_masks.keys():
        assert(key not in util.flatten(description["PREDICTOR_VARIABLES"].values()))
    for pvar in util.flatten(description["TARGET_VARIABLES"].values()):
        assert(pvar in res_masks.keys())
    for pvar in res_masks.keys():
        assert(pvar in util.flatten(description["TARGET_VARIABLES"].values()))
    return res_variables, res_masks, c_dates


def load_variables_and_timesteps_precip_weighted(description, dataset_folder):
    """
    Similar to load variables, now load months and average them, weighted by precipitation amount.
    @param description: Description of data set
    @param dataset_folder: Directory to search for fitting data sets.
    @return:
    """
    assert "START_YEAR" in description.keys()
    assert "END_YEAR" in description.keys()
    assert "PREDICTOR_VARIABLES" in description.keys()
    assert "TARGET_VARIABLES" in description.keys()
    assert "LATITUDES_SLICE" in description.keys()
    assert description["PRECIP_WEIGHTING"] is True
    assert ["prec"] in description["PREDICTOR_VARIABLES"].values() or ["precip"] in description["PREDICTOR_VARIABLES"].values()
    assert description["TIMESCALE"] == "YEARLY"
    assert description["GRID_TYPE"] == "Flat"
    variables = {}
    masks = {}

    description["TIMESCALE"] = "MONTHLY"
    datasets_monthly = get_required_datasets(description, dataset_folder)
    description["TIMESCALE"] = "YEARLY"
    datasets_yearly = get_required_datasets(description, dataset_folder)

    # make sure that all datasets that have a non-trivial time axis share the same calendar and units.

    description["CALENDAR"], description["T_UNITS"] = util.load_units_cals(description, list(datasets_yearly.values())+list(datasets_monthly.values()))

    c_years = np.array(get_shared_timesteps(description, dataset_folder))
    c_mask = np.logical_and(c_years >= description["START_YEAR"],
                            c_years < description["END_YEAR"])
    c_dates = c_years[c_mask]
    try:
        p_data = np.squeeze(datasets_monthly["prec"].variables["prec"][:].data[..., description["LATITUDES_SLICE"][0]:
                                                                                description["LATITUDES_SLICE"][1], :])
    except KeyError:
        p_data = np.squeeze(datasets_monthly["precip"].variables["precip"][:].data[..., description["LATITUDES_SLICE"][0]:
                                                                                description["LATITUDES_SLICE"][1], :])
    for dataset_name, dataset in datasets_monthly.items():  # loop over all used datasets
        description["LATITUDES"], description["LONGITUDES"] = util.load_longitudes_latitudes(description, dataset)
        try:
            years = util.get_years_months(dataset.variables["t"][:].data, dataset.variables["t"].units,
                                          dataset.variables["t"].calendar)[0]
        except KeyError:
            years = util.get_years_months(dataset.variables["time"][:].data, dataset.variables["time"].units,
                                          dataset.variables["time"].calendar)[0]

        variables[dataset_name] = {}
        masks[dataset_name] = {}

        for variable_name in dict(description["PREDICTOR_VARIABLES"], **description["TARGET_VARIABLES"])[dataset_name]:
            variables[dataset_name][variable_name] = {}
            if dataset.variables[variable_name][:].data.shape[0] > 1:  # only if time dimension is not trivial
                data = np.squeeze(dataset.variables[variable_name][:].data[..., description["LATITUDES_SLICE"][0]:
                                                                           description["LATITUDES_SLICE"][1], :])
                # load the marker for the missing value in the dataset
                try:
                    missing_value = dataset.variables[variable_name].missing_value
                    data[data == missing_value] = np.nan
                except AttributeError:
                    pass

                weights = p_data
                masked_var = np.ma.MaskedArray(data, mask=np.isnan(data))
                var_yearly_data = np.ma.zeros((len(c_dates), *data.shape[1:]))
                if variable_name == "prec" or variable_name == "precip":
                    for i, yr in enumerate(c_dates):
                        i_mask = (yr == years)
                        var_yearly_data[i, ...] = np.ma.average(masked_var[i_mask, ...], axis=0)
                else:
                    for i, yr in enumerate(c_dates):
                        i_mask = (yr == years)
                        var_yearly_data[i, ...] = np.ma.average(masked_var[i_mask, ...], weights=weights[i_mask, ...],
                                                                axis=0)
                res = var_yearly_data
            else:
                res = np.squeeze(
                    np.repeat(dataset.variables[variable_name][:][..., description["LATITUDES_SLICE"][0]:
                                                                  description["LATITUDES_SLICE"][1], :],
                              repeats=len(c_dates), axis=0))
            variables[dataset_name][variable_name] = res

            if dataset_name in description["TARGET_VARIABLES"].keys():
                masks[dataset_name][variable_name] = np.ma.getmaskarray(variables[dataset_name][variable_name])

    res_variables = {}
    res_masks = {}

    for dataset_name, dataset in variables.items():  # loop over all used datasets
        v = dict(description["PREDICTOR_VARIABLES"], **description["TARGET_VARIABLES"])
        for variable_name in v[dataset_name]:  # loop over all variables we want to use from this dataset
            res_variables[variable_name] = dataset[variable_name]

    # exclude timesteps with missing values in predictor variables.
    masked_timesteps = np.zeros(list(res_variables.values())[0].shape[0], dtype=bool)
    for vs in description["PREDICTOR_VARIABLES"].values():
        for v in vs:
            if (np.ma.getmaskarray(res_variables[v]) != False).any():
                m = np.where(np.mean(np.ma.getmaskarray(res_variables[v]), axis=(-1, -2, -3)) > 0)[0]
                masked_timesteps[m] = True
    c_dates = c_dates[~masked_timesteps]

    for dataset_name, dataset_masks in masks.items():  # loop over all used datasets
        v = dict(description["PREDICTOR_VARIABLES"], **description["TARGET_VARIABLES"])
        for variable_name in v[dataset_name]:  # loop over all variables we want to use from this dataset
            if variable_name in dataset_masks.keys():
                res_masks[variable_name] = dataset_masks[variable_name][~masked_timesteps, ...]

    for key, v in res_variables.items():
        res_variables[key] = v[~masked_timesteps, ...].data
    return res_variables, res_masks, c_dates


def create_precip_weighted_dataset(description, dataset_folder, output_folder):
    """
    create a dataset that we can do ML on from climate model output file in folder dataset_folder and given a dict of
    decisions in the creation process (description). Weight variables by precipitation amount.
    """
    assert "DO_SHUFFLE" in description.keys()
    assert np.logical_xor("TEST_FRACTION" in description.keys(), "SPLIT_YEAR" in description.keys())
    assert description["PRECIP_WEIGHTING"] is True
    assert description["TIMESCALE"] == "YEARLY"
    assert description["GRID_TYPE"] == "Flat"

    print("loading variables")
    # load the selected climate variables.
    variables, masks, c_dates = load_variables_and_timesteps_precip_weighted(description, dataset_folder)

    # split the variables into predictors and targets.
    pvars = util.flatten(description["PREDICTOR_VARIABLES"].values())
    tvars = util.flatten(description["TARGET_VARIABLES"].values())
    predictors = np.concatenate(tuple([variables[p_var] for p_var in pvars]), axis=1)
    targets = np.concatenate(tuple([variables[t_var] for t_var in tvars]), axis=1)
    masks = np.concatenate(tuple([masks[t_var] for t_var in tvars], ))

    indices = np.arange(predictors.shape[0])
    # if we want to reload a dataset with a specific configuration of indices for training and testing
    if "INDICES_TEST" in description.keys():
        x_train, x_test = util.train_test_split_by_indices(description, predictors)
        y_train, y_test = util.train_test_split_by_indices(description, targets)
        indices_train, indices_test = util.train_test_split_by_indices(description, indices)
        timesteps_train, timesteps_test = util.train_test_split_by_indices(description, c_dates)
        masks_train, masks_test = util.train_test_split_by_indices(description, masks)
    elif "TEST_FRACTION" in description.keys():
        seed = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max)
        x_train, x_test = util.train_test_split_by_proportion(description, predictors, seed)
        y_train, y_test = util.train_test_split_by_proportion(description, targets, seed)
        indices_train, indices_test = util.train_test_split_by_proportion(description, indices, seed)
        timesteps_train, timesteps_test = util.train_test_split_by_proportion(description, c_dates, seed)
        masks_train, masks_test = util.train_test_split_by_proportion(description, masks, seed)
    else:
        test_mask = c_dates > description["SPLIT_YEAR"]
        x_train = predictors[~test_mask, ...]
        x_test = predictors[test_mask, ...]
        y_train = targets[~test_mask, ...]
        y_test = targets[test_mask, ...]
        indices_train = indices[~test_mask, ...]
        indices_test = indices[test_mask, ...]
        timesteps_train = c_dates[~test_mask, ...]
        timesteps_test = c_dates[test_mask, ...]
        masks_train = masks[~test_mask, ...]
        masks_test = masks[test_mask, ...]

    x_train = x_train.reshape(x_train.shape[0], len(pvars), -1, x_train.shape[-1])
    x_test = x_test.reshape(x_test.shape[0], len(pvars), -1, x_test.shape[-1])
    y_train = y_train.reshape(y_train.shape[0], len(tvars), -1, y_train.shape[-1])
    y_test = y_test.reshape(y_test.shape[0], len(tvars), -1, y_test.shape[-1])
    masks_train = masks_train.reshape(masks_train.shape[0], len(tvars), -1, masks_train.shape[-1])
    masks_test = masks_test.reshape(masks_test.shape[0], len(tvars), -1, masks_test.shape[-1])

    name = util.create_hash_from_description(description)
    folder_name = os.path.join(output_folder, "dset_{}".format(name))

    if util.test_if_folder_exists(folder_name):
        raise FileExistsError("Specified dataset already exists.")
    else:
        os.makedirs(folder_name)
    dataset_file = os.path.join(folder_name, "dataset.gz")
    description_file = os.path.join(folder_name, "description.gz")

    dataset = {}
    dataset['test'] = {
        'predictors': x_test,
        'targets': y_test,
        'masks': masks_test
    }
    dataset['train'] = {
        'predictors': x_train,
        'targets': y_train,
        'masks': masks_train
    }

    print("writing pickle")
    with gzip.open(dataset_file, 'wb') as f:
        pickle.dump(dataset, f)
    print("done")

    dataset_dict = dict(description, **{
        "INDICES_TRAIN": indices_train,
        "INDICES_TEST": indices_test,
        "TIMESTEPS_TEST": timesteps_test,
        "GRID_SHAPE": x_test.shape[-2:]
    })

    print("writing dataset description")
    with gzip.open(description_file, 'wb') as f:
        pickle.dump(dataset_dict, f)
    print("done")


def create_monthly_dataset(description, dataset_folder, output_folder):
    """
    create a dataset that we can do ML on from climate model output file in folder dataset_folder and given a dict of
    decisions in the creation process (description).
    """
    assert "DO_SHUFFLE" in description.keys()
    assert np.logical_xor("TEST_FRACTION" in description.keys(), "SPLIT_YEAR" in description.keys())
    assert description["TIMESCALE"] == "MONTHLY"

    print("loading variables")
    # load the selected climate variables.
    variables, masks, c_dates = load_variables_and_timesteps_months(description, dataset_folder)
    # split the variables into predictors and targets.
    pvars = util.flatten(description["PREDICTOR_VARIABLES"].values())
    tvars = util.flatten(description["TARGET_VARIABLES"].values())

    predictors = [[variables[p_var][d_m] for d_m in description["MONTHS_USED_IN_PREDICTION"]] for p_var in pvars]
    predictors = np.array(predictors)
    predictors = predictors.reshape(predictors.shape[0]*predictors.shape[1], *predictors.shape[2:])
    predictors = predictors.transpose(1, 0, 2, 3)

    targets = np.concatenate(tuple([variables[t_var][0] for t_var in tvars], ))
    masks = np.concatenate(tuple([masks[t_var] for t_var in tvars], ))

    indices = np.arange(predictors.shape[0])

    # if we want to reload a dataset with a specific configuration of indices for training and testing
    if "INDICES_TEST" in description.keys():
        x_train, x_test = util.train_test_split_by_indices(description, predictors)
        y_train, y_test = util.train_test_split_by_indices(description, targets)
        indices_train, indices_test = util.train_test_split_by_indices(description, indices)
        timesteps_train, timesteps_test = util.train_test_split_by_indices(description, c_dates)
        masks_train, masks_test = util.train_test_split_by_indices(description, masks)
    elif "TEST_FRACTION" in description.keys():
        seed = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max)
        x_train, x_test = util.train_test_split_by_proportion(description, predictors, seed)
        y_train, y_test = util.train_test_split_by_proportion(description, targets, seed)
        indices_train, indices_test = util.train_test_split_by_proportion(description, indices, seed)
        timesteps_train, timesteps_test = util.train_test_split_by_proportion(description, c_dates, seed)
        masks_train, masks_test = util.train_test_split_by_proportion(description, masks, seed)
    else:
        test_mask = c_dates[:, 0] > description["SPLIT_YEAR"]
        x_train = predictors[~test_mask, ...]
        x_test = predictors[test_mask, ...]
        y_train = targets[~test_mask, ...]
        y_test = targets[test_mask, ...]
        indices_train = indices[~test_mask, ...]
        indices_test = indices[test_mask, ...]
        timesteps_train = c_dates[~test_mask, ...]
        timesteps_test = c_dates[test_mask, ...]
        masks_train = masks[~test_mask, ...]
        masks_test = masks[test_mask, ...]

    x_train = x_train.reshape(x_train.shape[0], len(description["MONTHS_USED_IN_PREDICTION"])*len(pvars), -1, x_train.shape[-1])
    x_test = x_test.reshape(x_test.shape[0], len(description["MONTHS_USED_IN_PREDICTION"])*len(pvars), -1, x_test.shape[-1])
    y_train = y_train.reshape(y_train.shape[0], len(tvars), -1, y_train.shape[-1])
    y_test = y_test.reshape(y_test.shape[0], len(tvars), -1, y_test.shape[-1])
    masks_train = masks_train.reshape(masks_train.shape[0], len(tvars), -1, masks_train.shape[-1])
    masks_test = masks_test.reshape(masks_test.shape[0], len(tvars), -1, masks_test.shape[-1])
    dataset = {'test': {
        'predictors': x_test,
        'targets': y_test,
        'masks': masks_test
    }, 'train': {
        'predictors': x_train,
        'targets': y_train,
        'masks': masks_train
    }}

    name = util.create_hash_from_description(description)
    folder_name = os.path.join(output_folder, "dset_{}".format(name))
    if util.test_if_folder_exists(folder_name):
        raise FileExistsError("Specified dataset already exists.")
    else:
        os.makedirs(folder_name)
    dataset_file = os.path.join(folder_name, "dataset.gz")
    description_file = os.path.join(folder_name, "description.gz")

    print("writing pickle")
    with gzip.open(dataset_file, 'wb') as f:
        pickle.dump(dataset, f)
    print("done")

    dataset_dict = dict(description, **{
        "INDICES_TRAIN": indices_train,
        "INDICES_TEST": indices_test,
        "TIMESTEPS_TEST": timesteps_test,
        "GRID_SHAPE": x_test.shape[-2:]
    })

    print("writing dataset description")
    with gzip.open(description_file, 'wb') as f:
        pickle.dump(dataset_dict, f)
    print("done")
