from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import numpy as np
import os.path
import pickle
import gzip

from train import find_and_load_dataset
import util


def get_rmse(predictions, targets, masks=None):
    """
    Calculate the RMSE between predictions and targets. Assume the shape of the input starts with the timesteps.
    @param predictions: Model predictions (rescaled)
    @param targets: Ground truth
    @param masks: Masks that contain information on missing data
    @return: RMSE, same shape as input without time dimension
    """

    ps = predictions.reshape(predictions.shape[0], -1)
    gt = targets.reshape(targets.shape[0], -1)

    res = np.zeros((ps.shape[1]))
    if masks is None:
        for i in range(ps.shape[1]):
            res[i] = mean_squared_error(ps[:, i], gt[:, i], squared=False)
    else:
        masks = masks.reshape(masks.shape[0], -1)
        for i in range(ps.shape[1]):
            res[i] = mean_squared_error(ps[~masks[:, i], i], gt[~masks[:, i], i], squared=False)
    return res.reshape(*predictions.shape[1:])


def get_r2(predictions, targets, masks=None):
    """
    Calculate the R^2-score between predictions and targets.
    Assume the shape of the input starts with the timesteps.

    @param predictions: Model predictions (rescaled)
    @param targets: Ground truth
    @param masks: Masks that contain information on missing data
    @return: R^2-score, same shape as input without time dimension
    """
    rmse = get_rmse(predictions, targets, masks)  # calculate the RMSE
    std = np.nanstd(targets, axis=0)  # calculate the standard deviation of the ground truth
    if (std == 0).any():
        raise ValueError("Standard deviation of the test set is zero along at least one dimension.")
    return 1 - rmse/std


def get_weighted_average(data, dataset_description):
    """
    Calculate an area-weighted average of a lat/lon field of grid boxes.

    @param data: Data to be averaged
    @param dataset_description: Description containing information on the dataset.
    @return: Area weighted average of quantity data.
    """
    assert "LATITUDES" in dataset_description.keys()
    assert "LONGITUDES" in dataset_description.keys()

    # load the true dataset description taking input as conditions.
    latitudes = np.array(dataset_description["LATITUDES"])
    longitudes = np.array(dataset_description["LONGITUDES"])

    weights = np.tile(np.cos(np.deg2rad(latitudes))[:, None], len(longitudes))[np.newaxis, ...]
    data_avg = np.average(data, weights=weights, axis=(1, 2))
    return data_avg


def get_correlation(predictions, targets):
    """
    Compute the correlation between predictions and targets. For each grid box, ignore timesteps with NaNs.
    @param predictions: Model predictions (rescaled)
    @param targets: Ground truth
    @return: Correlation coefficients for every grid box and variable
    """
    pearson_correlation = np.zeros(targets.shape[1:])
    for k in range(targets.shape[-3]):
        for i in range(targets.shape[-2]):
            for j in range(targets.shape[-1]):
                nas = np.logical_or(np.isnan(predictions[:, k, i, j]), np.isnan(targets[:, k, i, j]))
                pearson_correlation[k, i, j] = pearsonr(predictions[~nas, k, i, j], targets[~nas, k, i, j])[0]
    return pearson_correlation


def load_compatible_available_runs(base_folder, conditions):
    """
    Given a base folder to search in, list all the runs, that match the specified conditions. Search is shallow, so
    runs in subdirectories won't be detected.

    @param base_folder: Folder in which we want to search for runs.
    @param conditions: dict of conditions on data set and (model and training)
    @return: List of paths to compatible folders
    """
    assert "DATASET_DESCRIPTION" in conditions.keys()
    assert "MODEL_TRAINING_DESCRIPTION" in conditions.keys()
    counter = 0
    predictions_list = []
    descriptions_list = []
    for folder in [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]:
        files = [f for f in os.listdir(os.path.join(base_folder, folder)) if
                 os.path.isfile(os.path.join(base_folder, folder, f))]
        if "descriptions.gz" in files and "predictions.gz" in files:
            with gzip.open(os.path.join(base_folder, folder, "descriptions.gz"), 'rb') as f:
                descriptions = pickle.load(f)
                dataset_description = descriptions["DATASET_DESCRIPTION"]
                model_training_description = descriptions["MODEL_TRAINING_DESCRIPTION"]

            if util.check_dict_conditions(dataset_description, conditions["DATASET_DESCRIPTION"]) and \
                    util.check_dict_conditions(model_training_description, conditions["MODEL_TRAINING_DESCRIPTION"]):
                counter += 1
                with gzip.open(os.path.join(base_folder, folder, "predictions.gz"), 'rb') as f:
                    prs = pickle.load(f)
                predictions_list.append(prs)
                descriptions_list.append({"DATASET_DESCRIPTION": dataset_description,
                                          "MODEL_TRAINING_DESCRIPTION": model_training_description})
    print("{} matching runs found".format(counter))
    return predictions_list, descriptions_list


def undo_scaling(model_training_description, predictions, train_targets):
    """
    If necessary, undo the scaling of the variables.
    @param model_training_description: Description of model and training
    @param predictions: Predictions to be rescaled
    @param train_targets: Target variables of the original data set, before rescaling
    @return:
    """
    rescaled_predictions = np.zeros_like(predictions)
    for j, mode in enumerate(model_training_description["S_MODE_TARGETS"]):
        if mode == "Global":
            std = np.mean(np.nanstd(train_targets, axis=0, keepdims=True), axis=(1, 2), keepdims=True)
            std[std == 0] = 1
            mean = np.nanmean(train_targets, axis=(0, 1, 2), keepdims=True)
            rescaled_predictions[:, j, ...] = (predictions[:, j, ...] * std) + mean
        elif mode == "Pixelwise":
            std = np.nanstd(train_targets, axis=0, keepdims=True)
            std[std == 0] = 1
            mean = np.nanmean(train_targets, axis=0, keepdims=True)
            rescaled_predictions[:, j, ...] = (predictions[:, j, ...] * std) + mean
        elif mode == "Global_mean_pixelwise_std":
            std = np.mean(np.nanstd(train_targets, axis=0, keepdims=True), axis=(1, 2), keepdims=True)
            std[std == 0] = 1
            mean = np.nanmean(train_targets, axis=0, keepdims=True)
            rescaled_predictions[:, j, ...] = (predictions[:, j, ...] * std) + mean
        elif mode == "Pixelwise_mean_global_std":
            std = np.nanstd(train_targets, axis=0, keepdims=True)
            std[std == 0] = 1
            mean = np.nanmean(train_targets, axis=(0, 1, 2), keepdims=True)
            rescaled_predictions[:, j, ...] = (predictions[:, j, ...] * std) + mean
        elif mode == "None":
            rescaled_predictions[:, j, ...] = predictions[:, j, ...]
        else:
            raise NotImplementedError("{} is not a valid keyword for standardization".format(mode))
    return rescaled_predictions


def get_rescaled_predictions_and_gt(descriptions, predictions):
    """
    For a given dataset and model description, load the ground truth test set and rescale the predictions.
    @param descriptions: Descriptions for dataset and (model and training)
    @param predictions: Predictions of the ML model. Potentially to be rescaled.
    @return: Rescaled predictions and ground truth for the test set.
    """
    assert "DATASET_DESCRIPTION" in descriptions.keys()
    assert "MODEL_TRAINING_DESCRIPTION" in descriptions.keys()

    dataset_description = descriptions["DATASET_DESCRIPTION"]
    model_training_description = descriptions["MODEL_TRAINING_DESCRIPTION"]

    assert "DATASET_FOLDER" in model_training_description.keys()

    dataset = find_and_load_dataset(model_training_description["DATASET_FOLDER"], dataset_description, use_prints=False)

    train_targets = dataset["train"]["targets"]
    test_targets = dataset["test"]["targets"]
    if dataset_description["GRID_TYPE"] == "Flat":
        test_masks = dataset["test"]["masks"]

    rescaled_predictions = undo_scaling(model_training_description, predictions, train_targets)

    if dataset_description["GRID_TYPE"] == "Flat":
        return rescaled_predictions, test_targets, test_masks
    else:
        return rescaled_predictions, test_targets


def get_rescaled_predictions_and_gt_split_into_months(descriptions, predictions):
    """
    For a given dataset and model description, load the ground truth test set and rescale the predictions.
    Then split the data into subarray for each month given in descriptions["DATASET_DESCRIPTION"]["MONTHS_USED"]
    @param descriptions: Descriptions for dataset and (model and training)
    @param predictions: Predictions of the ML model. Potentially to be rescaled.
    @return: List of rescaled predictions and targets for the months of the year (Jan: 0, Dec: 11).
    """
    assert "DATASET_DESCRIPTION" in descriptions.keys()
    assert "MODEL_TRAINING_DESCRIPTION" in descriptions.keys()

    dataset_description = descriptions["DATASET_DESCRIPTION"]
    model_training_description = descriptions["MODEL_TRAINING_DESCRIPTION"]

    assert dataset_description["TIMESCALE"] == "MONTHLY"
    assert "DATASET_FOLDER" in model_training_description.keys()

    dataset = find_and_load_dataset(model_training_description["DATASET_FOLDER"], dataset_description, use_prints=False)
    _, t_months, _ = util.get_year_mon_day_from_timesteps(np.array(dataset_description["TIMESTEPS_TEST"]),
                                                          dataset_description["REFERENCE_DATE"])

    train_targets = dataset["train"]["targets"]
    test_targets = dataset["test"]["targets"]
    test_masks = dataset["test"]["masks"]
    rescaled_predictions = undo_scaling(model_training_description, predictions, train_targets)
    rescaled_predictions_months = []
    test_masks_months = []
    test_targets_months = []
    for i in range(12):
        rescaled_predictions_months.append(rescaled_predictions[t_months == i])
        test_targets_months.append(test_targets[t_months == i])
        test_masks_months.append(test_masks[t_months == i])

    return rescaled_predictions_months, test_targets_months, test_masks_months


def load_data_for_comparison(base_folder, conditions):
    """
    For all runs, that match the corresponding definition, load the descriptions of dataset and (model and training)
    and return ground truth and rescaled predictions.
    @param base_folder: Folder in which we want to search for runs.
    @param conditions: dict of conditions on data set and (model and training)
    @return: Lists of descriptions, rescaled predictions and ground truth
    """
    predictions_list, descriptions_list = load_compatible_available_runs(base_folder, conditions)
    rescaled_predictions_list = []
    ground_truth_list = []
    masks_list = []
    for i in range(len(predictions_list)):
        if descriptions_list[i]["DATASET_DESCRIPTION"]["TIMESCALE"] == "MONTHLY":
            rp, gt, masks = get_rescaled_predictions_and_gt_split_into_months(descriptions_list[i], predictions_list[i])
            masks_list.append(masks)
        elif descriptions_list[i]["DATASET_DESCRIPTION"]["TIMESCALE"] == "YEARLY":
            if descriptions_list[i]["DATASET_DESCRIPTION"]["GRID_TYPE"] == "Ico":
                rp, gt = get_rescaled_predictions_and_gt(descriptions_list[i], predictions_list[i])
                masks_list.append(None)
            else:
                rp, gt, masks = get_rescaled_predictions_and_gt(descriptions_list[i], predictions_list[i])
                masks_list.append(masks)
        else:
            raise NotImplementedError("Invalid timescale.")
        rescaled_predictions_list.append(rp)
        ground_truth_list.append(gt)

    return descriptions_list, rescaled_predictions_list, ground_truth_list, masks_list
