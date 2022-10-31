import util
from train import load_data, append_coords, find_and_load_dataset_description
import os.path
import pickle
import gzip

import numpy as np

import torchvision.transforms as T
import torch


def predict_randomforest(x, model):
    """
    Predict with the pixelwise random forest model.
    @param x: Input data, should have shape (n_timesteps, n_predictors, n_lat, n_lon)
    @param model: A trained instance of a random forest model.
    @return: Predictions of the models on the testset
    """
    x = append_coords(x)
    n_timesteps, n_predvars, n_lat, n_lon = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(-1, n_predvars)

    predict_test = model.predict(x)

    return predict_test.reshape(n_timesteps, 1, n_lat, n_lon)


def predict_save_randomforest_pixelwise(dataset_description, model_training_description, base_folder,
                                        model, output_folder, save_model=False):
    """
    Predict on the given data set with an already trained randomforest pixelwise method.
    Then store the model, the description and the results.
    @param dataset_description: Details on the used dataset
    @param model_training_description: Details on training and model
    @param base_folder: Folder to load the model data from
    @param model: List of lists of trained pixelwise linear regression model
    @param output_folder: Folder to save output in
    @param save_model: Whether or not we want to save model
    @return:
    """

    assert "DATASET_FOLDER" in model_training_description.keys()
    dataset_description = find_and_load_dataset_description(model_training_description["DATASET_FOLDER"],
                                                            dataset_description)
    if model_training_description["CREATE_VALIDATIONSET"]:
        _, _, test_ds = load_data(dataset_description, model_training_description, base_folder)
    else:
        _, test_ds = load_data(dataset_description, model_training_description, base_folder)
    x_te = test_ds[:][0].numpy()

    predictions = predict_randomforest(x_te, model)
    descriptions = {"DATASET_DESCRIPTION": dataset_description,
                    "MODEL_TRAINING_DESCRIPTION": model_training_description}
    s1 = util.create_hash_from_description(dataset_description)
    s2 = util.create_hash_from_description(model_training_description)
    folder_name = os.path.join(output_folder, s1 + s2)
    predictions_file = os.path.join(folder_name, "predictions.gz")
    model_file = os.path.join(folder_name, "model.gz")
    descriptions_file = os.path.join(folder_name, "descriptions.gz")
    if util.test_if_folder_exists(folder_name):
        raise FileExistsError("Specified configuration of data set, model and training configuration already exists.")
    else:
        os.makedirs(folder_name)

    print("writing predictions")
    with gzip.open(predictions_file, 'wb') as f:
        pickle.dump(predictions, f)

    if save_model:
        print("writing model")
        with gzip.open(model_file, 'wb') as f:
            pickle.dump(model, f)

    print("writing descriptions")
    with gzip.open(descriptions_file, 'wb') as f:
        pickle.dump(descriptions, f)
    print("done")


def predict_linreg(x, models):
    """
    Predict with the linear regression model.
    @param x: Input data, should have shape (n_timesteps, n_predictors, n_lat, n_lon)
    @param models: 2d list containing linear regression models for each pixel
    @return: Predictions of the models on the testset
    """
    n_timesteps, n_predvars, n_lat, n_lon = x.shape

    predict_test = np.zeros((n_timesteps, 1, n_lat, n_lon))
    for i in range(x.shape[-2]):
        for j in range(x.shape[-1]):
            predict_test[..., i, j] = models[i][j].predict(x[..., i, j])
    return predict_test


def predict_save_linreg_pixelwise(dataset_description, model_training_description,  base_folder,
                                  models, output_folder, save_model=False):
    """
    Predict on the given data set with an already trained randomforest pixelwise method.
    Then store the model, the description and the results.
    @param dataset_description: Details on the used dataset
    @param model_training_description: Details on training and model
    @param base_folder: Folder to load the model data from
    @param models: List of lists of trained pixelwise linear regression model
    @param output_folder: Folder to save output in
    @param save_model: Whether or not we want to save model
    @return:
    """

    assert "DATASET_FOLDER" in model_training_description.keys()
    dataset_description = find_and_load_dataset_description(model_training_description["DATASET_FOLDER"],
                                                            dataset_description)

    if model_training_description["CREATE_VALIDATIONSET"]:
        _, _, test_ds = load_data(dataset_description, model_training_description, base_folder)
    else:
        _, test_ds = load_data(dataset_description, model_training_description, base_folder)

    x_te = test_ds[:][0].numpy()

    predictions = predict_linreg(x_te, models)
    full_dataset_description = find_and_load_dataset_description(model_training_description["DATASET_FOLDER"],
                                                                 dataset_description)
    descriptions = {"DATASET_DESCRIPTION": full_dataset_description,
                    "MODEL_TRAINING_DESCRIPTION": model_training_description}

    s1 = util.create_hash_from_description(full_dataset_description)
    s2 = util.create_hash_from_description(model_training_description)
    folder_name = os.path.join(output_folder, s1 + s2)
    predictions_file = os.path.join(folder_name, "predictions.gz")
    model_file = os.path.join(folder_name, "model.gz")
    descriptions_file = os.path.join(folder_name, "descriptions.gz")
    if util.test_if_folder_exists(folder_name):
        raise FileExistsError("Specified configuration of data set, model and training configuration already exists.")
    else:
        os.makedirs(folder_name)

    print("writing predictions")
    with gzip.open(predictions_file, 'wb') as f:
        pickle.dump(predictions, f)

    if save_model:
        print("writing model")
        with gzip.open(model_file, 'wb') as f:
            pickle.dump(models, f)

    print("writing descriptions")
    with gzip.open(descriptions_file, 'wb') as f:
        pickle.dump(descriptions, f)
    print("done")


def predict_pca(x, pca, pca_targets, model):
    """
    use the trained pca's and the regression to predict on test set.
    """
    n_timesteps, n_predvars, n_lat, n_lon = x.shape
    x_test = x.reshape(n_timesteps, -1)

    x_test_rescaled = pca.transform(x_test)
    predict_test = model.predict(x_test_rescaled)
    predict_test = pca_targets.inverse_transform(predict_test)
    return predict_test.reshape((n_timesteps, 1, n_lat, n_lon))


def predict_save_pca(dataset_description, model_training_description, base_folder,
                     pca, pca_targets, model, output_folder, save_model=False):
    """
    Predict on the given data set with an already trained pca method.
    Then store the model, the description and the results.
    @param save_model: Whether or not we want to store the model
    @param dataset_description: Description of the dataset
    @param model_training_description: Description of training and model
    @param base_folder: Folder in which the dataset is loaded from
    @param output_folder: Folder in which to store output
    @param pca: Previously trained PCA of the predictor variables
    @param pca_targets: Previously trained PCA of the target variables
    @param model: Regression model
    """
    dataset_description = find_and_load_dataset_description(model_training_description["DATASET_FOLDER"],
                                                            dataset_description)
    if model_training_description["CREATE_VALIDATIONSET"]:
        _, _, test_ds = load_data(dataset_description, model_training_description, base_folder)
    else:
        _, test_ds = load_data(dataset_description, model_training_description, base_folder)
    x_te = test_ds[:][0].numpy()

    predictions = predict_pca(x_te, pca, pca_targets, model)
    descriptions = {"DATASET_DESCRIPTION": dataset_description,
                    "MODEL_TRAINING_DESCRIPTION": model_training_description}

    s1 = util.create_hash_from_description(dataset_description)
    s2 = util.create_hash_from_description(model_training_description)
    folder_name = os.path.join(output_folder, s1 + s2)
    predictions_file = os.path.join(folder_name, "predictions.gz")
    model_file = os.path.join(folder_name, "model.gz")
    pca_file = os.path.join(folder_name, "pca.gz")
    pca_targets_file = os.path.join(folder_name, "pca_targets.gz")
    descriptions_file = os.path.join(folder_name, "descriptions.gz")

    if util.test_if_folder_exists(folder_name):
        raise FileExistsError("Specified configuration of dataset, model and training configuration already exists.")
    else:
        os.makedirs(folder_name)

    print("writing predictions")
    with gzip.open(predictions_file, 'wb') as f:
        pickle.dump(predictions, f)

    if save_model:
        print("writing model")
        with gzip.open(model_file, 'wb') as f:
            pickle.dump(model, f)
        with gzip.open(pca_file, 'wb') as f:
            pickle.dump(pca, f)
        with gzip.open(pca_targets_file, 'wb') as f:
            pickle.dump(pca_targets, f)

    print("writing descriptions")
    with gzip.open(descriptions_file, 'wb') as f:
        pickle.dump(descriptions, f)
    print("done")


def predict_unet(x, x_loader, model_training_description, model):
    """
    Predict with a trained instance of a UNet.
    @param x: Input dataset
    @param x_loader: Loader for x data
    @param model_training_description: Description of the model and training
    @param model: Trained UNet model
    @return:
    """
    assert model_training_description["MODEL_TYPE"] in ["UNet_Ico", "UNet_Flat"]
    predictions_model = torch.zeros_like(x[...][1])

    # loop over test loader
    for idx, batch in enumerate(x_loader):
        model.eval()
        with torch.no_grad():
            predictors = batch[0]
            predictors = predictors.to(model_training_description["DEVICE"])
            outputs = model(predictors)
            predictions_model[idx * model_training_description["BATCH_SIZE"]:(idx + 1) *
                              model_training_description["BATCH_SIZE"], ...] = outputs.cpu()
    return predictions_model


def predict_save_unet(dataset_description, model_training_description, base_folder, model, output_folder,
                      save_model=False):
    """
    Predict on the given data set with an already trained pca method.
    Then store the model, the description and the results.
    @param save_model: Whether or not we want to store the model
    @param dataset_description: Description of the dataset
    @param model_training_description: Description of training and model
    @param base_folder: Folder in which the dataset is loaded from
    @param output_folder: Folder in which to store output
    @param model: UNet model
    """
    if model_training_description["CREATE_VALIDATIONSET"] is True:
        _, _, testloader, _, _, testset = load_data(dataset_description, model_training_description, base_folder)
    else:
        _, testloader, _, testset = load_data(dataset_description, model_training_description, base_folder)

    dataset_description = find_and_load_dataset_description(model_training_description["DATASET_FOLDER"],
                                                            dataset_description)

    predictions = predict_unet(testset, testloader, model_training_description, model)
    if model_training_description["MODEL_TYPE"] == "UNet_Flat":
        predictions = T.Resize(size=dataset_description["GRID_SHAPE"])(predictions).numpy()
    descriptions = {"DATASET_DESCRIPTION": dataset_description,
                    "MODEL_TRAINING_DESCRIPTION": model_training_description}

    s1 = util.create_hash_from_description(dataset_description)
    s2 = util.create_hash_from_description(model_training_description)
    folder_name = os.path.join(output_folder, s1 + s2)
    predictions_file = os.path.join(folder_name, "predictions.gz")
    model_file = os.path.join(folder_name, "model.gz")
    descriptions_file = os.path.join(folder_name, "descriptions.gz")

    if util.test_if_folder_exists(folder_name):
        print(model_training_description)
        print(util.create_hash_from_description(model_training_description))
        raise FileExistsError("Specified configuration of dataset, model and training configuration already exists.")
    else:
        os.makedirs(folder_name)

    print("writing predictions")
    with gzip.open(predictions_file, 'wb') as f:
        pickle.dump(predictions, f)

    if save_model:
        print("writing model")
        with gzip.open(model_file, 'wb') as f:
            pickle.dump(model, f)

    print("writing descriptions")
    with gzip.open(descriptions_file, 'wb') as f:
        pickle.dump(descriptions, f)
    print("done")


def interpolate_data_between_grids(data, in_description, out_description):
    """
    When doing cross-prediction, data can lie on different grids.
    We use interpolation to go from one grid to another.
    Latitudes and Longitudes are extracted from the descriptions of the datasets.

    Assume data has shape (..., n_lats, n_lons).
    @param data: Data to be interpolated
    @param in_description: Description of the data set we interpolate from
    @param out_description: Description of the data set we interpolate to
    @return: Interpolated data set.
    """

    from scipy.interpolate import RegularGridInterpolator

    # for latitudes, one can specify invalid latitudes by using "LATITUDES_SLICE". These are already excluded here
    lat_in = in_description["LATITUDES"]
    lon_in = in_description["LONGITUDES"]

    lat_out = out_description["LATITUDES"]
    lon_out = out_description["LONGITUDES"]

    lat_mg_out, lon_mg_out = np.meshgrid(lat_out, lon_out, indexing='ij')

    ds = data.shape
    data = data.reshape(-1, ds[-2], ds[-1])  # flatten everything but lat and lon

    res = np.zeros((data.shape[0], len(lat_out), len(lon_out)))

    for i in range(len(data)):
        interp = RegularGridInterpolator((lat_in, lon_in), data[i], bounds_error=False, fill_value=None)
        res[i] = interp((lat_mg_out, lon_mg_out))

    return res
