import util
import os

import pickle
import gzip

import numpy as np
import torch
import torchvision.transforms as T
import torch.utils.data as data_utils
import torch.nn as nn
from torch.nn import BatchNorm3d as IcoBatchNorm2d

from torch.utils.tensorboard import SummaryWriter

from functools import partial

from ico_unet import UNet, IcoUNet

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def find_and_load_dataset(base_folder, conditions, keywords_blacklist=[], use_prints=False):
    """
    given conditions on the dataset description, find a valid dataset. If there is more than one valid one, we need to
    specify conditions more precisely and raise an Error
    """
    counter = 0
    matching = []
    for folder in [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]:
        files = [f for f in os.listdir(os.path.join(base_folder, folder)) if
                 os.path.isfile(os.path.join(base_folder, folder, f))]
        if "dataset.gz" in files and "description.gz" in files:
            with gzip.open(os.path.join(base_folder, folder, "description.gz"), 'rb') as f:
                tmp_description = pickle.load(f)
            if util.check_dict_conditions(tmp_description, conditions, keywords_blacklist=keywords_blacklist, use_prints=use_prints):
                counter += 1
                matching.append(os.path.join(
                    base_folder, folder, "dataset.gz"))
                with gzip.open(os.path.join(base_folder, folder, "dataset.gz"), 'rb') as g:
                    dataset = pickle.load(g)
    if counter > 1:
        raise ValueError(
            "More than one directory matches the criteria: {}, refine conditions".format(matching))
    elif counter == 1:
        return dataset
    else:
        raise ValueError("No matching folder found")


def find_and_load_dataset_description(base_folder, conditions, keywords_blacklist=[], use_prints=False):
    """
    given conditions on the dataset description, find a valid dataset. If there is more than one valid one, we need to
    specify conditions more precisely and raise an Error
    """
    res = {}
    counter = 0
    for folder in [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]:
        files = [f for f in os.listdir(os.path.join(base_folder, folder)) if
                 os.path.isfile(os.path.join(base_folder, folder, f))]
        if "dataset.gz" in files and "description.gz" in files:
            with gzip.open(os.path.join(base_folder, folder, "description.gz"), 'rb') as f:
                tmp_description = pickle.load(f)
            if util.check_dict_conditions(tmp_description, conditions, keywords_blacklist=keywords_blacklist, use_prints=use_prints):
                counter += 1
                res = {**res, **tmp_description}
    if counter > 1:
        raise ValueError(
            "More than one directory matches the criteria, refine conditions.")
    elif counter == 1:
        return res
    else:
        raise ValueError("No matching folder found")


def load_data(dataset_description, model_training_description, base_folder, use_prints=False):
    """
    load data. Goal is that this method can be applied for all models.
    @param dataset_description: Details on the dataset
    @param model_training_description: Details on the model and training
    @param base_folder: Folder in which to search for dataset.
    @param use_prints: If true, print what condition is violated when trying to load from folders.

    @return:
    """
    assert "DATASET_FOLDER" in model_training_description.keys()
    assert "S_MODE_PREDICTORS" in model_training_description.keys()
    assert "S_MODE_TARGETS" in model_training_description.keys()
    assert "MODEL_TYPE" in model_training_description.keys()
    assert "CREATE_VALIDATIONSET" in model_training_description.keys()

    if model_training_description["MODEL_TYPE"] in ["UNet_Ico", "UNet_Flat"]:
        assert "DEPTH" in model_training_description.keys()
        assert "BATCH_SIZE" in model_training_description.keys()
        assert "NUM_EPOCHS" in model_training_description.keys()

    # load the full dataset_description
    dataset_description_full = find_and_load_dataset_description(
        base_folder, dataset_description, use_prints=use_prints)
    dataset = find_and_load_dataset(base_folder, dataset_description_full)

    train_predictors = torch.from_numpy(
        dataset["train"]["predictors"].astype(np.float32))
    train_targets = torch.from_numpy(
        dataset["train"]["targets"].astype(np.float32))
    test_predictors = torch.from_numpy(
        dataset["test"]["predictors"].astype(np.float32))
    test_targets = torch.from_numpy(
        dataset["test"]["targets"].astype(np.float32))

    if not dataset_description_full["GRID_TYPE"] == "Ico":
        train_masks = torch.from_numpy(dataset["train"]["masks"].astype(bool))
        test_masks = torch.from_numpy(dataset["test"]["masks"].astype(bool))
        # there are problems with the interpolations if we use nans
        # so if there are any, convert them to a numerical value here - back later
        test_targets = torch.nan_to_num(test_targets, nan=1e20)
        train_targets = torch.nan_to_num(train_targets, nan=1e20)

    # we need to resize images such that they fulfil the divisibility constraint of the UNet.
    # to do so we augment to the next biggest int that fulfils the divisibility constraint.
    if model_training_description["MODEL_TYPE"] == "UNet_Flat":
        divisor = 2 ** model_training_description["DEPTH"]
        h_augment = int(np.ceil(train_predictors.shape[-2]/divisor)*divisor)
        w_augment = int(np.ceil(train_predictors.shape[-1]/divisor)*divisor)
        # print(divisor, h_augment, w_augment, train_predictors.shape[-2], train_predictors.shape[-1])
        resize = T.Resize(size=(h_augment, w_augment))
        train_predictors = resize(train_predictors)
        train_targets = resize(train_targets)
        test_predictors = resize(test_predictors)
        test_targets = resize(test_targets)
        train_masks = resize(train_masks.float())
        test_masks = resize(test_masks.float())
        # for the masks we want to even mask pixels where only part of the image was occluded...
        test_masks = (test_masks != 0)
        train_masks = (train_masks != 0)

    if not dataset_description_full["GRID_TYPE"] == "Ico":
        test_masks = ~test_masks
        train_masks = ~train_masks
        test_targets[~test_masks] = np.nan
        train_targets[~train_masks] = np.nan

    train_predictors, train_targets, test_predictors, test_targets = standardize(train_predictors, train_targets,
                                                                                 test_predictors, test_targets,
                                                                                 dataset_description_full,
                                                                                 model_training_description)

    if not dataset_description_full["GRID_TYPE"] == "Ico":
        test_dataset = data_utils.TensorDataset(
            test_predictors, test_targets, test_masks)
    else:
        test_dataset = data_utils.TensorDataset(test_predictors, test_targets)

    if model_training_description["CREATE_VALIDATIONSET"]:
        assert "SHUFFLE_VALIDATIONSET" in model_training_description.keys()
        if not dataset_description_full["GRID_TYPE"] == "Ico":
            tmp_train_dataset = data_utils.TensorDataset(
                train_predictors, train_targets, train_masks)
        else:
            tmp_train_dataset = data_utils.TensorDataset(
                train_predictors, train_targets)
        l = len(tmp_train_dataset)
        # split dataset into train and validataion set:
        if model_training_description["SHUFFLE_VALIDATIONSET"]:
            train_dataset, validation_dataset = data_utils.random_split(
                tmp_train_dataset, [int(0.9 * l), l - int(0.9 * l)])
        else:
            train_dataset = torch.utils.data.Subset(
                tmp_train_dataset, range(int(0.1 * l), l))
            # Use first 10% as valiationset
            validation_dataset = torch.utils.data.Subset(
                tmp_train_dataset, range(int(0.1 * l)))

        if model_training_description["MODEL_TYPE"] in ["UNet_Flat", "UNet_Ico"]:
            train_loader = data_utils.DataLoader(train_dataset, batch_size=model_training_description["BATCH_SIZE"],
                                                 shuffle=True)
            validation_loader = data_utils.DataLoader(validation_dataset,
                                                      batch_size=model_training_description["BATCH_SIZE"], shuffle=True)
            test_loader = data_utils.DataLoader(test_dataset, batch_size=model_training_description["BATCH_SIZE"],
                                                shuffle=False)
            return train_loader, validation_loader, test_loader, train_dataset, validation_dataset, test_dataset

        elif model_training_description["MODEL_TYPE"] in ["LinReg_Pixelwise", "RandomForest_Pixelwise", "PCA_Flat", "PCA_Ico"]:
            return train_dataset, validation_dataset, test_dataset

        else:
            raise NotImplementedError("Specified model type not implemented")

    else:
        if not dataset_description_full["GRID_TYPE"] == "Ico":
            train_dataset = data_utils.TensorDataset(
                train_predictors, train_targets, train_masks)
        else:
            train_dataset = data_utils.TensorDataset(
                train_predictors, train_targets)

        if model_training_description["MODEL_TYPE"] in ["UNet_Flat", "UNet_Ico"]:
            train_loader = data_utils.DataLoader(train_dataset,
                                                 batch_size=model_training_description["BATCH_SIZE"], shuffle=True)
            test_loader = data_utils.DataLoader(test_dataset,
                                                batch_size=model_training_description["BATCH_SIZE"], shuffle=False)
            return train_loader, test_loader, train_dataset, test_dataset

        elif model_training_description["MODEL_TYPE"] in ["LinReg_Pixelwise", "RandomForest_Pixelwise", "PCA_Flat", "PCA_Ico"]:
            return train_dataset, test_dataset

        else:
            raise NotImplementedError("Specified model type not implemented")


def standardize(train_predictors, train_targets, test_predictors, test_targets, dataset_description, model_training_description):
    """
    Standardize the data with the procedures selected in model_training_description.

    @param train_predictors: Unstandardized train_predictors
    @param train_targets: Unstandardized test_targets
    @param test_predictors: Unstandardized test_predictors
    @param test_targets: Unstandardized test_targets
    @param dataset_description: Parameters of the dataset
    @param model_training_description: Parameters of model and training
    @return: Rescaled versions of train_predictors, train_targets, test_predictors, test_targets
    """
    n_predictors = train_predictors.shape[1]
    n_targets = train_targets.shape[1]

    # assert that standardize mode has one element for each variable.
    assert len(model_training_description["S_MODE_PREDICTORS"]) == n_predictors
    assert len(model_training_description["S_MODE_TARGETS"]) == n_targets
    assert all(
        [mode in ["None", "Pixelwise", "Global_mean_pixelwise_std", "Pixelwise_mean_global_std", "Global"] for mode in
         model_training_description["S_MODE_PREDICTORS"]])
    assert all(
        [mode in ["None", "Pixelwise", "Global_mean_pixelwise_std", "Pixelwise_mean_global_std", "Global"] for mode in
         model_training_description["S_MODE_TARGETS"]])
    # predictors:
    for i, mode in enumerate(model_training_description["S_MODE_PREDICTORS"]):
        if mode == "Global":  # Global normalization: Use same standard deviation for each pixel
            mean = torch.mean(
                train_predictors[:, i, ...], dim=(0, 1, 2), keepdim=True)
            std = torch.mean(torch.std(
                train_predictors[:, i, ...], dim=0, keepdim=True), dim=(1, 2), keepdim=True)
            std[std == 0] = 1  # avoid dividing by zero

        elif mode == "Global_mean_local_std":  # Subtract the global mean, but divide by local standard deviation
            mean = torch.mean(
                train_predictors[:, i, ...], dim=(0, 1, 2), keepdim=True)
            std = torch.std(train_predictors[:, i, ...], dim=0, keepdim=True)
            std[std == 0] = 1  # avoid dividing by zero

        elif mode == "Pixelwise_mean_global_std":  # Subtract the global mean, but divide by local standard deviation
            mean = torch.mean(train_predictors[:, i, ...], dim=0, keepdim=True)
            std = torch.mean(torch.std(
                train_predictors[:, i, ...], dim=0, keepdim=True), dim=(1, 2), keepdim=True)
            std[std == 0] = 1  # avoid dividing by zero

        elif mode == "Pixelwise":  # Subtract pixelwise mean and divide each pixel by its own standard deviation
            mean = torch.mean(train_predictors[:, i, ...], dim=0, keepdim=True)
            std = torch.std(train_predictors[:, i, ...], dim=0, keepdim=True)
            std[std == 0] = 1  # avoid dividing by zero

        train_predictors[:, i, ...] = (
            train_predictors[:, i, ...] - mean) / std
        test_predictors[:, i, ...] = (test_predictors[:, i, ...] - mean) / std
    # targets:
    for i, mode in enumerate(model_training_description["S_MODE_TARGETS"]):
        if mode == "Global":  # Global normalization: Use same standard deviation for each pixel
            mean = np.nanmean(
                train_targets[:, i, ...], axis=(0, 1, 2), keepdims=True)
            std = torch.mean(np.nanstd(
                train_targets[:, i, ...], axis=0, keepdims=True), dim=(1, 2), keepdim=True)
            std[std == 0] = 1  # avoid dividing by zero

        elif mode == "Global_mean_local_std":  # Subtract the global mean, but divide by local standard deviation
            mean = np.nanmean(
                train_targets[:, i, ...], axis=(0, 1, 2), keepdims=True)
            std = np.nanstd(train_targets[:, i, ...], axis=0, keepdims=True)
            std[std == 0] = 1  # avoid dividing by zero

        elif mode == "Pixelwise_mean_global_std":  # Subtract the local mean, but divide by global standard deviation
            mean = np.nanmean(train_targets[:, i, ...], axis=0, keepdims=True)
            std = torch.mean(np.nanstd(
                train_targets[:, i, ...], axis=0, keepdims=True), dim=(1, 2), keepdim=True)
            std[std == 0] = 1  # avoid dividing by zero

        elif mode == "Pixelwise":  # Subtract pixelwise mean and ivide each pixel by its own standard deviation
            mean = np.nanmean(train_targets[:, i, ...], axis=0, keepdims=True)
            std = np.nanstd(train_targets[:, i, ...], axis=0, keepdims=True)
            std[std == 0] = 1  # avoid dividing by zero

        train_targets[:, i, ...] = (train_targets[:, i, ...] - mean) / std
        test_targets[:, i, ...] = (test_targets[:, i, ...] - mean) / std

    if model_training_description["MODEL_TYPE"] == "UNet_Ico":
        train_predictors = torch.unsqueeze(train_predictors, dim=2)
        test_predictors = torch.unsqueeze(test_predictors, dim=2)

    if not dataset_description["GRID_TYPE"] == "Ico":
        test_targets = torch.nan_to_num(test_targets, nan=1e20)
        train_targets = torch.nan_to_num(train_targets, nan=1e20)

    return train_predictors, train_targets, test_predictors, test_targets


def train_global_model(X_train, Y_train):
    from sklearn.linear_model import LinearRegression
    """get the trained model"""
    regressor = LinearRegression().fit(X_train, Y_train)
    return regressor


def train_lasso(X_train, Y_train):
    """get the trained LASSO model"""
    from sklearn.linear_model import MultiTaskLassoCV
    lasso = MultiTaskLassoCV().fit(X_train, Y_train)
    return lasso


def train_onedim_lasso(X_train, Y_train):
    """get trained LASSO with one-dimensional output"""
    from sklearn.linear_model import LassoCV
    lasso = LassoCV().fit(X_train, Y_train)
    return lasso


def train_pca(dataset_description, model_training_description, base_folder):
    """
    Train PCA and regression model on the training data. In opposition to the version in the Jonathan_PCA_methods notebook,
    we don't rescale here seperately, rescaling is already done in the dataloader.
    Assume inputdata of shape (n_timesteps, n_variables, n_lat, n_lon).
    """
    dataset_description = find_and_load_dataset_description(
        base_folder, dataset_description)
    assert "N_PC_PREDICTORS" in model_training_description.keys()
    assert "N_PC_TARGETS" in model_training_description.keys()
    assert "REGTYPE" in model_training_description.keys()
    assert dataset_description["TIMESCALE"] == "YEARLY"

    if not model_training_description["CREATE_VALIDATIONSET"]:
        train_ds, _ = load_data(dataset_description,
                                model_training_description, base_folder)
    else:
        train_ds, _, _ = load_data(
            dataset_description, model_training_description, base_folder)
    x_tr = train_ds[:][0].numpy()
    y_tr = train_ds[:][1].numpy()
    if dataset_description["GRID_TYPE"] == "Flat":
        masks_tr = train_ds[:][2].numpy()
        assert (masks_tr == True).all(
        ), "No missing values allowed in target variables when training PCA methods."

    x_train = x_tr.reshape(x_tr.shape[0], -1)
    y_train = y_tr.reshape(y_tr.shape[0], -1)
    # PCA
    pca = PCA(n_components=model_training_description["N_PC_PREDICTORS"])
    principal_components = pca.fit_transform(x_train)
    pca_targets = PCA(n_components=model_training_description["N_PC_TARGETS"])
    principal_components_targets = pca_targets.fit_transform(y_train)
    # Get the model
    if model_training_description["REGTYPE"] == 'lasso':
        model = train_lasso(principal_components, principal_components_targets)
    elif model_training_description["REGTYPE"] == 'linreg':
        model = train_global_model(
            principal_components, principal_components_targets)
    else:
        raise NotImplementedError(
            "This regression model is currently not implemented.")
    return pca, pca_targets, model


def weighted_mse_loss(output, target, weights):
    """
    compute weighted mean squared error loss. Use the cell-size as weight.
    Inputs should have shape (batchsize, adjusted_height, adjusted_width)
    """
    # print("output",output.shape, "target",target.shape, "weights", weights.shape)
    return (weights * (output - target) ** 2).mean()


def masked_weighted_mse_loss(output, target, masks, weights):
    """
    compute weighted mean squared error loss. Use the cell-size as weight.
    Inputs should have shape (batchsize, adjusted_height, adjusted_width). Masks out missing values as given in masks.
    """
    # print("output",output.shape, "target",target.shape, "weights", weights.shape)
    return (weights * (output - target) ** 2)[masks].mean()


def get_masked_area_weighted_mse_loss(dataset_description, model_training_description):
    assert "GRID_SHAPE" in dataset_description.keys()
    assert "DEVICE" in model_training_description.keys()
    assert dataset_description["GRID_TYPE"] == "Flat"

    divisor = 2 ** model_training_description["DEPTH"]
    width = dataset_description["GRID_SHAPE"][1]
    height = dataset_description["GRID_SHAPE"][0]
    lat_max = dataset_description["LATITUDES"][0]
    divisor = 2 ** model_training_description["DEPTH"]
    adjusted_height = int(np.ceil(height / divisor) * divisor)
    adjusted_width = int(np.ceil(width / divisor) * divisor)

    area_weights = torch.cos(
        torch.linspace(-lat_max, lat_max, adjusted_height) * (2 * np.pi) / 360)
    area_weights = area_weights.view(1, -1, 1).repeat(1, 1, adjusted_width)
    area_weights = (adjusted_width * adjusted_height /
                    torch.sum(area_weights)) * area_weights
    area_weights = area_weights.to(model_training_description["DEVICE"])
    return partial(masked_weighted_mse_loss, weights=area_weights)


def get_masked_mse_loss(dataset_description, model_training_description):
    assert "GRID_SHAPE" in dataset_description.keys()
    assert "DEVICE" in model_training_description.keys()
    assert dataset_description["GRID_TYPE"] == "Flat"

    divisor = 2 ** model_training_description["DEPTH"]
    width = dataset_description["GRID_SHAPE"][1]
    height = dataset_description["GRID_SHAPE"][0]
    adjusted_height = int(np.ceil(height / divisor) * divisor)
    adjusted_width = int(np.ceil(width / divisor) * divisor)
    const_weights = torch.ones((1, adjusted_height, adjusted_width))
    const_weights = const_weights.to(model_training_description["DEVICE"])
    return partial(masked_weighted_mse_loss, weights=const_weights)


def get_area_weighted_mse_loss(dataset_description, model_training_description):
    assert "GRID_SHAPE" in dataset_description.keys()
    assert "DEVICE" in model_training_description.keys()

    divisor = 2 ** model_training_description["DEPTH"]
    width = dataset_description["GRID_SHAPE"][1]
    height = dataset_description["GRID_SHAPE"][0]
    lat_max = dataset_description["LATITUDES"][0]
    adjusted_height = int(np.ceil(height / divisor) * divisor)
    adjusted_width = int(np.ceil(width / divisor) * divisor)
    area_weights = torch.cos(
        torch.linspace(-lat_max, lat_max, adjusted_height) * (2 * np.pi) / 360)
    area_weights = area_weights.view(1, -1, 1).repeat(1, 1, adjusted_width)
    area_weights = (adjusted_width * adjusted_height /
                    torch.sum(area_weights)) * area_weights
    area_weights = area_weights.to(model_training_description["DEVICE"])
    return partial(weighted_mse_loss, weights=area_weights)


def train_unet(dataset_description, model_training_description, base_folder, use_tensorboard=False):
    dataset_description = find_and_load_dataset_description(
        base_folder, dataset_description)
    assert model_training_description["MODEL_TYPE"] in [
        "UNet_Flat", "UNet_Ico"]
    assert "DEPTH" in model_training_description.keys()
    assert "IN_CHANNELS" in model_training_description.keys()
    assert "CHANNELS_FIRST_CONV" in model_training_description.keys()
    assert "OUT_CHANNELS" in model_training_description.keys()
    assert "FMAPS" in model_training_description.keys()
    assert "ACTIVATION" in model_training_description.keys()
    assert "NORMALIZATION" in model_training_description.keys()
    assert "LOSS" in model_training_description.keys()
    assert "DEVICE" in model_training_description.keys()
    assert "OPTIMIZER" in model_training_description.keys()

    if not dataset_description["GRID_TYPE"] == "Ico":
        assert model_training_description["LOSS"] in [
            "Masked_MSELoss", "Masked_AreaWeightedMSELoss"]
    if model_training_description["MODEL_TYPE"] == "UNet_Flat":
        assert "USE_CYLINDRICAL_PADDING" in model_training_description.keys()
        assert "USE_COORD_CONV" in model_training_description.keys()
        assert dataset_description["GRID_TYPE"] == "Flat"
        assert model_training_description["NORMALIZATION"] != IcoBatchNorm2d
    elif model_training_description["MODEL_TYPE"] == "UNet_Ico":
        assert model_training_description["LOSS"] == "MSELoss"
        assert dataset_description["GRID_TYPE"] == "Ico"
        assert model_training_description["NORMALIZATION"] != torch.nn.BatchNorm2d
    else:
        raise NotImplementedError(
            "Only UNet_Ico and UNet_Flat implemented in this method")

    if use_tensorboard:
        s1 = util.create_hash_from_description(dataset_description)
        s2 = util.create_hash_from_description(model_training_description)
        s3 = "_log"
        folder_name = os.path.join(base_folder, s1 + s2 + s3)
        print("To open tensorboard, run tensorboard --logdir={}".format(folder_name))
        writer = SummaryWriter(folder_name)

    # initialize model, loss and optimizer and move to device
    if model_training_description["MODEL_TYPE"] == "UNet_Flat":
        unet = UNet(depth=model_training_description["DEPTH"],
                    in_channels=model_training_description["IN_CHANNELS"],
                    channels_first_conv=model_training_description["CHANNELS_FIRST_CONV"],
                    use_cylindrical_padding=model_training_description["USE_CYLINDRICAL_PADDING"],
                    use_coord_conv=model_training_description["USE_COORD_CONV"],
                    out_channels=model_training_description["OUT_CHANNELS"],
                    fmaps=model_training_description["FMAPS"],
                    activation=model_training_description["ACTIVATION"],
                    norm_type=model_training_description["NORMALIZATION"])
    else:
        unet = IcoUNet(in_res=dataset_description["RESOLUTION"],
                       depth=model_training_description["DEPTH"],
                       in_channels=model_training_description["IN_CHANNELS"],
                       channels_first_conv=model_training_description["CHANNELS_FIRST_CONV"],
                       out_channels=model_training_description["OUT_CHANNELS"],
                       fmaps=model_training_description["FMAPS"],
                       activation=model_training_description["ACTIVATION"],
                       norm_type=model_training_description["NORMALIZATION"])

    # translate the options that are stored in the model_training_description
    optimizer_dict = {"Adam": torch.optim.Adam(unet.parameters())}
    loss_dict = {}
    if dataset_description["GRID_TYPE"] == "Flat":
        loss_dict["Masked_AreaWeightedMSELoss"] = get_masked_area_weighted_mse_loss(dataset_description,
                                                                                    model_training_description)
        loss_dict["Masked_MSELoss"] = get_masked_mse_loss(
            dataset_description, model_training_description)
    elif dataset_description["GRID_TYPE"] == "Ico":
        loss_dict["MSELoss"] = nn.MSELoss()
    else:
        raise NotImplementedError("Invalid grid type")

    criterion = loss_dict[model_training_description["LOSS"]]
    optimizer = optimizer_dict[model_training_description["OPTIMIZER"]]
    unet.to(model_training_description["DEVICE"])
    # criterion = criterion.to(device)

    # save number of parameters in the description file
    # model_training_description["#params"] = sum(x.numel() for x in unet.parameters())
    if type(model_training_description["NUM_EPOCHS"]) == int:
        assert model_training_description["CREATE_VALIDATIONSET"] is False
        train_loader, test_loader, train_dataset, test_dataset = load_data(dataset_description,
                                                                           model_training_description,
                                                                           base_folder)
        # start training
        print("Starting training")
        for epoch in range(model_training_description["NUM_EPOCHS"]):
            running_loss = 0
            n_batches = 0
            for i, data in enumerate(train_loader):
                unet.train()
                if dataset_description["GRID_TYPE"] == "Ico":
                    predictors, targets = data
                    predictors = predictors.to(
                        model_training_description["DEVICE"])
                    targets = targets.to(model_training_description["DEVICE"])
                elif dataset_description["GRID_TYPE"] == "Flat":
                    predictors, targets, masks = data
                    predictors = predictors.to(
                        model_training_description["DEVICE"])
                    targets = targets.to(model_training_description["DEVICE"])
                    masks = masks.to(model_training_description["DEVICE"])
                else:
                    raise NotImplementedError("Invalid grid type")
                optimizer.zero_grad()
                outputs = unet(predictors)

                if dataset_description["GRID_TYPE"] == "Ico":
                    loss = criterion(outputs, targets)
                elif dataset_description["GRID_TYPE"] == "Flat":
                    loss = criterion(outputs, targets, masks)
                else:
                    raise NotImplementedError("Invalid grid type")

                loss.backward()
                running_loss += loss.item()
                n_batches += 1

                optimizer.step()

                print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                    epoch + 1, model_training_description["NUM_EPOCHS"], i + 1,
                    len(
                        train_dataset) // model_training_description["BATCH_SIZE"],
                    loss.item()), end="")
            if use_tensorboard:
                writer.add_scalar(
                    'training loss', running_loss/n_batches, epoch)
            print("")

            total_MSE = 0
            n_batches = 0
            for data in test_loader:
                unet.eval()
                if dataset_description["GRID_TYPE"] == "Ico":
                    predictors, targets = data
                    with torch.no_grad():
                        predictors = predictors.to(
                            model_training_description["DEVICE"])
                        targets = targets.to(
                            model_training_description["DEVICE"])
                        outputs = unet(predictors)
                        total_MSE += criterion(outputs, targets)
                        n_batches += 1
                elif dataset_description["GRID_TYPE"] == "Flat":
                    predictors, targets, masks = data
                    with torch.no_grad():
                        predictors = predictors.to(
                            model_training_description["DEVICE"])
                        targets = targets.to(
                            model_training_description["DEVICE"])
                        masks = masks.to(model_training_description["DEVICE"])
                        outputs = unet(predictors)
                        total_MSE += criterion(outputs, targets, masks)
                        n_batches += 1
                else:
                    raise NotImplementedError("Invalid grid type")
            print('Test MSE: {0}'.format(total_MSE / n_batches))
            if use_tensorboard:
                writer.add_scalar('test loss', total_MSE, epoch)

        return unet

    elif model_training_description["NUM_EPOCHS"] == "early_stopping":
        assert model_training_description["CREATE_VALIDATIONSET"] is True
        assert "PATIENCE" in model_training_description.keys()
        assert "SHUFFLE_VALIDATIONSET" in model_training_description.keys()
        train_loader, validation_loader, test_loader, train_dataset, validation_dataset, test_dataset = load_data(dataset_description,
                                                                                                                  model_training_description,
                                                                                                                  base_folder)
        increase_counter = 0
        best_validation_mse = float("inf")
        # start training
        print("Starting training")
        epoch = 0

        while increase_counter <= model_training_description["PATIENCE"]:
            running_loss = 0
            n_batches = 0
            for i, data in enumerate(train_loader):
                unet.train()
                if dataset_description["GRID_TYPE"] == "Ico":
                    predictors, targets = data
                    predictors = predictors.to(
                        model_training_description["DEVICE"])
                    targets = targets.to(model_training_description["DEVICE"])
                elif dataset_description["GRID_TYPE"] == "Flat":
                    predictors, targets, masks = data
                    predictors = predictors.to(
                        model_training_description["DEVICE"])
                    targets = targets.to(model_training_description["DEVICE"])
                    masks = masks.to(model_training_description["DEVICE"])
                else:
                    raise NotImplementedError("Invalid grid type")
                optimizer.zero_grad()
                outputs = unet(predictors)

                if dataset_description["GRID_TYPE"] == "Ico":
                    loss = criterion(outputs, targets)
                elif dataset_description["GRID_TYPE"] == "Flat":
                    loss = criterion(outputs, targets, masks)
                else:
                    raise NotImplementedError("Invalid grid type")

                loss.backward()

                running_loss += loss.item()
                n_batches += 1

                optimizer.step()
                if i % 30 == 0:
                    print('\rEpoch [{0}], Iter [{1}/{2}] Loss: {3:.4f}'.format(
                        epoch + 1, i +
                        1, len(
                            train_dataset) // model_training_description["BATCH_SIZE"],
                        loss.item()), end="")

            if use_tensorboard:
                writer.add_scalar(
                    'training loss', running_loss / n_batches, epoch)
            print("")

            total_MSE = 0
            n_batches = 0
            for data in validation_loader:
                unet.eval()
                if dataset_description["GRID_TYPE"] == "Ico":
                    predictors, targets = data
                    with torch.no_grad():
                        predictors = predictors.to(
                            model_training_description["DEVICE"])
                        targets = targets.to(
                            model_training_description["DEVICE"])
                        outputs = unet(predictors)
                        total_MSE += criterion(outputs, targets)
                        n_batches += 1

                elif dataset_description["GRID_TYPE"] == "Flat":
                    predictors, targets, masks = data
                    with torch.no_grad():
                        predictors = predictors.to(
                            model_training_description["DEVICE"])
                        targets = targets.to(
                            model_training_description["DEVICE"])
                        masks = masks.to(model_training_description["DEVICE"])
                        outputs = unet(predictors)
                        total_MSE += criterion(outputs, targets, masks)
                        n_batches += 1
                else:
                    raise NotImplementedError("Timescale not implemented")

            validation_mse = total_MSE / n_batches
            if use_tensorboard:
                writer.add_scalar('validation loss', validation_mse, epoch)
            # print('Validation MSE: {0}'.format(validation_mse))
            if validation_mse < best_validation_mse:
                increase_counter = 0
                best_validation_mse = validation_mse
                torch.save(unet, os.path.join(base_folder, "cp.pt"))
            else:
                increase_counter += 1
            epoch += 1
            # print("counter: {}".format(increase_counter), "best mse: {:.4f}".format(best_validation_mse))

        # load the checkpointed file
        unet = torch.load(os.path.join(base_folder, "cp.pt"))
        total_MSE = 0
        n_batches = 0
        for data in test_loader:
            unet.eval()

            if dataset_description["GRID_TYPE"] == "Ico":
                predictors, targets = data
                with torch.no_grad():
                    predictors = predictors.to(
                        model_training_description["DEVICE"])
                    targets = targets.to(model_training_description["DEVICE"])
                    outputs = unet(predictors)
                    total_MSE += criterion(outputs, targets)
                    n_batches += 1

            elif dataset_description["GRID_TYPE"] == "Flat":
                predictors, targets, masks = data
                with torch.no_grad():
                    predictors = predictors.to(
                        model_training_description["DEVICE"])
                    targets = targets.to(model_training_description["DEVICE"])
                    masks = masks.to(model_training_description["DEVICE"])
                    outputs = unet(predictors)
                    total_MSE += criterion(outputs, targets, masks)
                    n_batches += 1
            else:
                raise NotImplementedError("Timescale not implemented")
        test_mse = total_MSE / n_batches
        print('Test MSE: {0}'.format(test_mse))
        if use_tensorboard:
            writer.add_scalar('test loss', test_mse, epoch)
        return unet
    else:
        raise NotImplementedError(
            "Only early stopping and int number of epochs implemented.")


def train_linreg_pixelwise(dataset_description, model_training_description, base_folder):
    """
    Train a linear regression models for each grid box, using climate variables from the same grid box.

    @param dataset_description: Details on data set creation.
    @param model_training_description: Details on model and training procedure
    @param base_folder: Folder from which to store results in.
    @return: List of lists of trained models.
    """
    dataset_description = find_and_load_dataset_description(
        base_folder, dataset_description)
    assert model_training_description["MODEL_TYPE"] == "LinReg_Pixelwise"
    assert dataset_description["GRID_TYPE"] == "Flat"
    assert dataset_description["TIMESCALE"] == "YEARLY"
    models = []

    if not model_training_description["CREATE_VALIDATIONSET"]:
        train_ds, _ = load_data(dataset_description,
                                model_training_description, base_folder)
    else:
        train_ds, _, _ = load_data(
            dataset_description, model_training_description, base_folder)

    x_tr = train_ds[:][0].numpy()
    y_tr = train_ds[:][1].numpy()
    masks_tr = train_ds[:][2].numpy()
    assert (masks_tr == True).all(
    ), "No missing values allowed in target variables when training Linreg baseline."

    for i in range(x_tr.shape[-2]):
        models.append([])
        for j in range(x_tr.shape[-1]):
            model = LinearRegression().fit(x_tr[..., i, j], y_tr[..., i, j])
            models[-1].append(model)
    return models


def train_random_forest_pixelwise(dataset_description, model_training_description, base_folder, verbose=0, n_jobs=1):
    """
    Train a linear regression models for each grid box, using climate variables from the same grid box.
    Assumes that the data is loaded in same format.

    @param dataset_description: Details on data set creation.
    @param model_training_description: Details on model and training procedure
    @param base_folder: Folder from which to store results in.
    @return: List of lists of trained models.
    """
    dataset_description = find_and_load_dataset_description(
        base_folder, dataset_description)
    assert model_training_description["MODEL_TYPE"] == "RandomForest_Pixelwise"
    assert dataset_description["GRID_TYPE"] == "Flat"
    assert dataset_description["TIMESCALE"] == "YEARLY"
    if not model_training_description["CREATE_VALIDATIONSET"]:
        train_ds, _ = load_data(dataset_description,
                                model_training_description, base_folder)
    else:
        train_ds, _, _ = load_data(
            dataset_description, model_training_description, base_folder)

    x_tr = train_ds[:][0].numpy()
    y_tr = train_ds[:][1].numpy()
    masks_tr = train_ds[:][2].numpy()
    assert (masks_tr == True).all(
    ), "No missing values allowed in target variables when training Random forest baseline."

    # append coordinates to predictor variables, lon as cos(lon), sin(lon)
    x_tr = append_coords(x_tr)

    # combine information from all pixels into one training set incorporating all grid boxes.
    x_tr = x_tr.transpose(0, 2, 3, 1).reshape(-1, x_tr.shape[1])
    y_tr = y_tr.transpose(0, 2, 3, 1).reshape(-1, y_tr.shape[1])

    model = RandomForestRegressor(
        verbose=verbose, n_jobs=n_jobs).fit(x_tr, np.squeeze(y_tr))
    return model


def append_coords(data):
    """
    Append coordinates to variables. Input data is assumed to be of shape (n_timesteps, n_predictor_variables, lat, lon)
    @param data: Input data
    @return: Input data, with coordinates appended to predictor variables.
    """

    lat_size = data.shape[-2]
    lon_size = data.shape[-1]
    lats = np.linspace(-1, 1, lat_size)
    lons = np.linspace(-np.pi, np.pi, lon_size)
    lons_sin = np.sin(lons)
    lons_cos = np.cos(lons)

    # reshape:
    lats = np.repeat(lats[:, np.newaxis], repeats=lon_size, axis=-1)
    lons_sin = np.repeat(lons_sin[np.newaxis, :], repeats=lat_size, axis=0)
    lons_cos = np.repeat(lons_cos[np.newaxis, :], repeats=lat_size, axis=0)
    lats = np.repeat(lats[np.newaxis, np.newaxis, :, :],
                     repeats=data.shape[0], axis=0)
    lons_sin = np.repeat(
        lons_sin[np.newaxis, np.newaxis, :, :], repeats=data.shape[0], axis=0)
    lons_cos = np.repeat(
        lons_cos[np.newaxis, np.newaxis, :, :], repeats=data.shape[0], axis=0)

    return np.concatenate((data, lons_sin, lons_cos, lats), axis=1)
