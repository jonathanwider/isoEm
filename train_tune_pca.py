from train import train_pca, load_data, find_and_load_dataset_description
from predict import predict_pca
from evaluate import get_weighted_average, get_r2

import numpy as np


def train_tune_pca(dataset_description, model_training_description, base_folder, n_pc_in=None, n_pc_out=None):

    # assert dataset_description["GRID_TYPE"] == "Flat"
    assert model_training_description["CREATE_VALIDATIONSET"] is True
    assert model_training_description["SHUFFLE_VALIDATIONSET"] is False
    # assert model_training_description["MODEL_TYPE"] == "PCA_Flat"

    assert "N_PC_PREDICTORS" not in model_training_description.keys()
    assert "N_PC_TARGETS" not in model_training_description.keys()
    full_dataset_description = find_and_load_dataset_description(base_folder, dataset_description)

    _, validation_ds, test_ds = load_data(full_dataset_description, model_training_description, base_folder)

    n_in_max = None
    n_out_max = None
    r2_max = - np.inf

    for i, (n_in, n_out) in enumerate(zip(n_pc_in, n_pc_out)):
        d_tmp = {**model_training_description,
                 "N_PC_PREDICTORS": n_in,
                 "N_PC_TARGETS": n_out}
        pca, pca_targets, model = train_pca(full_dataset_description, d_tmp, base_folder)
        predictions = predict_pca(validation_ds[:][0], pca, pca_targets, model)
        r2 = get_r2(predictions, validation_ds[:][1])
        if dataset_description["GRID_TYPE"] == "Flat":
            r2_mean = get_weighted_average(r2, full_dataset_description)
        elif dataset_description["GRID_TYPE"] == "Ico":
            r2_mean = np.mean(r2, axis=(1,2))
        else:
            raise NotImplementedError("Invalid grid type")
        if r2_mean > r2_max:
            n_in_max = n_in
            n_out_max = n_out
            r2_max = r2_mean
        print("[{}/{}], N_PC_IN={}, N_PC_OUT={}, R2={}. Best: {}".format(i+1, len(n_pc_in), n_in, n_out, r2_mean, r2_max), end="\r")
    print("Best results: N_PC_IN: {} N_PC_OUT: {}, R2_mean, validationset: {}".format(n_in_max, n_out_max, r2_max))
    print("Retrain including validation set.")

    model_training_description["CREATE_VALIDATIONSET"] = False
    d_tmp = {**model_training_description,
             "N_PC_PREDICTORS": n_in_max,
             "N_PC_TARGETS": n_out_max}
    pca, pca_targets, model = train_pca(full_dataset_description, d_tmp, base_folder)

    # do predictions on test set.
    predictions = predict_pca(test_ds[:][0], pca, pca_targets, model)
    r2 = get_r2(predictions, test_ds[:][1])
    r2_mean = get_weighted_average(r2, full_dataset_description)

    print("Result on test set: {}".format(r2_mean))

    model_training_description["CREATE_VALIDATIONSET"] = True
    model_training_description["N_PC_TARGETS"] = n_in_max
    model_training_description["N_PC_PREDICTORS"] = n_out_max

    return pca, pca_targets, model