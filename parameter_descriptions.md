# Specifying keywoards and valid parameters that can be used to create datasets and configure ML-Methods and training:

During data set creation, set up of the ML methods and the training, choices can be made. The following table gives a list of possible parameters and valid configurations.

## Datasets (dataset_description):

| Name                       | Purpose                                                                                                             |
|----------------------------|---------------------------------------------------------------------------------------------------------------------|
| DSET_NB                    | A number assigne to the dataset. Used to create multple datasets with the same configuration                        |
| GRID_TYPE                  | "Flat" or "Ico"                                                                                                     |
| CALENDAR                   | Stores the calendar types of the used climate model (filled automatically)                                          |
| T_UNITS                    | Stores the time units used by the climate model (filled automatically)                                              |
| REFERENCE_DATE             | Stores the reference date used by the climate model, i.e. in reference to what date timesteps are stored (filled automatically)                                                                                                                                     |                                                                                                         
| DO_SHUFFLE                 | Whether or not the data should be shuffled before splitting into test and training set (our default: don't shuffle) |
| TEST_FRACTION              | What fraction of the dataset is split of as training set                                                            |
| CLIMATE_MODEL              | Specify which climate model is used: Valid arguments: "iHadCM3", "GISS", "ECHAM5", "isoGSM", "iCESM"                |
| LATITUDES_SLICE            | If using a flat grid,  exclude this number of pixels at bottom and top of latitude, e.g. [1,-1] excludes the first and last latitude value. (flat grid only)                                                                                                                   |
| LATITUDES                  | Array containing the latitudes of the used flat grid (flat grids only)                                              |
| LONGITUDES                 | Array containing the longitudes of the used flat grid (flat grids only)                                             |
| START_YEAR                 | Year with which the valid part of the last millennium run starts                                                    |
| END_YEAR                   | Year with which the valid part of the last millennium run ends                                                      |
| DATASETS_NO_GAPS           | Datasets containing variables of which we require variables to be present at each time step.                        |
| DATASETS_USED              | Used datasets. Names need to match with file names, e.g. "tsurf" to use "tsurf.nc" or "tsurf_yearly.nc"             |
| PREDICTOR_VARIABLES        | Which of the variables are used as predictors, dict containing DATASETS_USED as keys and variable name als values.  |
| TARGET_VARIABLES           | Which of the variables are used as targets, dict containing DATASETS_USED as keys and variable name als values.     |
| TIMESCALE                  | What timescale we want to work on: "yearly" or "monthly"                                                            |
| MONTHS_USED                | If on monthly timescale: one can decide to only use certain months in the dataset, e.g ```[0,1]``` would indicate a dataset only containing January and February timesteps.                                                                                                    |
| MONTHS_USED_IN_PREDICTION  | For each timestep, it is possible to include timesteps from future or past in the emulation: To use the current and the previous months in the prediction, e.g. ```np.sort([0,-1]).tolist()``` or ```[0]``` to only use the current months                                 |
| RESOLUTION                 | When using icosahedral data: Refinement level of the grid (interpolated data on the given resolution required)      |
| INTERPOLATE_CORNERS        | Ico: Whether or not we interpolate the corners pixels of the icosahedron                                            |
| INTERPOLATION              | The type of interpolation used by cdo, usually ```cons1``` (first order conservative), ```NN``` (nearest neighbor) possible to.                                                                                                                                                |
| INDICES_TEST               | Indices of the testset in the dataset (automatically generated)                                                     |
| INDICES_TRAIN              | Indices of the trainingset in the dataset (automatically generated)                                                 |
| SPLIT_YEAR                 | Alternatively to giving a TEST_FRACTION, one can also provide a year, based on which the data will be split into test and training set                                                                                                                                       |
| PRECIP_WEIGHTING           | Whether or not to weight individual months by precipitation amount when creating yearly datasets (only for yearly time scale)                                                                                                                                                    |
|                            |                                                                                                                     |
|                            |                                                                                                                     |

## ML models and training (model_training_description):
| Name                       | Purpose                                                                                                             |
|----------------------------|---------------------------------------------------------------------------------------------------------------------|
| MODELTYPE                  | Type of model we want to use: Valid choices: ```UNet_Flat```, ```UNet_Ico```, ```LinReg_Pixelwise```, ```RandomForest_Pixelwise```, ````PCA_Flat```, ```PCA_Ico```                                                                                       |
| CREATE_VALIDATIONSET       | Whether or not we want to create a validationset. The validation set is split of the training set.                                                                                                                                                                                |
| SHUFFLE_VALIDATIONSET      | Whether to shuffle the training set before creating the validation set or split it off chronologically              |
| DATASET_FOLDER             | Folder that was given as "base folder" when creating the corresponding dataset, i.e. the folder in which the data set folder is stored                                                                                                                                          |
| RUN_NR                     | Number of run with the given configuration of data set, model and training (to run the same configuration multiple times)                                                                                                                                                        |
| BATCHSIZE                  | UNet Batchsize of UNet Models (UNets only)                                                                          |
| DEPTH                      | Depth of UNets Models (UNets only)                                                                                  |
| NUM_EPOCHS                 | Number of epochs for training UNets, can be set to "early_stopping" or an integer val                               |
| PATIENCE                   | If using early stopping, this parameter determines for how many epochs we train without an improvement of the global minimum loss before aborting the training                                                                                                                  |
| S_MODE_PREDICTORS          | Standardization mode for prediction variables: Can be set individually for each variable. Valid choices: ```"None"```, ```"Pixelwise"```, ```"Global_mean_pixelwise_std"```, ```"Pixelwise_mean_global_std"```, ```"Global"```                                            |
| S_MODE_TARGETS             | Standardization mode for target variables. Valid choices: ```"None"```, ```"Pixelwise"```, ```"Global_mean_pixelwise_std"```, ```"Pixelwise_mean_global_std"```, ```"Global"```                                                                                                  |
| N_PC_PREDICTORS            | PCA-regression: Number of principle components for predictor variables                                              |
| N_PC_TARGETS               | PCA-regression: Number of principle components for target variables                                                 |
| REGTYPE                    | PCA-regression: Type of regression model used in the reduced space. Valid choices are ```"linreg"``` and ```"lasso"```                                                                                                                                                            |



