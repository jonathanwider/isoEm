# Specifying choices keywoards and valid parameters that can be used to create datasets and configure ML-Methods and training:

## Datasets:

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
|                            |                                                                                                                     |
|                            |                                                                                                                     |
|                            |                                                                                                                     |

## ML models and training (model_training_description):
| Name                       | Purpose                                                                                                             |
|----------------------------|---------------------------------------------------------------------------------------------------------------------|
| Model-training-description |                                                                                                                     |
| MODELTYPE                  | Type of model we want to use                                                                                        |
| CREATE_VALIDATIONSET       | Whether or not we want to create a validationset                                                                    |
| SHUFFLE_VALIDATIONSET      |                                                                                                                     |
|                            |                                                                                                                     |
| DATASET_FOLDER             |                                                                                                                     |
|                            |                                                                                                                     |
| RUN_NR                     | Number of run in the dataset                                                                                        |
|                            |                                                                                                                     |
| BATCHSIZE                  | Batchsize of UNet Models                                                                                            |
| DEPTH                      | Depth of UNets Models                                                                                               |
| NUM_EPOCHS                 | Number of Epochs for training UNet, can be set to "early_stopping"                                                  |
| PATIENCE                   | If using early stopping, this parameter determines for how many epochs we train without improvement before aborting |
|                            |                                                                                                                     |
| S_MODE_PREDICTORS          | Standardization mode for prediction variables                                                                       |
| S_MODE_TARGETS             | Standardization mode for target variables                                                                           |
|                            |                                                                                                                     |
| N_PC_PREDICTORS            | Number of principle components for predictor variables                                                              |
| N_PC_TARGETS               | Number of principle components for target variables                                                                 |
| REGTYPE                    | Type of according Linear Model                                                                                      |
|                            |                                                                                                                     |
|                            |                                                                                                                     |


