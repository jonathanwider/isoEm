# Specifying choices keywoards and valid parameters that can be used to create datasets and configure ML-Methods and training:

## Datasets:

| Name                | Purpose                                                                                                             |
|----------------------------|---------------------------------------------------------------------------------------------------------------------|
| DSET_NB                    | A number assigne to the dataset. Used to create multple datasets with the same configuration                        |
| GRID_TYPE                  | "Flat" or "Ico"                                                                                                     |
|                            |                                                                                                                     |
| CALENDAR                   | Stores the calendar types of the used climate model (filled automatically)                                          |
| T_UNITS                    | Stores the time units used by the climate model (filled automatically)                                              |
| REFERENCE_DATE             | Stores the reference date used by the climate model, i.e. in reference to what date timesteps are stored (filled automatically)                                                                                                                                     |                                                                                                         
| DO_SHUFFLE                 | Whether timesteps are shuffled or not                                                                               |
| TEST_FRACTION              | Fraction of dataset used for testing                                                                                |
|                            |                                                                                                                     |
| CLIMATE_MODEL              | Specify which climate model is used                                                                                 |
| LATITUDES_SLICE            | If using a flat grid,  exclude this number of pixels at bottom and top of latitude.                                 |
| LATITUDES                  | Array of latitudes in the dataset                                                                                   |
| LONGITUDES                 |                                                                                                                     |
|                            |                                                                                                                     |
| START_YEAR                 | Year with which the valid part of the last millennium run starts                                                    |
| END_YEAR                   | Year with which the valid part of the last millennium run ends                                                      |
| DATASETS_NO_GAPS           | Datasets containing variables of which we require variables to be present at each time step.                        |
| DATASETS_USED              | Names need to match file names.                                                                                     |
| PREDICTOR_VARIABLES        | Which of the variables are used as predictors, dict containing DATASETS_USED as keys and variable name als values.  |
| TARGET_VARIABLES           | Which of the variables are used as targets, dict containing DATASETS_USED as keys and variable name als values.     |
|                            |                                                                                                                     |
| TIMESCALE                  | What timescale we want to work on.                                                                                  |
| MONTHS_USED                | Which months of the year we want to use in the prediction                                                           |
| MONTHS_USED_IN_PREDICTION  | Lagged months used in prediction, 0 excluded automatically.                                                         |
|                            |                                                                                                                     |
| RESOLUTION                 | Ico: Refinement level                                                                                               |
| INTERPOLATE_CORNERS        | Ico: Whether or not we interpolate corners                                                                          |
| INTERPOLATION              | The type of interpolation used by cdo                                                                               |
|                            |                                                                                                                     |
| INDICES_TEST               | Indices of the testset in the dataset (automatically generated)                                                     |
| INDICES_TRAIN              | Indices of the trainingset in the dataset (automatically generated)                                                 |
|                            |                                                                                                                     |
|                            |                                                                                                                     |
|                            |                                                                                                                     |
|                            |                                                                                                                     |
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


