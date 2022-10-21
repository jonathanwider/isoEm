# Basic strategy:
- Simplify by reducing number of jupyter notebooks and moving to script files.
- Don't create folders if not necessary.
- Save dataset-description within model-description.
- Only use the condition check implemented yesterday.

## Datasets: Which entries do we need?

| name                       | Purpose                                                                                                             |
|----------------------------|---------------------------------------------------------------------------------------------------------------------|
| DSET_NB                    | Create multiple data sets with identical arguments                                                                  |
| GRID_TYPE                  | Flat or icosahedral                                                                                                 |
|                            |                                                                                                                     |
| CALENDAR                   | Calendar used by the climate model (filled automatically)                                                           |
| T_UNITS                    | Time units used by the climate model (filled automatically)                                                         |
| REFERENCE_DATE             | Reference date used by the climate model (filled automatically)                                                     |
|                            |                                                                                                                     |
|                            |                                                                                                                     |
| DO_SHUFFLE                 | Whether timesteps are shuffled or not                                                                               |
| TEST_FRACTION              | Fraction of dataset used for testing                                                                                |
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

todos today:
- [ ] Fix memory issues
- [ ] Make sure implementations reproduce results from thesis

TODOs:
- [x] Monthly dataset
- [x] Precip weighted dataset
- [x] UNet, train
- [x] UNet, predict
- [x] UNet spherical, train
- [x] UNet spherical, predict
- [x] Other Baselines
    - [x] Test Random forest -> super slow.
    - [x] Validation set for selecting n_pc for PCA regression
- [x] Compare notebooks
- [x] Train Monthly network
    
- [ ] Write Plotting functions from compare notebooks
- [ ] fix problem that makes it possible to run runs with the same config twice
- [x] Test Ico Data loading
- [ ] Precip weighting, monthly in compare notebooks
- [ ] make crossprediction possible
    - [x] Load data from zenodo
    - [ ] Investigate:
        - [ ] Calendar
        - [ ] Timesteps
        - [ ] Mean state
        - [ ] Grid Shape
        - [ ] Variable Names
    - [ ] Think about which timesteps we want to use in crossprediction. All? Only new test set? Does the new testset need to align with the old one?

More todo's: 
- [ ] Interpolations between grid
    - [ ] Can cdo be used from within python?
    - [ ] Grid description file generation
    - [ ] Automate interpolation to ...
        - [ ] … generate datasets
        - [ ] … interpolate results
