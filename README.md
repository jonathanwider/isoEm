## Contents

* [Introduction](#Introduction)
* [Installation](#Installation)
* [Content-description](#Content-description)
* [Sources](#Sources)


# Introduction

This repository documents the code that was written for the Master's Thesis "Can Convolutional Neural Networks Replace the Physics-based
Simulation of Oxygen Isotope Ratios in Precipitation?". Parts of the code are adapted from other repositories, see [Sources](#Sources).

The underlying simulation data is not provided with this repository, it is stored locally together with the created plots and Output of the emulation runs on the servers of the Stacy/Spacy group. 






# Installation

After cloning this repository, the required packages can be installed to a virtual or conda environment using `pip`. When working with `conda`, create and activate an environment 
`conda create -n my-env`,
`conda activate my-env`

and install `pip`:
`conda install pip`.

The required packages can then be installed with 
`pip install -r requirements.txt`

Additionally, to create some of the plots, `cartopy` must be installed via `conda` or `pip`. To open the jupyter notebooks, use `pip install notebook`, if not already installed.

[Climate Data Operators (CDO)](https://code.mpimet.mpg.de/projects/cdo/) is required to interpolate between grids.

Feel free to contact me in case of problems with the installation.

# Content-description
The repository contains tools to:
* create datasets from netcdf4 simulation files.
* run and create icosahedral and flat UNet models
* compare the results
* interpolate dataset between different grid (implemented: icosahedral and flat)

## Workflow
A typical workflow for the isotope-emulation task is as follows:
* OPTIONAL: If necessary, interpolate the flat basis data to the icosahedral grid (this only has to be done once) using one of the scripts in the `Scripts/` directory.
* Open one of the jupyter notebooks for Dataset generation (names of these notebooks start with `Dataset`). Within the notebook there are many possibilities to configure the dataset. One can for example select different predictor variables, the way the data is split to test and training set and more. A folder is created, in which the dataset is stored together with a `dataset-description.gz` file that stores information on the creation of the dataset.
* Open a notebook for running the architectures (names of these notebooks start with `Run`). These notebooks with determining a dataset-folder to work on based on selecting open parameters of the dataset. The notebooks let you configure many parameters of the used models and training. The results of each run get saved in a subfolder of the dataset-folder. The name of the folder in which the data gets stored is selected by hashing the dictionary containing model and training configurations. Before saving a run, hash-collisions are checked in order to avoid running accidentally running runs for the same configuration multiple times. Inside the directory, we store a file containing the model and training configuration (`model_training_description.gz`) and the predictions (`predictions.gz`) of the model on the test set.
* OPTIONAL: If necessary, the predictions can then be interpolated from one grid to another. This is necessary when comparing flat and icosahedral network architectures. The procedure is a little complicated: First netcdf4 files are computed in the notebooks starting with `Netcdf_from_`. These then need to be copied to the `Scripts` directory. In the scripts directory there are two python scripts than control shell scripts executing the interpolation using `CDO` (Climate Data Operators). The interpolated results must then be copied back to the original directory.

* Results can then be compared using one of the notebooks starting with `Compare`. These notebooks first load the model predictions and the required groundtruth from the dataset. They undo the standardization of the predictions and later provide plotting functions - thus creating the plots used in this thesis.

In the following we give a short description of the files and directories in this repository. 

| Foldername/Filename | Description |
| ----------- | ----------- |
| CoordConv | Adapted from [CoordConv](https://github.com/walsvid/CoordConv), we implement a version that can deal with cylindral padding and with the coordinate discontinuity in the longitudes(360°-0°) |
| Grids | Stores the grid description files that are necessary to interpolate between flat and icosahedral grid. | 
| MNIST_data | Holds data for the spherical MNIST validation task. Processed datasets not included, can be created from original MNIST using `gendata.py`|
| Scripts | Scripts to interpolate between icosahedral and flat grid. Sample grid description files are already contained. Files to be interpolated must be in this directory |
| groupy | [GrouPy](https://github.com/tscohen/GrouPy). This folder implements the building blocks for the icosahedral CNN. We combine a [forked pytorch version](https://github.com/adambielski/GrouPy) and adaptations to the hexagonal grid ([hexaconv](https://github.com/ehoogeboom/hexaconv)).|
| `Best_configuration_yearly_original_grid.ipynb` | Notebook that goes into details on the results of the best performing methods (sections: 4.2.5, 5.2.1)|
| `Compare_Architectures_On_ico_grid.ipynb` | Used for comparing predictions from networks on the icosahedral grid (sections: 4.2.2, 5.2.3) |
| `Compare_Architectures_On_original_grid.ipynb` | Used for comparing predictions from networks on the flat grid (sections: 4.2.2, 5.2.3) |
| `Compare_Hyperparameter_effects_On_original_grid.ipynb` | Compare results for different "hyperparameter-tuned architectures (sections: 4.2.4, 5.2.5) |
| `Compare_Modifications_to_flat_UNet.ipynb` | Used for comparing predictions from different "modified" flat architectures (sections: 4.2.1, 5.2.2) |
| `Compare_Results_On_ico_grid.ipynb` | "Base" notebook that was used to develop plotting and comparing on icosahedral grid |
| `Compare_Results_On_original_grid.ipynb` | "Base" notebook that was used to develop plotting and comparing on flat grid, used to create several plots in appendix |
| `Compare_Results_On_original_grid_monthly.ipynb` | Used to compare results for monthly architectures (sections: 4.3, 5.3) |
| `Compare_Results_On_original_grid_n_pc.ipynb` | Used for small side experiment: Dependence of the baseline achitecture on the number of Principal Components (Appendix) |
| `Compare_Results_On_original_grid_precip_weighted.ipynb` | To compare results with and without weighting the isotopic ratios by precipitation amount (section: Appendix) |
| `Dataset_from_interpolated_files.ipynb` | Create an icosahedral dataset. Before doing this, the flat data needs to be interpolated to the icosahedral grid using the scripts in the `Script` directory |
| `Dataset_from_original_netcdf_files_flat.ipynb` | Create a dataset from the original plate carrée output |
| `Dataset_from_original_netcdf_files_flat_months.ipynb` | Create a dataset from the original plate carrée output on monthly scale. Allows selecting single months and using predictor variables from multiple months. |
| `Dataset_from_original_netcdf_files_flat_months.ipynb` | Create a dataset from the original plate carrée output while weighting the variables by the amount of precipitation at every gridbox and timestep. |
| `Explore_hadcm3_dataset.ipynb` | Documents some investigations into the used HadCM3 datasets. |
| `Grid_description_files.ipynb` | Used to create grid description files that are required to perform interpolations with cdo. |
| `Hyperparameter_tuning_flat.ipynb` | Notebooks that was used to tune the learning rate of the flat network architecture using ray-tune. Output of the tuning not including in this repository, results are stored on the Stacy server however.|
| `Netcdf_from_flat_ML_model_output.ipynb` | Create NetCDF files that are required to interpolate the output of ML model to other grids if necessary. |
| `Netcdf_from_ico_ML_model_output.ipynb` | Create NetCDF files that are required to interpolate the output of ML model to other grids if necessary, other interpolation direction |
| `PCA_dependence_on_nb_PC.ipynb` | Used to run the classical baseline for different numbers of principle components. |
| `Plot_MNIST_digits.ipynb` | Plots some sample digits from the icosahedral/spherical MNIST dataset. |
| `Plot_correlations_meanstate_variance.ipynb` | As the name suggests: plots correlation maps, mean state and variance in the HadCM3 dataset (monthly and yearly) |
| `Plot_spherical_data.ipynb` | Plot some examples of data on the "bulged out" grid of the icosahedron. |
| `Single_variables_yearly_original_dataset.ipynb` | Calculates correlation coefficients between simulation and emulation at ice core drilling sites for simulations with single predictor variables. |
| `Test_equivariance.ipynb` | Testing the equivariance of the icosahedral CNN, was used for debugging. |
| `Test_ico_grid.ipynb` | Some plots of the icosaherdral grid and the charts that cover it. |
| `Test_padding.ipynb` | Testing the padding of the icosahedral CNN (had to be reimplemented by us). |
| `Validate_IcoCNN_on_IcoMNIST.ipynb` | Icosahedral MNIST validation experiment. Run here, results get stored by tensorboard in ``Validate` folder. |
| `base.py` | Base Modules that are used in the configurable UNet. |
| `gendata.py ` | Generate the icosahedral MNIST datasets from the standard one. Allows applying rotations (either random or icosahedral) |
| `ico_unet.py ` | Classes defining the flat and icosahedral UNets. |
| `icosahedron.py ` | Creation of the icosahedral grid. Rotations of the grid. Plotting functions. |
| `modules.py` | Defines several modules used in the icosahedral and flat UNets (padding, convolutions, batch norm, ...) |
| `run_classical.ipynb ` | Run the classical baseline on flat data. |
| `run_ico_classical.ipynb` |  Run the classical baseline on icosahedral data. |
| `run_ico_unet_systematic.ipynb` |  Run the icosahedral UNet. |
| `run_unet_systematic_flat-even-deeper.ipynb` | Run a deeper version of the flat UNet. |
| `run_unet_systematic_flat-even-wider.ipynb` | Run a wider version of the flat UNet. |
| `run_unet_systematic_flat-monthly.ipynb` | Run the flat UNet on monthly data. |
| `run_unet_systematic_flat.ipynb` | Run the flat UNet on yearly data. |
# Sources

The spherical Network architecture is a implemented based on the paper [Gauge Equivariant Convolutional Networks and the Icosahedral CNN](http://proceedings.mlr.press/v97/cohen19d/cohen19d.pdf).
The repository is based on publically available source code from ehoogeboom's [hexaconv](https://github.com/ehoogeboom/hexaconv) ([paper](https://arxiv.org/pdf/1803.02108.pdf))
and makes use of the [forked pytorch version](https://github.com/adambielski/GrouPy) of [GrouPy](https://github.com/tscohen/GrouPy) ([paper](https://arxiv.org/pdf/1602.07576.pdf)). 
Additionally code for creating the spherical-MNIST dataset is adapted from [S2CNN](https://github.com/jonas-koehler/s2cnn).
Code for CoordConv is from [CoordConv](https://github.com/walsvid/CoordConv) ([paper](https://proceedings.neurips.cc/paper/2018/file/60106888f8977b71e1f15db7bc9a88d1-Paper.pdf)). For implementing UNets, we use a flexible [decoder-encoder skeleton](https://github.com/imagirom/ConfNets).
