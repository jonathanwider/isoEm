## Contents

* [Introduction](#Introduction)
* [Installation](#Installation)
* [Content-description](#Content-description)
* [Sources](#Sources)


# Introduction

This repository contains code to emulate stable oxygen isotopes in precipitation using various machine learning methods. Parts of the code are adapted from other repositories, see [Sources](#Sources).

The underlying simulation data is not provided with this repository, but a script to download it from [zenodo](https://zenodo.org/record/6610684) is included.






# Installation

[Climate Data Operators (CDO)](https://code.mpimet.mpg.de/projects/cdo/) is required to interpolate between grids. If not already installed, it can be done by `sudo apt install cdo`.

To install the requirements, we can use conda:
`conda create -n <my-env-name> python=3.8`,
`conda activate <my-env-name>`

The required packages can then be installed with 
`pip install -r requirements.txt`
and 
`conda install -c conda-forge cartopy`.




# Content-description
The repository contains tools to:
* Download required climate simulation files from zenodo and the MNIST dataset for a validation task of one type of network architecture (icosahedral CNN)
* run and create icosahedral and flat UNet models as well as simpler baseline models
* compare the results
* interpolate dataset between different grid (implemented: icosahedral and flat)

## Workflow

### Isotope emulation:
* The ```download_required_files.py``` script can be used to downloads the required climate simulation data from [zenodo](https://zenodo.org/record/6610684). For each variable we set a range of physically valid values. We interpolate all simulation data to the grid of the iHadCM3 cliamte model using bilinear interpolation from [CDO](https://code.mpimet.mpg.de/projects/cdo/). In addition to the monthly data sets, yearly averaged data sets are created. If the ```--createico``` flag is set, the yearly data sets are interpolated to the icosahedral grid with a first order conservative remapping (by default). These interpolated datasetsare required for experiments with the icosahedral neural network architectures. 
* When creating data sets and running experiments, a lot of free choices remain. The parameter choices are then passed to functions as dicts, usually ```dataset_description``` for setttings related to the creation of the dataset and ```model_training_description``` for parameters that concern the training procedure and ML models that are used. For details on choices of these lists, see [parameter_descriptions.md](/parameter_descriptions.md). Parameters to reproduce our results are already specified in the corresponding jupyter notebooks.
* To run experiments on the **yearly averaged data sets** use ```Experiments_yearly.ipynb``` notebook. The simulation results are stored in a directory, whose name is a hash that contains the information on parameters set during dataset creation, the training and the selected ML model.
* To evaluate the results of the yearly experiments use the ```Experiments_yearly_evaluate.ipynb``` notebook. In general, when evaluating parameters of the desired dataset, ML model and training procedure can be specified - and the suitable results folder is opened automatically.
* Plots for the results can be created using the ```Plot_results.ipynb``` notebook.
* To run and evaluate the experiments on **monthly timescale**, use the ```Experiments_monthly.ipynb``` notebook.
* To run, evaluate and plot the ***crossprediction*** experiments, use the ```Experiments_crossprediction.ipynb``` notebook.

### Validation experiment:
* The ```gendata.py``` script downloads the MNIST data set. The data set is projected onto an icosahedral grid of refinement level $r=4$. It is possible to select from a range of rotation types that can be applied to test and training set.
* The ```Experiments_validate_MNIST.ipynb``` can be used to recreate the validation experiment once the corresponding datasets have been created with ```gendata.py```.

## Files and subdirectories and their content:

| Directory or file name| Short description of content|
| ------------- |-------------|
| ```CoordConv/``` | Code for [CoordConv](https://github.com/walsvid/CoordConv). Slightly modified to work with cylindrical padding (fact that longitudes on earth have periodic boundary condition).|
|```Grids/```| Directory grid description files get stored in. CDO can extract the natural grid from .nc files if necessary.|
|```Scripts/```| Contains scripts that are used to interpolate between the flat plate carrée grid and the icosahedral grid. These scripts get used by funcitons in ```interpolate.py``` |
|```Scripts/```| Contains scripts that are used to interpolate between the flat plate carrée grid and the icosahedral grid. These scripts get used by functions in ```interpolate.py```. The directory also contains the used grid description files.|
| ```groupy/``` | A [forked pytorch version of groupy](https://github.com/adambielski/GrouPy) that was adapted to hexagonal convolution as in [hexaconv](https://github.com/ehoogeboom/hexaconv). Used to build the convolutional layers of the icosahedral neural network|
| ```Experiments_crossprediction.ipynb``` | Crossprediction experiments: Train network on data from one climate model, then predict on other climate models. |
| ```Experiments_monthly_data.ipynb``` | Experiments on monthly data. On this time scale there are a lot more missing values than on annual scale. |
| ```Experiments_yearly.ipynb``` | Experiments on yearly data. Only producing datasets, training models and making predictions|
| ```Experiments_yearly_evaluate.ipynb``` | Evaluate the results of previously run experiments on yearly data|
| ```Plot_MNIST_digits.ipynb``` | Plots some of the digits from the icoMNIST dataset.|
| ```Plot_results.ipynb``` | Generate plots presented in the paper given that datasets and models have been trained an created previously|
| ```base.py``` | Defines some basic functions that are used in the construction of the flat and icosahedral UNets.|
| ```datasets.py``` | Functions to create data sets used in training from climate model data. Here e.g. the split into test and training set happens and the used variables are selected.|
| ```download_required_files.py``` | Downloads the climate model data from [zenodo](https://zenodo.org/record/6610684) and applies preprocessing, see [isotope emulation workflow](#Workflow). |
| ```evaluate.py``` | Functions for evaluating trained models (e.g. to compute metrics like correlation or R2-score) |
| ```generate_grid_description_files.py``` | Interpolation from and to the icosahedral grid with [CDO](https://code.mpimet.mpg.de/projects/cdo/) requires grid description files of the icosahedral grid. If a configuration that deviates from our choice is to be implemented (e.g. in resolution), this script can generate the corresponding description file. |
| ```ìco_unet.py``` | Classes for flat and icosahedral UNet architectures. |
| ```icosahedron.py``` | Defines a class that represents the icosahedral grid. |
| ```ìnterpolate.py``` | Functions to interpolate predictions or entire climate model data sets between grids. They make use of the scripts stored in ```Scripts/```|
| ```modules.py``` | Predefines some modules that are used in the construction of flat and icosahedral UNet |
| ```plotting.py``` | Plotting functions including map plots for plate carrée and icosahedral data. Maps require ```cartopy``` to be installed. |
| ```predict.py``` | Functions to do predictions with trained ML-models. |
| ```preprocess.sh``` | Preprocessing script used by ```download_required_files.py``` |
| ```train.py``` | Functions to train ML-models on previously created data sets. |
| ```train_tune_pca.py``` | Defines a function that does hyperparameter selection for the PCA-regression baseline. |
| ```util.py``` | Small helper functions. |
| ```validation_experiment_MNIST_gendata.py``` | Download and generate the icoMNIST dataset |
| ```Validation_experiment_MNIST.ipynb``` | Recreation of a task from [Gauge Equivariant Convolutional Networks and the Icosahedral CNN](http://proceedings.mlr.press/v97/cohen19d/cohen19d.pdf) to validate our implementation of the icosahedral neural network|


# Sources

The spherical Network architecture is a implemented based on the paper [Gauge Equivariant Convolutional Networks and the Icosahedral CNN](http://proceedings.mlr.press/v97/cohen19d/cohen19d.pdf).
The repository is based on publically available source code from ehoogeboom's [hexaconv](https://github.com/ehoogeboom/hexaconv) ([paper](https://arxiv.org/pdf/1803.02108.pdf))
and makes use of the [forked pytorch version](https://github.com/adambielski/GrouPy) of [GrouPy](https://github.com/tscohen/GrouPy) ([paper](https://arxiv.org/pdf/1602.07576.pdf)). 
Additionally code for creating the spherical-MNIST dataset is adapted from [S2CNN](https://github.com/jonas-koehler/s2cnn).
Code for CoordConv is from [CoordConv](https://github.com/walsvid/CoordConv) ([paper](https://proceedings.neurips.cc/paper/2018/file/60106888f8977b71e1f15db7bc9a88d1-Paper.pdf)). For implementing UNets, we adapt a flexible [decoder-encoder skeleton](https://github.com/imagirom/ConfNets).
