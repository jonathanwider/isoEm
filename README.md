## Contents

* [Introduction](#Introduction)
* [Installation](#Installation)
* [Content-description](#Content-description)
* [Sources](#Sources)


# Introduction

This repository contains code to emulate stable oxygen isotopes in precipitation using various machine learning methods. Parts of the code are adapted from other repositories, see [Sources](#Sources).

The underlying simulation data is not provided with this repository, but a script to download it from [zenodo](https://zenodo.org/record/6610684) is included.






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


# Content-description
The repository contains tools to:
* Download required climate simulation files from zenodo and the MNIST dataset for a validation task of one type of network architecture (icosahedral CNN)
* run and create icosahedral and flat UNet models as well as simpler baseline models
* compare the results
* interpolate dataset between different grid (implemented: icosahedral and flat)

## Workflow

### Validation experiment:
* The ```gendata.py``` script downloads the MNIST data set. The data set is projected onto an icosahedral grid of refinement level $r=4$. It is possible to select from a range of rotation types that can be applied to test and training set.
* The ```Experiments_validate_MNIST.ipynb``` can be used to recreate the validation experiment once the corresponding datasets have been created with ```gendata.py```.

### Isotope emulation:
* The ```download_required_files.py``` script can be used to downloads the required climate simulation data from [zenodo](https://zenodo.org/record/6610684). For each variable we set a range of physically valid values. We interpolate all simulation data to the grid of the iHadCM3 cliamte model using bilinear interpolation from [CDO](https://code.mpimet.mpg.de/projects/cdo/). In addition to the monthly data sets, yearly averaged data sets are created. If the ```--createico``` flag is set, the yearly data sets are interpolated to the icosahedral grid with a first order conservative remapping (by default). These interpolated datasetsare required for experiments with the icosahedral neural network architectures. 
* To run experiments on the **yearly averaged data sets** use ```Experiments_yearly.ipynb``` notebook. The simulation results are stored in a directory, whose name is a hash that contains the information on parameters set during dataset creation, the training and the selected ML model.
* To evaluate the results of the yearly experiments use the ```Experiments_yearly_evaluate.ipynb``` notebook.
* Plots for the results can be created using the ```Plot_results.ipynb``` notebook.
* To run and evaluate the experiments on **monthly timescale**, use the ```Experiments_monthly.ipynb``` notebook.
* To run, evaluate and plot the ***crossprediction*** experiments, use the ```Experiments_crossprediction.ipynb``` notebook.

# Sources

The spherical Network architecture is a implemented based on the paper [Gauge Equivariant Convolutional Networks and the Icosahedral CNN](http://proceedings.mlr.press/v97/cohen19d/cohen19d.pdf).
The repository is based on publically available source code from ehoogeboom's [hexaconv](https://github.com/ehoogeboom/hexaconv) ([paper](https://arxiv.org/pdf/1803.02108.pdf))
and makes use of the [forked pytorch version](https://github.com/adambielski/GrouPy) of [GrouPy](https://github.com/tscohen/GrouPy) ([paper](https://arxiv.org/pdf/1602.07576.pdf)). 
Additionally code for creating the spherical-MNIST dataset is adapted from [S2CNN](https://github.com/jonas-koehler/s2cnn).
Code for CoordConv is from [CoordConv](https://github.com/walsvid/CoordConv) ([paper](https://proceedings.neurips.cc/paper/2018/file/60106888f8977b71e1f15db7bc9a88d1-Paper.pdf)). For implementing UNets, we use a flexible [decoder-encoder skeleton](https://github.com/imagirom/ConfNets).
