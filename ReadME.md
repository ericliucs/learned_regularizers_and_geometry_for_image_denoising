# Learned Regularizers and Geometry for Image Denoising

*Code by [Ryan Cecil](https://www.cecil-ryan.com/). Project advised by [Dr. Stacey Levine](https://www.duq.edu/academics/faculty/stacey-levine).*

This repository is a fork of Ryan Cecil's work on [this paper](https://www.bmvc2021-virtualconference.com/assets/papers/1117.pdf) for further testing
and experimentation.

## Table of Contents
- [Learned Regularizers and Geometry for Image Denoising](#learned-regularizers-and-geometry-for-image-denoising)
  - [Table of Contents](#table-of-contents)
  - [Denoising Models <a name="denoisingmodels"></a>](#denoising-models-)
  - [Getting Started](#getting-started)
    - [Installing Packages](#installing-packages)
    - [Setting PythonPath](#setting-pythonpath)
    - [Loading Data](#loading-data)
    - [Testing](#testing)
  - [Reproduction](#reproduction)
  - [Implementation and Training](#implementation-and-training)
  - [Citations](#citations)
  - [Acknowledgements](#acknowledgements)

## Denoising Models <a name="denoisingmodels"></a>

The currently implemented denoising models are:

1. [Trainable Nonlinear Reaction Diffusion Model (TNRD)](https://ieeexplore.ieee.org/document/7527621)
2. [GF-TNRD](https://www.bmvc2021-virtualconference.com/assets/papers/1117.pdf) and variations.
3. KB-TNRD
3. [DnCNN](https://arxiv.org/abs/1608.03981)
5. [Total Deep Variation (TDV)](https://arxiv.org/abs/2001.05005)

## Getting Started

To get started, first git clone the repository. Then, follow the next few steps.

### Installing Packages

There are a variety of python packages that are required to run the main denoising code. See ``requirements.txt`` to
view the packages that are required. I highly recommend that you create a new Conda virtual 
environment prior to installing the packages. The code was last tested with Python version ``3.7.10``.

**Important:** The packages are listed in requirements.txt, which is used to build an environment using the command
``conda create --name <name> --file requirements.txt``. **However**, if you try to run this command, you will get
a ``PackagesNotFoundError`` as conda cannot find certain packages that were installed through pip.

Provided is the ``list_export_to_yaml.awk`` file which can be used to convert the requirements to a YAML file,
which can be used to create an environment in a way that Conda can find all the packages. To create an environment
from the YAML file, run ``conda env create --name <name> --file environment.yml``.

The only package not able to be installed through pip is the [optox](https://github.com/VLOGroup/optox) package built by Kerstin Hammernik. 
This package is used to implement custom operators in both TNRD and TDV. The file ``install_optox.sh`` is
a typical way of installing the optox package that should be modified for your machine. Also included is a
``install_optox_new.sh`` file that I use in case of issues with cloning optox from GitHub.

**Important:** Some files make use of the PyYAML package, which is not included in the requirements. Run 
``conda install pyyaml`` in your conda environment to install it.

**Important:** optox depends on CUDA samples that are no longer included in the toolkit from version 11.6 and onwards.
Ensure that you use CUDA Toolkit 11.5.2 at the latest, as newer versions will not work.

### Setting PythonPath

Once all requirements are installed, before running any other code, please make sure that your python path is appropriately set. To set the path,
in the top-level directory folder, call ``export PYTHONPATH="${PYTHONPATH}:$PWD"``.

### Loading Data

Generating the BSDS train and test sets into the appropriate directories should be as simple as calling 
``python3 data/load_data.py``.

### Testing

Lastly, to ensure that all the code for training and testing is working, I would advise attempting to
run ``python3 reproduce/test/test.py`` to try training a small TNRD model on the GPU.

## Reproduction

The ``reproduce`` folder contains a variety of code snippets that allow the user to easily 
reproduce a few different results. The results are organized by directory and the current ones
that can be reproduced are:

1. ``denoisers`` - Trains, tests, and compares the performance of the denoising models mentioned above.
2. ``extra`` - Experiments with the GF-TNRD architecture inspired by the work in my thesis.
3. ``paper`` - Reproduces main table from [our paper](https://www.bmvc2021-virtualconference.com/assets/papers/1117.pdf).
4. ``test`` - Simply trains and tests a small TNRD denoising model to ensure that the code and packages are
        working correctly.
5. ``thesis`` - Reproduces results from my Duquesne thesis which will be published in 2023.

## Implementation and Training

Here is a quick overview of where the main implementation and training code are stored:

- ``data`` - Stores training and testing data and a file to download the data.
- ``denoising/generator`` - Code for the main keras data generator that is used during training and testing.
- ``denoising/models`` - Contains both code to define models and a directory to store trained models.
- ``denoising/training`` - Code that defines loss functions and custom callbacks to be used during training.
- ``denoising/util.py`` - Contains miscellaneous functions that are used throughout.

## Citations

If you happen to use any of their newly proposed architectures or variations, 
please cite the following conference paper:

- [Learned Regularizers and Geometry for Image Denoising](https://www.bmvc2021-virtualconference.com/conference/papers/paper_1117.html) by Stacey Levine, Ryan Cecil, and Marcelo Bertalmio.



## Acknowledgements

This work was supported by [Dr. Stacey Levine](https://www.duq.edu/academics/faculty/stacey-levine), the [Mathematics and 
Computer Science Department](https://www.duq.edu/academics/schools/liberal-arts/academics/departments-and-centers/mathematics-and-computer-science) 
at Duquesne University, and an [NSF](https://www.nsf.gov/) grant.