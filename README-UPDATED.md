# Updated Instructions

## Getting Started

To get started, first clone the repository. Then, follow the next few steps as the project cannot be built 
conventionally.

### Installing (Most) Packages

There are a variety of python packages that are required to run the main denoising code. See ``requirements.txt`` to
view the packages that are required. The code was last tested with Python version ``3.7.10``.

The packages are listed in requirements.txt, which is used to build an environment using the command
``conda create --name <name> --file requirements.txt``. **However**, if you try to run this command, you will get
a ``PackagesNotFoundError`` as conda cannot find certain packages that were installed through pip.

Provided is the ``list_export_to_yaml.awk`` file which can be used to convert the requirements to a YAML file,
which can be used to create an environment in a way that Conda can find all the packages. To create an environment
from the YAML file, run ``conda env create --name <name> --file environment.yml``.

**Important:** Some files make use of the PyYAML package, which is not included in the requirements. Once the 
environment is created, run ``conda install pyyaml`` in your environment to install it.

### CUDA

The last package that is needed, optox, depends on the CUDA Toolkit - more specifically, included CUDA
samples. However, these are no longer included in toolkit versions later than 11.5.2, so ensure that you install
11.5.2. at the latest. The toolkit can be found at https://developer.nvidia.com/cuda-toolkit-archive.

### optox

The only package not able to be installed through pip is the [optox](https://github.com/VLOGroup/optox) package built by Kerstin Hammernik. 
The file ``install_optox.sh`` is a typical way of installing the optox package that should be modified for your machine.
**Important:** Replace the paths in the file to your CUDA toolkit directories.

Also included is a``install_optox_new.sh`` file that I have used in case of issues with cloning optox from GitHub.

### Setting PythonPath

Once all requirements are installed, before running any other code, please make sure that your Python path is 
appropriately set. To set the path, in the top-level directory folder, call ``export PYTHONPATH="${PYTHONPATH}:$PWD"``.
All Python modules in the project must be run from the top-level directory with the environment variable set.

### Loading Data

Generating the BSDS train and test sets into the appropriate directories should be as simple as calling 
``python3 data/load_data.py`` from the top-level directory.

### Testing

Lastly, to ensure that all the code for training and testing is working, I would advise attempting to
run ``python3 reproduce/test/test.py`` to try training a small TNRD model on the GPU.

## Reproduction (from original README)

The ``reproduce`` folder contains a variety of code snippets that allow the user to easily 
reproduce a few different results. The results are organized by directory and the current ones
that can be reproduced are:

1. ``denoisers`` - Trains, tests, and compares the performance of the denoising models mentioned above.
2. ``extra`` - Experiments with the GF-TNRD architecture inspired by the work in my thesis.
3. ``paper`` - Reproduces main table from [our paper](https://www.bmvc2021-virtualconference.com/assets/papers/1117.pdf).
4. ``test`` - Simply trains and tests a small TNRD denoising model to ensure that the code and packages are
        working correctly.
5. ``thesis`` - Reproduces results from my Duquesne thesis which will be published in 2023.

## Implementation and Training (from original README)

Here is a quick overview of where the main implementation and training code are stored:

- ``data`` - Stores training and testing data and a file to download the data.
- ``denoising/generator`` - Code for the main keras data generator that is used during training and testing.
- ``denoising/models`` - Contains both code to define models and a directory to store trained models.
- ``denoising/training`` - Code that defines loss functions and custom callbacks to be used during training.
- ``denoising/util.py`` - Contains miscellaneous functions that are used throughout.
