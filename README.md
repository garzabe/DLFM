### Deep Learning for Flux Modeling (DLFM)

## About:

Motivated by the results in Albert et al., 2017 (*citation*) and other studies incorporating ecosystem memory into modeling ecological fluxes, this project provides a test bed for modeling flux data with machine learning (ML) and deep learning (DL).

## Installation:

This project was built and tested in Python 3.11. It is recommended to install >=3.11, as earlier versions are not guaranteed to work.

From a terminal/command line, ensure pip is installed and updated:

``` python3 -m ensurepip --upgrade ```

Clone the repository by either downloading the code zip file or running (requires git to be installed on your computer):

```git clone https://github.com/garzabe/DLFM```

Optionally, create and activate a virtual environment with ```venv``` or ```conda``` to encapsulate the required packages and versions for this project:

```
python3 -m virtualenv ./.venv
source .venv/bin/activate
```

or with ```conda```:

```
conda create --name ".venv" python=3.11
source .venv/bin/activate
```


After ```cd DLFM``` into the repository directory, install all required python packages with:

``` pip install -r requirements.txt ```

Note: This requires a few 100MB of storage to install all packages.

Finally, download your favorite flux dataset ```.csv``` and place it in the ```data``` directory.

## Quick Start

TODO: How do we want users to simply specify their dataset file? Force them to rename to data.csv or have a config file/constant variable file where they can copy the filename?

Run the following command to train an LSTM on your flux dataset with ***default input variables*** inputs and NEP output:

```python3 model_analysis.py```

The results of the hyperparameter tuning process will be written out in the ```results``` directory. Example predictions on the training set and evaluation set and training curves will be written out in the ```images``` directory.

## File Breakdown

# (data_handler.py)[https://github.com/garzabe/DLFM/data_hander.py]

Data pre-processing steps. Primarily contains just the ```prepare_data()``` function.

Also includes dataset wrapper classes that enable use with both pyTorch models and sklearn models

# (model_class.py)[https://github.com/garzabe/DLFM/model_class.py]

Contains the model architectures and wrapper classes

As of 5/30, contains the following model architectures:

- XGBoost
- RandomForest
- Artificial Neural Network (ANN)
- LSTM
- xLSTM

# (train.py)[https://github.com/garzabe/DLFM/train.py]

Training, testing and evaluation procedures

```train_test_eval.py```:

```train_hparam.py```:

```train_test.py```:

```train_kfold.py```:

```plot_predictions.py```: 

# (model_analysis.py)[https://github.com/garzabe/DLFM/model_analysis.py]

Some pre-written analysis procedures in ```plot_sequence_importance()``` and ```variable_importance()``` 

Also includes a default set of hyperparameters for the pre-included model architectures (TODO move this to model_class)

[comment]: # (- ()[https://github.com/garzabe/DLFM/])

## Model Performance

include perhaps the table of model performances on US-Me2