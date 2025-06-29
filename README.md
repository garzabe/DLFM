# Deep Learning for Flux Modeling (DLFM)

## About:

Motivated by the results in Albert et al., 2017 ([doi:10.1007/s00442-017-3853-0](https://doi.org/10.1007/s00442-017-3853-0)) and other studies incorporating ecosystem memory into modeling ecological fluxes, this project provides a test bed for modeling flux data with machine learning (ML) and deep learning (DL).

## Installation:

This project was built and tested in Python 3.11. It is recommended to install ```>=3.11```, as earlier versions are not guaranteed to work.

From a terminal/command line, ensure pip is installed and updated:

``` python3 -m ensurepip --upgrade ```

Clone the repository by either downloading the code zip file or running (requires git to be installed on your machine):

```git clone https://github.com/garzabe/DLFM```

Optionally, create and activate a virtual environment with ```virtualenv``` or ```conda``` to encapsulate the required packages and versions for this project:

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

Run the following command to train an LSTM on your flux dataset with ***default input variables*** inputs and NEP output:

```python3 model_analysis.py```

NOTE: By default, the procedure looks for data in the filepath ```data/data.csv```. If your csv has a different name, either rename it to ```data.csv``` or change the ```data_filepath``` variable in the main function.

The results of the hyperparameter tuning process will be written out in the ```results``` directory. Example predictions on the training set and evaluation set and training curves will be written out in the ```images``` directory.

## File Breakdown

### [data_handler.py](https://github.com/garzabe/DLFM/blob/main/data_handler.py)

Defines the ```AmeriFLUXDataset``` classes and ```prepare_data()``` function.

The ```AmeriFLUXDataset``` class is a wrapper of the ```torch.utils.data.Dataset``` class used for PyTorch projects that is compatible with Sci-Kit Learn projects. Data can be directly accessed with the ```get_X()``` and ```get_y()``` functions.

### [model_class.py](https://github.com/garzabe/DLFM/blob/main/model_class.py)

Contains the PyTorch and sklearn model architecture classes.

All PyTorch model architectures are templated from ```NEPModel``` which only requires a ```forward(x)``` function, which acts similarly to the callable ```__call__``` function.

All SKLearn model architectures are templated from ```SKLModel``` which requires implementation of ```fit(X,y)```, ```predict(X)```, and ```__str___()```.


As of 6/28, ```model_class.py``` has the following model architectures implemented:

- XGBoost
- RandomForest
- Artificial Neural Network (ANN)
- LSTM
- xLSTM

### [train.py](https://github.com/garzabe/DLFM/blob/main/train.py)

Training, testing and evaluation procedures

#### ```train_test_eval(model_class, site, input_columns, **kwargs)```

The primary procedure for training new models. Calls ```train_hparam()``` to tune hyperparameters and train the model(s) with optimal hyperparameters, and evaluates the performance. Shows example predictions on the evaluation dataset with ```plot_predictions()``` and records hyperparameter tuning and evaluation results in the ```results``` directory.

Accepted kwargs:

- num_folds: The number of folds to use in K-fold cross validation
- skip_eval: Whether to skip the example prediction and evaluation procedures
- skip_curve: Whether to skip plotting training curves for each final model trained

#### ```train_hparam(model_class, site, input_columns, **kwargs)```

 Performs a grid search hyperparameter tuning procedure and trains model(s) with the optimal hyperparameter set.

Accepted kwargs:
- num_folds:
- num_models: The number of models to train with optimal hyperparameters
- optimizer_class: The PyTorch optimizer class to use for training. Defaults to Stochastic Gradient Descent SGD
- doy: Whether to include the day-of-year (1-366) as an input variable to the model
- skip_eval:
- skip_curve:
- flatten: Whether to flatten the 2-dimensional time series data on the temporal dimension for non-RNN models

#### ```train_kfold(num_folds, model_class, lr, bs, epochs, train_data, device, num_features, weight_decay=0, momentum=0, optimizer_class=torch.optim.SGD, **model_kwargs)```

The K-fold cross validation procedure for a given hyperparameter set. Trains models up to ```num_folds``` times depending on data availability. Returns the average R-squared explainability and mean squared error (MSE) among all successful folds.

Folds are determined by calendar year - this means that it is only possible to cross validate on as many folds as there are calendar years in the dataset. In addition, if a fold's validation set has fewer than $200 - \lfloor \sqrt{80 \times l} \rfloor$, where $l$ is the input sequence length, it is skipped. 

#### ```train_test(train_dataloader, test_dataloader, model, epochs, loss_fn, optimizer, device, skip_curve=False, context=None)```

Given training and validation datasets, trains and validates a model. If skip_curve is False, plots the training curve for the model as well. Returns a full history of the training and validation procedure.



#### ```plot_predictions.py```




### [model_analysis.py](https://github.com/garzabe/DLFM/blob/main/model_analysis.py)

The main file where model-data analysis is done. Includes pre-written analysis procedures in ```plot_sequence_importance()``` and ```variable_importance()```:

#### ```plot_sequence_importance(site, input_columns, model_class, num_models=5, max_sequence_length=90, flatten=False, **kwargs)```

To determine the prevalence of ecosystem memory in a model architecture, we train models on inputs with varying sequence lengths and compare prediction performance. With all other parameters remaining constant, any improvement in prediction performance as the sequence length increases can potentially be attributed to a better model of ecosystem memory.

Given a set of hyperparameters in ```kwargs```, trains ```num_models``` models with sequence lengths from 1 day to ```max_sequence_length``` days. After training all models, plots the average performance (R-squared and MSE) on training and evaluation sets with 95% confidence intervals against time.

#### ```variable_importance(site, input_columns, model_class, var_names, timesteps=[1], sequence_length=1, **model_kwargs)```

When we provide an RNN time-series inputs for a prediction on the final day of the sequence, we would like to understand how *important* the previous inputs are to the final prediction. Our approach here is to observe the variance of partial derivatives of the prediction with respect to individual inputs under the assumption that weight parameters tied to *important* inputs will diverge from 0 during the training process and result in variance in the partial derivative.

Trains a single model with the given sequence length and for each evaluation datapoint,variable, and timestep in ```timesteps```, calculates the partial derivative of predicted NEP with respect to the input and the variance with respect to each variable-timestep. Variances are plotted against timesteps.


## Model Performance

include perhaps the table of model performances on US-Me2