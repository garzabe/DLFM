import torch
from torch import nn, optim, autograd, cuda, Tensor
from torch.utils.data import DataLoader
import numpy as np
import scipy.stats as st
from datetime import datetime

from data_handler import  Site, prepare_data
from train import train_test_eval, fmt_date_string, train_hparam #, feature_pruning
from model_class import FirstANN, DynamicANN, RNN, LSTM, XGBoost, RandomForest, xLSTM

import matplotlib.pyplot as plt

default_hparams = {FirstANN: {'batch_size': 64, 'epochs': 800, 'lr': 0.001, 'weight_decay': 0.01},
                   DynamicANN: {'layer_dims': [(10,6)], 'epochs': [400], 'batch_size': 64, 'lr': 0.001, 'weight_decay': 0.01, 'flatten': True},
                   RNN: {'hidden_state_size': 15, 'num_layers': 1, 'epochs': 2000, 'batch_size': 64, 'lr': 0.001, 'weight_decay': 0.01}, # [8, 15]
                   LSTM: {'hidden_state_size': 8, 'num_layers': 3, 'epochs': 100, 'batch_size': 64, 'lr': 0.01, 'weight_decay': 0.00, 'momentum': 0.001}, # [8, 15]
                   xLSTM: {'epochs': 600, 'batch_size': 64, 'lr': 0.005, 'weight_decay': 0.0, 'momentum': 0.001},
                   XGBoost: {'lr': 0.01, 'n_estimators': 1000},
                   RandomForest: {'n_estimators': 10000}}

def test_tte():
    simple_cols = ['P', 'PPFD_IN', 'D_SNOW']
    # This should take little time to run

    print("TTE with xLSTM")
    train_test_eval(xLSTM, site=Site.Me2, input_columns=simple_cols, num_folds=2, epochs=1, slr=1e-2, batch_size=64, time_series=True, sequence_length=7)
    
    # print("TTE with DynamicANN")
    # train_test_eval(DynamicANN, site=Site.Me2, input_columns=simple_cols, layer_dims=[(2,),(2,2)], num_folds=2, epochs=5, lr=1e-2, batch_size=64)
    # print("TTE with DynamicANN & multiple stat intervals")
    # train_test_eval(DynamicANN, layer_dims=[(2,),(2,2)], num_folds=2, epochs=5, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, stat_interval=[None,7,14])
    # print("TTE with DynamicANN & ustar=na")
    # train_test_eval(DynamicANN, layer_dims=[(2,),(2,2)], num_folds=2, epochs=5, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, ustar='na')
    # # test time series + flatten for linear models
    # print("TTE with DynamicANN & flattened time series data")
    # train_test_eval(DynamicANN, layer_dims=[(2,),(2,2)], num_folds=2, epochs=5, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, time_series=True, sequence_length=7, flatten=True)
    # print("TTE with DynamicANN and seasonal data (summer)")
    # train_test_eval(DynamicANN, site=Site.Me2, num_folds=2, input_columns=simple_cols, epochs=2, season='summer')
    
    # test time series data preparation and RNN predictor model
    #print("TTE with RNN and match sequence length")
    #train_test_eval(RNN, site=Site.Me2, input_columns=simple_cols, num_folds=2, epochs=1, lr=1e-2, batch_size=64, time_series=True, sequence_length=7, match_sequence_length=31)
    # print("TTE with RNN")
    # train_test_eval(RNN, site=Site.Me2, input_columns=simple_cols, num_folds=2, epochs=1, slr=1e-2, batch_size=64, time_series=True, sequence_length=7)
    # print("TTE with RNN and ustar=na")
    # train_test_eval(RNN, site=Site.Me2, input_columns=simple_cols, num_folds=2, sequence_length=12, time_series=True, ustar='na')
    # print("TTE with RNN and seasonal data (winter)")
    # train_test_eval(RNN, site=Site.Me2, input_columns=simple_cols, num_folds=2, sequence_length=12, time_series=True, season='winter')
    #print("TTE with LSTM")
    #train_test_eval(LSTM, site=Site.Me2, input_columns=simple_cols, num_folds=2, epochs=1, lr=1e-2, batch_size=64, time_series=True, sequence_length=50)

def test_sklearn():
    simple_cols=['P','PPFD_IN']

    print("TTE with XGBoost")
    train_test_eval(XGBoost, site=Site.Me2, input_columns=simple_cols, num_folds=2, lr=[0.1, 1], n_estimators=[10,100], sklearn_model=True, num_models=10)
    print("TTE with RandomForest")
    train_test_eval(RandomForest, site=Site.Me2, input_columns=simple_cols, num_folds=2, n_estimators=[10,100], sklearn_model=True, num_models=10)

def search_longest_sequence(input_columns, ustar=None):
    longest_sequence_low = 1
    longest_sequence_high = 1000
    while longest_sequence_low != longest_sequence_high-1:
        sequence_length = (longest_sequence_low + longest_sequence_high)//2
        print(f"Testing {sequence_length}")
        r2 = train_test_eval(RNN, site=Site.Me2, input_columns=input_columns, epochs=1, num_folds=2, time_series=True, sequence_length=sequence_length, skip_eval=True)
        if r2 == -np.inf:
            print("TTE failed to train a model due to sequence length")
            longest_sequence_high=sequence_length
        else:
            print("TTE trained a model with the given sequence length")
            longest_sequence_low=sequence_length
    print(f"The longest sequence found was {longest_sequence_low}")
    return longest_sequence_low

def plot_sequence_importance(site, input_columns, model_class, num_models=5, max_sequence_length=90, flatten=False, **kwargs):
    r2_results = []
    mse_results = []
    r2_t_results = []
    mse_t_results = []
    results = []
    sequence_args = default_hparams[model_class]
    sequence_args.update(kwargs)
    #sequence_args['time_series'] = True
    sequence_args['match_sequence_length'] = max_sequence_length
    sequence_args['flatten'] = flatten

    if flatten and model_class.__name__ in ['RNN', 'LSTM', 'xLSTM']:
        print("Argument error: model class cannot be an RNN and take flattened time series data")
        return
    
    use_stat_interval = model_class.__name__ not in ['RNN', 'LSTM', 'xLSTM'] and not flatten
    if use_stat_interval:
        sequence_args['time_series'] = False
    
    sequence_lengths = list(range(1, max_sequence_length+1))
    for sl in range(1, max_sequence_length+1):
        sl_arg = {'stat_interval' if use_stat_interval else 'sequence_length': sl}
        iter = []
        for _ in range(num_models):
            r2, mse, r2_t, mse_t = train_test_eval(model_class, site, input_columns, **sl_arg, **sequence_args)
            iter.append([r2, mse, r2_t, mse_t])
        iter = np.array(iter)
        #print("Iteration")
        #print(iter.shape)
        #print(iter)
        means = iter.sum(axis=0)/num_models
        #print("means")
        #print(means.shape)
        #print(means)
        for var, mean in zip(iter, means):
            st.t.interval(0.95, len(iter)-1, loc=mean, scale=st.sem(var))
        quartile25 = np.percentile(iter, 25, axis=0)
        quartile75 = np.percentile(iter, 75, axis=0)
        # axis 0: timestep; axis 1: variable
        var_metrics = np.array([means, quartile25, quartile75])
        #print("variable metrics (not transposed)")
        #print(var_metrics.shape)
        #print(var_metrics)
        #var_metrics = var_metrics.transpose()
        #print("variable metrics (transposed)")
        #print(var_metrics.shape)
        #print(var_metrics)
        results.append(var_metrics)
        r2_results.append(r2)
        r2_t_results.append(r2_t)
        mse_results.append(mse)
        mse_t_results.append(mse_t)

        #results = np.array(results)

        # axis 0: variable; axis 1: timesteps
        #results = results.transpose(axes=[0,1])
    """
    r2_results.sort()
    r2_t_results.sort()
    mse_results.sort()
    mse_t_results.sort()
    r2_25 = np.percentile(r2_results, 25)
    r2_75 = np.percentile(r2_results, 75)
    r2_t_25 = np.percentile(r2_t_results, 25)
    r2_t_75 = np.percentile(r2_t_results, 75)
    mse_25 = np.percentile(mse_results, 25)
    mse_75 = np.percentile(mse_results, 75)
    mse_t_25 = np.percentile(mse_t_results, 25)
    mse_t_75 = np.percentile(mse_t_results, 75)
    """

    #r2_results
    results = np.array(results)
    #print(results.shape)
    #print(results)

    #plt.rcParams['text.usetex'] = True
    dt_str = fmt_date_string()
    # R-squared on both evaluation and training sets
    plt.clf()
    plt.plot(sequence_lengths, results[:,0,0], label='Mean '+r'R^2'+' on evaluation set', color='g')
    plt.fill_between(sequence_lengths, results[:,1,0], results[:,2,0], alpha=0.1, color='g')
    plt.plot(sequence_lengths, results[:,0,2], label='Mean '+r'R^2'+' on training set', color='b')
    plt.fill_between(sequence_lengths, results[:,1,2], results[:,2,2], alpha=0.1, color='b')
    plt.xlabel('Input Sequence Length (Days)')
    plt.ylabel('R^2')
    plt.legend()
    plt.ylim((0.1, 1.0))
    plt.title(f'Importance of Sequence Length for {model_class.__name__} Predictions')
    plt.savefig(f'images/sequence_length_importance-{dt_str}::{model_class.__name__}-r2.png')

    # MSE on both evaluation and training sets
    plt.clf()
    plt.plot(sequence_lengths, results[:,0,1], label='Mean MSE on evaluation set', color='g')
    plt.fill_between(sequence_lengths, results[:,1,1], results[:,2,1], alpha=0.1, color='g')
    plt.plot(sequence_lengths, results[:,0,3], label='Mean MSE on training set', color='b')
    plt.fill_between(sequence_lengths, results[:,1,3], results[:,2,3], alpha=0.1, color='b')
    plt.xlabel('Input Sequence Length (Days)')
    
    plt.ylabel('MSE')
    plt.legend()
    plt.title(f'Importance of Sequence Length for {model_class.__name__} Predictions')
    plt.savefig(f'images/sequence_length_importance-{dt_str}::{model_class.__name__}-mse.png')

# Does an exhaustive search for the best hyperparameter configuration of a vanilla neural network
# we can optionally include multiple stat intervals to search on as well
def best_vanilla_network_search(site, input_columns, sequence_length=None, flatten=False):
    # To define a model architecture with a single hidden layer, you must add a comma after the layer dimension in the tuple
    # or Python will simplify it to an int
    if not flatten:
        train_test_eval(DynamicANN,site=site, input_columns=input_columns,
                        layer_dims=[(4, ), (8, ), (10, ), (4,4), (6,4), (10,4), (6,6), (10,6), (4,4,4)],
                        num_folds=7,
                        epochs=[200, 400, 800],
                        lr=[1e-3, 1e-2],
                        batch_size=[32,64],
                        stat_interval=sequence_length)
    else:
        train_test_eval(DynamicANN, site=site, input_columns=input_columns,
                        layer_dims=[(4, ), (8, ), (10, ), (4,4), (6,4), (10,4), (6,6), (10,6), (4,4,4)],
                        num_folds=7,
                        epochs=[200, 400, 800],
                        lr=[1e-3, 1e-2],
                        batch_size=[32,64],
                        sequence_length=sequence_length, time_series=True, flatten=True)
    
def best_rnn_search(site, input_columns, sequence_length, max_sequence_length=None, model_class = LSTM, dropout=0.0, weight_decay=0.0):
    train_test_eval(model_class, site=site, input_columns=input_columns,
                    num_folds=7,
                    epochs=[500, 2000, 5000],
                    lr=[1e-2, 1e-3],
                    batch_size=[64],
                    sequence_length= sequence_length, 
                    max_sequence_length=max_sequence_length,
                    hidden_state_size=[4,8,15],
                    num_layers=[1,2,3],
                    dropout=dropout,
                    weight_decay=weight_decay,
                    time_series=True)
    
def sensitivity_analysis(site, input_columns, model_class, var_names : list, timesteps=None, **model_kwargs):
    # train a model
    models, best, history = train_hparam(model_class, site, input_columns, skip_eval=False, skip_curve=False, **model_kwargs)
    device = ("cuda" if cuda.is_available() else "cpu")

    # given the var_name input (and potentially the time step), calculate the partial derivatives
    # of the model function wrt each input-output

    # we need to prepare data and go as far as preparing the dataloader here to have the exact tensors
    # that are tied to the outputs
    # is grads batched allows for batch gradient calculation
    train, _ = prepare_data(site, input_columns, **model_kwargs)
    vars_idx = [train.get_var_idx(var_name) for var_name in var_names]
    dates : list[datetime] = train.get_dates()
    dataloader = DataLoader(train, batch_size=1, shuffle=False)
    for model in models:
        model.eval()

    # leaving the synthetic option here, but we are NOT using this for the remainder of the project
    synthetic = False
    show_NEP = True
    # either show dates on the x-axis or show input values
    show_dates = True

    # TEST -------------------------------------------
    # evaluate partial derivatives between -2*sigma to 2*sigma
    # for xLSTM which considers the interplay between vars, this generated data might not be ideal
    # ZEROS EXCEPT
    if synthetic:
        test_values = [-2.0 + i*(4/100) for i in range(101)]
        test_inputs : dict[str, torch.Tensor] = {}
        if model_kwargs.get('sequence_length', None) is not None:
            input_shape = (1, int(model_kwargs['sequence_length']), len(input_columns))
        else:
            input_shape = (1, len(input_columns))
        for var_name, var_idx in zip(var_names, vars_idx):
            test_inputs[var_name] = {}
            for timestep in timesteps:
                test_inputs[var_name][timestep] = []
                for i,v in enumerate(test_values):
                    _input = torch.zeros(input_shape)
                    _input[0, -timestep-1, var_idx] = v
                    _input.requires_grad_()
                    test_inputs[var_name][timestep].append(_input)
        #test_inputs = torch.tensor(np.array(test_inputs))
        # jacobians = []
        # for test_input in test_inputs:
        #     test_input.requires_grad_()
        #     for model in models:
        #         pred : Tensor = model(test_input.to(device))
        #         jacobian = autograd.grad(pred[0,0], test_input)[0]
        #         jacobians.append(jacobian)
                # TODO: do stats on the jacobians
        test_partials = {}
        for var_name, var_idx in zip(var_names, vars_idx):
            test_partials[var_name] = {}
            for timestep in timesteps:
                test_partials[var_name][timestep] = []
                for i,v in enumerate(test_values):
                    pred = model(test_inputs[var_name][timestep][i].to(device))
                    jacobian = autograd.grad(pred[0,0], test_inputs[var_name][timestep][i])[0]
                    partial = jacobian[0,-timestep-1, var_idx]
                    test_partials[var_name][timestep].append((-1 if show_NEP else 1)*partial)
        
        # For each variable, make a plot
        # On each plot, there are len(timestep) partial curves for each timestep evaluated
        for var_name in var_names:
            plt.clf()
            plt.title(f"Partial Derivatives of predicted {'NEP' if show_NEP else 'NEE'} with respect to {var_name}")
            plt.xlabel("Input Value")
            plt.ylabel("Partial Derivative")
            plt.ylim(-2,2)
            #plt.xticks(test_values)
            for timestep in timesteps:
                unnorm = [v*train.stds[var_name] + train.means[var_name] for v in test_values]
                partial = test_partials[var_name][timestep]
                plt.plot(unnorm, partial, label=f"{timestep} days before prediction")
            plt.legend()
            plt.show()
    
    else:
        partials = {}
        inputs = {}
        for var_name in var_names:
            partials[var_name] = {}
            inputs[var_name] = {}
            for timestep in timesteps:
                partials[var_name][timestep] = []
                inputs[var_name][timestep] = []

        vars_idx = [train.get_var_idx(var_name) for var_name in var_names]
        for i, (X, _) in enumerate(dataloader):
                # X has shape (batch_size, sequence_length, var)
                X.requires_grad_()
                date = dates[i]
                for model in models:
                    pred : Tensor = model(X.to(device))
                    #x = X[:, -timestep-1, train.get_var_idx(var_name):train.get_var_idx(var_name)+1]
                    # TODO: isolate the value in X that corresponds with var_name:timestep
                    jacobian = autograd.grad(pred[0,0], X)[0]
                    # TODO: collect jacobians from each model and calculate mean, variance
                # get the partial we are interested in
                for var_name, var_idx in zip(var_names, vars_idx):
                    for timestep in timesteps:
                        partial = jacobian[0,-timestep-1, var_idx]
                        # TODO: if we are plotting dates on the x-axis
                        if show_dates:
                            # reduce precision by removing year
                            input = date.timetuple().tm_yday
                            inputs[var_name][timestep].append(input)
                        else:
                            # if we are plotting variable values on the x-axis
                            input = X[0,-timestep-1, var_idx]
                            inputs[var_name][timestep].append(input.detach().numpy())
                        
                        partials[var_name][timestep].append((-1 if show_NEP else 1)*partial.detach().numpy())
                        
        
        # For each variable, make a plot
        # On each plot, there are len(timestep) partial curves for each timestep evaluated
        for var_name in var_names:
            plt.clf()
            plt.title(f"Partial Derivatives of predicted {'NEP' if show_NEP else 'NEE'} with respect to {var_name}")
            plt.xlabel("Input Value")
            plt.ylabel("Partial Derivative")
            if show_dates:
                plt.xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
                           ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            plt.ylim(-2,2)
            #plt.xticks(test_values)
            for timestep in timesteps:
                if  show_dates:
                    unnorm = inputs[var_name][timestep]
                else:
                    unnorm = [v*train.stds[var_name] + train.means[var_name] for v in inputs[var_name][timestep]]
                partial = partials[var_name][timestep]
                xy = sorted(zip(unnorm, partial), key=lambda e: e[0])
                _x = [e[0] for e in xy]
                _y = [e[1] for e in xy]
                # since the data is naturally circular, wrap the data with copies of itself
                # so that the polyfit curve is forced to be continuous from Dec-Jan
                if show_dates:
                    _x_prior = [__x-365 for __x in _x]
                    _x_latter = [__x+365 for __x in _x]
                    _x_wrap = [*_x_prior, *_x, *_x_latter]
                    _y_wrap = _y*3
                    # adding two more sets of data increases the order of the function
                    f = np.poly1d(np.polyfit(_x_wrap, _y_wrap, 9))
                else:
                    f = np.poly1d(np.polyfit(_x,_y,3))
                plt.scatter(_x, _y, s=0.2)
                plt.plot(_x, f(_x), label=f"{timestep} days before prediction")
            plt.legend()
            #plt.show()

    # TODO: analyze the partials: average, distribution, etc.
    mean_partials = {}
    var_partials = {}
    for var_name in var_names:
        mean_partials[var_name] = []
        var_partials[var_name] = []
        for timestep in timesteps:
            mean_partial = sum(partials[var_name][timestep])/len(partials[var_name][timestep])
            # mean absolute partial derivative
            # handles odd funcs that would average to 0 and not recognize importance
            mean_abs_partial = sum([abs(partial) for partial in partials[var_name][timestep]])/len(partials[var_name][timestep])
            #print(f"The average absolute partial derivative for {var_name} at timestep {timestep} is {mean_abs_partial}")
            mean_partials[var_name].append(mean_abs_partial)

            var_partial = np.var(partials[var_name][timestep])
            #print(f"The variance of the partial derivative for {var_name} at timestep {timestep} is {var_partial}")
            var_partials[var_name].append(var_partial)


    return var_partials
    
    
me2_input_column_set = [
        'D_SNOW',
        'SWC_4_1_1',
        'RH',
        'PPFD_IN',
        'TS_1_6_1', # replacing TS_1_3_1
        'P',
        'WD',
        'WS',
        'TA_1_1_3',
]

me6_input_column_set = [
    'D_SNOW',
    'SWC_1_5_1',
    'SWC_1_2_1',
    'RH',
    'NETRAD',
    'PPFD_IN',
    'TS_1_5_1',
    'P',
    'WD',
    'WS',
    'TA_1_1_2'
]

def main():
    site = Site.Me2
    MAX_SEQUENCE_LENGTH=7

    #train_test_eval(XGBoost, site, me2_input_column_set, sequence_length=7, flatten=True, lr=[0.001, 0.01,0.1], n_estimators=[1000, 10000, 100000], num_folds=3)
    #train_test_eval(XGBoost, site, me2_input_column_set, sequence_length=31, flatten=True, lr=[0.001, 0.01,0.1], n_estimators=[1000, 10000, 100000], num_folds=3)
    #train_test_eval(XGBoost, site, me2_input_column_set, stat_interval=7,  lr=[0.001, 0.01,0.1], n_estimators=[1000, 10000, 100000], num_folds=3)
    #train_test_eval(XGBoost, site, me2_input_column_set, stat_interval=31,  lr=[0.001, 0.01,0.1], n_estimators=[1000, 10000, 100000], num_folds=3)

    #train_test_eval(RandomForest, site, me2_input_column_set, sequence_length=7, flatten=True,  n_estimators=[1000, 10000], num_folds=3)
    #train_test_eval(RandomForest, site, me2_input_column_set, sequence_length=31, flatten=True,  n_estimators=[1000, 10000], num_folds=3)
    #train_test_eval(RandomForest, site, me2_input_column_set, stat_interval=7,  n_estimators=[1000, 10000], num_folds=3)
    #train_test_eval(RandomForest, site, me2_input_column_set, stat_interval=31,  n_estimators=[1000, 10000], num_folds=3)
    
    #plot_sequence_importance(site, me2_input_column_set, XGBoost, max_sequence_length=31, num_models=5, flatten=True)
    #plot_sequence_importance(site, me2_input_column_set, XGBoost, num_models=5, max_sequence_length=31, flatten=False)
    
    #plot_sequence_importance(site, me2_input_column_set, XGBoost, max_sequence_length=31, flatten=True)
    #plot_sequence_importance(site, me2_input_column_set, XGBoost, max_sequence_length=31, flatten=False)
    #plot_sequence_importance(site, me2_input_column_set, xLSTM, max_sequence_length=31)
    #train_test_eval(LSTM, site, me2_input_column_set, lr=0.001, epochs=1000, sequence_length=180, hidden_state_size=8, num_layers=3)
    model_class = LSTM
    hparams : dict[str, str | int] = default_hparams[model_class]
    hparams.update({'sequence_length': MAX_SEQUENCE_LENGTH})
    var_partials = sensitivity_analysis(site, me2_input_column_set, model_class, me2_input_column_set, timesteps=list(range(0,MAX_SEQUENCE_LENGTH)), **hparams)
    plt.clf()
    for var_name in var_partials.keys():
        plt.plot(range(0,MAX_SEQUENCE_LENGTH), var_partials[var_name], label=var_name)
    plt.yscale('log')
    plt.xlabel('Days before prediction')
    plt.ylabel('Variance of Partial Derivative')
    plt.legend()
    plt.show()
                
if __name__=="__main__":
    main()