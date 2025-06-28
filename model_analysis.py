import torch
from torch import nn, optim, autograd, cuda, Tensor
from torch.utils.data import DataLoader
import numpy as np
import scipy.stats as st
from datetime import datetime

from data_handler import  Site, prepare_data, find_prefix, COLUMN_LABELS
from train import train_test_eval, fmt_date_string, train_hparam #, feature_pruning
from model_class import FirstANN, DynamicANN, RNN, LSTM, XGBoost, RandomForest, xLSTM, MODEL_HYPERPARAMETERS

import matplotlib.pyplot as plt

def plot_sequence_importance(site, input_columns, model_class, num_models=5, max_sequence_length=90, flatten=False, **kwargs):
    r2_results = []
    mse_results = []
    r2_t_results = []
    mse_t_results = []
    results = []
    sequence_args = MODEL_HYPERPARAMETERS[model_class]
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
    
    sequence_lengths = list(range(1, max_sequence_length+1, 2))
    for sl in sequence_lengths:
        sl_arg = {'sequence_length': sl}
        iter = []
        for _ in range(num_models):
            r2, mse, r2_t, mse_t = train_test_eval(model_class, site, input_columns, skip_eval=True, stats=use_stat_interval, **sl_arg, **sequence_args)
            iter.append([r2, mse, r2_t, mse_t])
        iter = np.array(iter)

        means = iter.sum(axis=0)/num_models

        ci_low = []
        ci_high = []
        for var, mean in zip(iter.transpose(), means):
            lo, hi = st.norm.interval(0.95, loc=mean, scale=st.sem(var))
            ci_low.append(lo)
            ci_high.append(hi)
        #quartile25 = np.percentile(iter, 25, axis=0)
        #quartile75 = np.percentile(iter, 75, axis=0)
        # axis 0: timestep; axis 1: variable
        var_metrics = np.array([means, ci_low, ci_high])

        results.append(var_metrics)
        r2_results.append(r2)
        r2_t_results.append(r2_t)
        mse_results.append(mse)
        mse_t_results.append(mse_t)

    results = np.array(results)


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
    plt.savefig(f'images/sequence_length_importance-{dt_str}-{model_class.__name__}-r2.png')

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
    plt.savefig(f'images/sequence_length_importance-{dt_str}-{model_class.__name__}-mse.png')

# Does an exhaustive search for the best hyperparameter configuration of a vanilla neural network
# we can optionally include multiple stat intervals to search on as well
def best_vanilla_network_search(site, input_columns, sequence_length=None, flatten=False):
    # To define a model architecture with a single hidden layer, you must add a comma after the layer dimension in the tuple
    # or Python will simplify it to an int
    model_hparams = MODEL_HYPERPARAMETERS[DynamicANN]
    if not flatten:
        train_test_eval(DynamicANN,site=site, input_columns=input_columns, num_folds=7, stat_interval=sequence_length,
                        **model_hparams)
    else:
        train_test_eval(DynamicANN, site=site, input_columns=input_columns, num_folds=7, sequence_length=sequence_length, flatten=True,
                        **model_hparams)
    
def best_rnn_search(site, input_columns, sequence_length, max_sequence_length=None, model_class = LSTM, dropout=[0.0, 0.001], weight_decay=[0.0, 0.001]):
    train_test_eval(model_class, site=site, input_columns=input_columns,
                    num_folds=7,
                    epochs=[100, 500],
                    lr=[1e-2, 1e-3],
                    batch_size=[64],
                    sequence_length= sequence_length, 
                    max_sequence_length=max_sequence_length,
                    hidden_state_size=[8,15],
                    num_layers=[2,3],
                    dropout=dropout,
                    weight_decay=weight_decay,
                    time_series=True)
    
def variable_importance(site, input_columns, model_class, var_names : list[str], timesteps : list[int] =[1], sequence_length=1, **model_kwargs):
    # train a model
    models, best, history = train_hparam(model_class, site, input_columns, skip_eval=False, skip_curve=False, sequence_length=sequence_length, **model_kwargs)
    device = ("cuda" if cuda.is_available() else "cpu")

    # given the var_name input (and potentially the time step), calculate the partial derivatives
    # of the model function wrt each input-output

    # we need to prepare data and go as far as preparing the dataloader here to have the exact tensors
    # that are tied to the outputs
    # is grads batched allows for batch gradient calculation
    train, _ = prepare_data(site, input_columns, sequence_length=sequence_length, **model_kwargs)
    vars_idx = [train.get_var_idx(var_name) for var_name in var_names]
    dates : list[datetime] = train.get_dates()
    dataloader = DataLoader(train, batch_size=1, shuffle=False)
    for model in models:
        model.eval()

    # leaving the synthetic option here, but we are NOT using this for the remainder of the project
    synthetic = False
    show_NEP = True
    # either show dates on the x-axis or show input values
    show_dates = False
    # either show a scatterplot or shows stats on the partials
    scatter = False
    
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

    # collect statistics on partials
    mean_partials = {}
    var_partials = {}
    sigma_partials = {}
    for var_name in var_names:
        mean_partials[var_name] = []
        var_partials[var_name] = []
        sigma_partials[var_name] = []
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

            sigma_partial = np.std(partials[var_name][timestep])
            sigma_partials[var_name].append(sigma_partial)

    if scatter: 
        
        # For each variable, make a plot
        # On each plot, there are len(timestep) partial curves for each timestep evaluated
        for var_name in var_names:
            var_prefix = (var_name)
            plt.clf()
            plt.title(f"Partial Derivatives of predicted {'NEP' if show_NEP else 'NEE'} with respect to {var_name}")
            plt.xlabel(COLUMN_LABELS[var_prefix]['y_label'])
            plt.ylabel(rf'$\delta$NEP / $\delta${var_name}')
            if show_dates:
                plt.xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
                            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            plt.ylim(-2,2)
            #plt.xticks(test_values)
            colors = ['b','r','g','m']
            for i, timestep in enumerate(timesteps):
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
                plt.scatter(_x, _y, s=0.6, c=colors[i])
                #plt.plot(_x, f(_x), label=f"{timestep} days before prediction")
            plt.legend([f"{timestep} day{'s' if timestep != 1 else ''} before prediction" for timestep in timesteps])
            plt.show()
    else:
        plt.clf()
        var_names.sort(key=lambda v: var_partials[v][0], reverse=True)
        max_var = max([var_partials[v][0] for v in var_names])
        # get a plot with each variable isolated
        for var_name in var_names:
            var_prefix = find_prefix(var_name)
            # create hidden plot for each other variable
            for v in var_names:
                if v==var_name:
                    plt.plot(timesteps, var_partials[var_name], label=COLUMN_LABELS[var_prefix]['title'])
                else:
                    v_p = find_prefix(v)
                    plt.plot(timesteps, [-1]*len(timesteps), label=COLUMN_LABELS[v_p]['title'])

            
            plt.xlabel('Days before prediction')
            plt.ylabel(f'Variance of Partial Derivatives of NEP Prediction')
            plt.xticks(list(range(0, max(timesteps)+1, max(timesteps) // 10)))
            plt.ylim(bottom=-0.02, top=max_var*1.2)
            plt.legend()
            plt.show()    
        for var_name in var_names:
            var_prefix = find_prefix(var_name)
            # create hidden plot for each other variable
            plt.plot(timesteps, var_partials[var_name], label=COLUMN_LABELS[var_prefix]['title'])
        plt.xlabel('Days before prediction')
        plt.ylabel(f'Variance of Partial Derivatives of NEP Prediction')
        plt.xticks(list(range(0, max(timesteps)+1, max(timesteps) // 10)))
        plt.ylim(bottom=-0.02, top=max_var*1.2)
        plt.legend()
        plt.show()         
        #plt.yscale('log')
        
        plt.show()

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
    SITE = Site.Me2
    COLUMNS = me2_input_column_set
    MAX_SEQUENCE_LENGTH=120
    sequence_lengths = [3,4,5,6,7,14,31,90]
    n_folds = 3
    n_models = 1

    model_class = LSTM
    flatten = False
    hparams : dict[str, str | int] = MODEL_HYPERPARAMETERS[model_class]
    hparams.update({'lr':0.001, 'epochs':2500, 'hidden_state_size': 8, 'num_layers': 3, 'dropout': 0.00, 'weight_decay': 0.00})

    variable_importance(SITE, COLUMNS, model_class, me2_input_column_set, timesteps=list(range(0,MAX_SEQUENCE_LENGTH)), sequence_length=MAX_SEQUENCE_LENGTH, **hparams)




if __name__=="__main__":
    main()
