from torch import nn, optim, autograd
from torch.utils.data import DataLoader
import numpy as np

from data_handler import  Site, prepare_data
from train import train_test_eval, fmt_date_string, train_hparam #, feature_pruning
from model_class import FirstANN, DynamicANN, RNN, LSTM, XGBoost, RandomForest, xLSTM

import matplotlib.pyplot as plt

default_hparams = {FirstANN: {'batch_size': 64, 'epochs': 800, 'lr': 0.001, 'weight_decay': 0.01},
                   DynamicANN: {'layer_dims': [(10,6), (10,4)], 'epochs': [400, 800], 'batch_size': 64, 'lr': 0.001, 'weight_decay': 0.01},
                   RNN: {'hidden_state_size': 15, 'num_layers': 1, 'epochs': 2000, 'batch_size': 64, 'lr': 0.001, 'weight_decay': 0.01}, # [8, 15]
                   LSTM: {'hidden_state_size': 8, 'num_layers': 1, 'epochs': 2000, 'batch_size': 64, 'lr': 0.001, 'weight_decay': 0.01}, # [8, 15]
                   xLSTM: {'epochs': 500, 'batch_size': 64, 'lr': 0.001, 'weight_decay': 0.0},
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

    if flatten and model_class.__name__ in ['RNN', 'LSTM']:
        print("Argument error: model class cannot be an RNN and take flattened time series data")
        return
    
    use_stat_interval = model_class.__name__ not in ['RNN', 'LSTM'] and not flatten
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
    plt.fill_between(sequence_lengths, results[:,1,0], results[:,2,0], alpha=0.1, color='b')
    plt.plot(sequence_lengths, results[:,0,2], label='Mean '+r'R^2'+' on training set', color='b')
    plt.fill_between(sequence_lengths, results[:,1,2], results[:,2,2], alpha=0.1, color='g')
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
    
def sensitivity_analysis(site, input_columns, model_class, var_name, timestep=None, **model_kwargs):
    # train a model
    model, best, history = train_hparam(model_class, site, input_columns, **model_kwargs)
    device = 'cuda' # TODO

    # given the var_name input (and potentially the time step), calculate the partial derivatives
    # of the model function wrt each input-output

    # TODO: how to get the partial using torch.autograd
    # we need to prepare data and go as far as preparing the dataloader here to have the exact tensors
    # that are tied to the outputs
    # is grads batched allows for batch gradient calculation
    train, _ = prepare_data(site, input_columns, **model_kwargs)
    # one single batch
    dataloader = DataLoader(train, batch_size=1)
    model.eval()
    partials = []
    for _, (X, _) in enumerate(dataloader):
            pred = model(X.to(device))
            # TODO: isolate the value in X that corresponds with var_name:timestep
            partial = autograd.grad(pred, X, is_grads_batched=False)
            partials.append(partial)

    # TODO: analyze the partials: average, distribution, etc.
    mean_partial = sum(partials)/len(partials)
    print(f"The average partial derivative for this variable is {mean_partial}")

    return mean_partial
    
    
me2_input_column_set = [
        'D_SNOW',
        'SWC_4_1_1',
        'RH',
        'PPFD_IN',
        'TS_1_3_1',
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
    MAX_SEQUENCE_LENGTH=14

    #train_test_eval(XGBoost, site, me2_input_column_set, sequence_length=7, flatten=True, lr=[0.001, 0.01,0.1], n_estimators=[1000, 10000, 100000], num_folds=3)
    #train_test_eval(XGBoost, site, me2_input_column_set, sequence_length=31, flatten=True, lr=[0.001, 0.01,0.1], n_estimators=[1000, 10000, 100000], num_folds=3)
    #train_test_eval(XGBoost, site, me2_input_column_set, stat_interval=7,  lr=[0.001, 0.01,0.1], n_estimators=[1000, 10000, 100000], num_folds=3)
    #train_test_eval(XGBoost, site, me2_input_column_set, stat_interval=31,  lr=[0.001, 0.01,0.1], n_estimators=[1000, 10000, 100000], num_folds=3)

    #train_test_eval(RandomForest, site, me2_input_column_set, sequence_length=7, flatten=True,  n_estimators=[1000, 10000], num_folds=3)
    #train_test_eval(RandomForest, site, me2_input_column_set, sequence_length=31, flatten=True,  n_estimators=[1000, 10000], num_folds=3)
    #train_test_eval(RandomForest, site, me2_input_column_set, stat_interval=7,  n_estimators=[1000, 10000], num_folds=3)
    #train_test_eval(RandomForest, site, me2_input_column_set, stat_interval=31,  n_estimators=[1000, 10000], num_folds=3)
    
    plot_sequence_importance(site, me2_input_column_set, XGBoost, max_sequence_length=31, num_models=5, flatten=True)
    plot_sequence_importance(site, me2_input_column_set, XGBoost, num_models=5, max_sequence_length=31, flatten=False)
    
    #plot_sequence_importance(site, me2_input_column_set, XGBoost, max_sequence_length=31, flatten=True)
    #plot_sequence_importance(site, me2_input_column_set, XGBoost, max_sequence_length=31, flatten=False)



                
if __name__=="__main__":
    main()