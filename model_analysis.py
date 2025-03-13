from torch import nn, optim
import numpy as np

from data_handler import  Site, get_site_vars
from train import train_test_eval, fmt_date_string #, feature_pruning
from model_class import FirstANN, DynamicANN, RNN, LSTM, XGBoost, RandomForest

import matplotlib.pyplot as plt

default_hparams = {DynamicANN: {'layer_dims': (6,6), 'epochs': 300, 'batch_size': 64, 'lr': 0.001, 'weight_decay': 0.1},
                   RNN: {'hidden_state_size': 15, 'num_layers': 1, 'epochs': 2000, 'batch_size': 64, 'lr': 0.001, 'weight_decay': 0.1}, # [8, 15]
                   LSTM: {'hidden_state_size': 8, 'num_layers': 1, 'epochs': 2000, 'batch_size': 64, 'lr': 0.001, 'weight_decay': 0.1}, # [8, 15]
                   XGBoost: {'lr': 0.5, 'n_estimators': 10000},
                   RandomForest: {'n_estimators': 10000}}

def test_tte():
    simple_cols = ['P', 'PPFD_IN', 'D_SNOW']
    # This should take little time to run
    
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
    print("TTE with RNN and match sequence length")
    train_test_eval(RNN, site=Site.Me2, input_columns=simple_cols, num_folds=2, epochs=1, lr=1e-2, batch_size=64, time_series=True, sequence_length=7, match_sequence_length=31)
    # print("TTE with RNN")
    # train_test_eval(RNN, site=Site.Me2, input_columns=simple_cols, num_folds=2, epochs=1, slr=1e-2, batch_size=64, time_series=True, sequence_length=7)
    # print("TTE with RNN and ustar=na")
    # train_test_eval(RNN, site=Site.Me2, input_columns=simple_cols, num_folds=2, sequence_length=12, time_series=True, ustar='na')
    # print("TTE with RNN and seasonal data (winter)")
    # train_test_eval(RNN, site=Site.Me2, input_columns=simple_cols, num_folds=2, sequence_length=12, time_series=True, season='winter')
    print("TTE with LSTM")
    train_test_eval(LSTM, site=Site.Me2, input_columns=simple_cols, num_folds=2, epochs=1, lr=1e-2, batch_size=64, time_series=True, sequence_length=50)

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

def plot_sequence_importance(site, input_columns, model_class, max_sequence_length=90, flatten=False, **kwargs):
    r2_results = []
    mse_results = []
    r2_t_results = []
    mse_t_results = []
    sequence_args = default_hparams[model_class]
    sequence_args.update(kwargs)
    sequence_args['time_series'] = True
    sequence_args['match_sequence_length'] = max_sequence_length

    if flatten and model_class.__name__ in ['RNN', 'LSTM']:
        print("Argument error: model class cannot be an RNN and take flattened time series data")
        return
    
    use_stat_interval = model_class.__name__ not in ['RNN', 'LSTM'] and not flatten
    if use_stat_interval:
        sequence_args['time_series'] = False
    
    sequence_lengths = list(range(1, max_sequence_length+1, 3))
    for sl in range(1, max_sequence_length+1, 3):
        sl_arg = {'stat_interval' if use_stat_interval else 'sequence_length': sl}
        r2, mse, r2_t, mse_t = train_test_eval(model_class, site, input_columns, **sl_arg, **sequence_args)
        r2_results.append(r2)
        r2_t_results.append(r2_t)
        mse_results.append(mse)
        mse_t_results.append(mse_t)

    #plt.rcParams['text.usetex'] = True
    dt_str = fmt_date_string()
    # R-squared on both evaluation and training sets
    plt.clf()
    plt.plot(sequence_lengths, r2_results, label='Mean '+r'R^2'+' on evaluation set')
    plt.plot(sequence_lengths, r2_t_results, label='Mean '+r'R^2'+' on training set')
    plt.xlabel('Input Sequence Length (Days)')
    plt.ylabel('R^2')
    plt.legend()
    plt.ylim((0.5, 1.0))
    plt.title(f'Importance of Sequence Length for {model_class.__name__} Predictions')
    plt.savefig(f'images/sequence_length_importance-{dt_str}::{model_class.__name__}-r2.png')

    # MSE on both evaluation and training sets
    plt.clf()
    plt.plot(sequence_lengths, mse_results, label='Mean MSE on evaluation set')
    plt.plot(sequence_lengths, mse_t_results, label='Mean MSE on training set')
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
    
me2_input_column_set = [
        'D_SNOW',
        # no data until 2006
        'SWC_1_7_1',
        # 2 7 1 has really spotty data
        #'SWC_2_7_1',
        #'SWC_3_7_1',
        'SWC_4_1_1',
        'SWC_1_2_1',
        'RH',
        'NETRAD',
        'PPFD_IN',
        'TS_1_3_1',
        #'V_SIGMA',
        'P',
        'WD',
        'WS',
        # TA 1 1 1 has no data until 2007
        'TA_1_1_3',
        # Trying out some new variables
        'G_2_1_1',
        'H',
        'LW_IN',
        'SW_IN',
        'H2O',
        'CO2',
        'LE'
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
    # me2_input_column_set = [
    #     'D_SNOW',
    #     # no data until 2006
    #     'SWC_1_7_1',
    #     # 2 7 1 has really spotty data
    #     #'SWC_2_7_1',
    #     #'SWC_3_7_1',
    #     'SWC_1_2_1',
    #     'RH',
    #     'NETRAD',
    #     'PPFD_IN',
    #     'TS_1_3_1',
    #     #'V_SIGMA',
    #     'P',
    #     'WD',
    #     'WS',
    #     # TA 1 1 1 has no data until 2007
    #     'TA_1_1_3',
    # ]

    # me6_input_column_set = [
    #     'D_SNOW',
    #     'SWC_1_5_1',
    #     'SWC_1_2_1',
    #     'RH',
    #     'NETRAD',
    #     'PPFD_IN',
    #     'TS_1_5_1',
    #     'P',
    #     'WD',
    #     'WS',
    #     'TA_1_1_2'
    # ]
    MAX_SEQUENCE_LENGTH=14

    #train_test_eval(LSTM, site, me2_input_column_set, optimizer_class=optim.Adam, lr=0.001)
    #train_test_eval(LSTM, site, me2_input_column_set, optimizer_class=optim.SGD, weight_decay=0.001)
    #train_test_eval(LSTM, site, me2_input_column_set, optimizer_class=optim.SGD, weight_decay=0.00, momentum=0.1)


    ### Preliminary hparam tuning to find strictly better parameters
    train_test_eval(LSTM, site, me2_input_column_set, lr=[0.01, 0.001], batch_size=[32,64], num_layers=[1,2], epochs=[500, 2000], sequence_length=7)
    train_test_eval(LSTM, site, me2_input_column_set, lr=[0.01, 0.001], batch_size=[32,64], num_layers=[1,2], epochs=[500, 2000], sequence_length=14)



    # new batch of hparam tuning - will take a long time to run
    # ~300 combos per model, 5 mins per combo, 12 models = ~12 days to run in total
    # check back in 3 days
    # 3 day sequences
    # best_rnn_search(site, me2_input_column_set, 3, model_class=LSTM, weight_decay=[0.0, 0.01, 0.001], momentum=[0.0, 0.1, 0.2])
    # best_vanilla_network_search(site, me2_input_column_set, sequence_length=3)
    # best_vanilla_network_search(site, me2_input_column_set, sequence_length=3, flatten=True)

    # check back in 3 days
    # 7 day sequences
    # best_rnn_search(site, me2_input_column_set, 7, model_class=LSTM, weight_decay=[0.1, 0.01, 0.0], momentum=[0.0, 0.1, 0.2])
    # best_vanilla_network_search(site, me2_input_column_set, sequence_length=7)
    # best_vanilla_network_search(site, me2_input_column_set, sequence_length=7, flatten=True)

    # # check back in 3 days
    # # 14 day sequences
    # best_rnn_search(site, me2_input_column_set, 14, model_class=LSTM, weight_decay=[0.1, 0.01, 0.0], momentum=[0.0, 0.1, 0.2])
    # best_vanilla_network_search(site, me2_input_column_set, sequence_length=14)
    # best_vanilla_network_search(site, me2_input_column_set, sequence_length=14, flatten=True)

    # # check back in 3 days
    # # 31 day sequences
    # best_rnn_search(site, me2_input_column_set, 31, model_class=LSTM, weight_decay=[0.1, 0.01, 0.0], momentum=[0.0, 0.1, 0.2])
    # best_vanilla_network_search(site, me2_input_column_set, sequence_length=31)
    # best_vanilla_network_search(site, me2_input_column_set, sequence_length=31, flatten=True)



    ### Testing scripts - these usually evoke any bugs present in the project
    #test_sklearn()
    #test_tte()
    # Ensuring that these models are actually converging to some optimum by checking the training curves
    #train_test_eval(DynamicANN, site, me2_input_column_set, epochs=2000, lr=0.001, layer_dims=(6,6), stat_interval=14)
    #train_test_eval(LSTM, site, me2_input_column_set, epochs=[2000, 10000], lr=[0.01, 0.001], weight_decay=[0, 0.1, 0.01], sequence_length=7, match_sequence_length=14, num_folds=3)
    #plot_sequence_importance(site, me2_input_column_set, LSTM, max_sequence_length=MAX_SEQUENCE_LENGTH, num_models=10, num_folds=1)

    #plot_sequence_importance(site, me2_input_column_set, RNN, max_sequence_length=MAX_SEQUENCE_LENGTH, num_models=10, num_folds=2)

    #plot_sequence_importance(site, me2_input_column_set, RandomForest, max_sequence_length=MAX_SEQUENCE_LENGTH, num_models=1, num_folds=1)
    #plot_sequence_importance(site, me2_input_column_set, RandomForest, max_sequence_length=MAX_SEQUENCE_LENGTH, num_models=1, num_folds=1, flatten=False)

    #plot_sequence_importance(site, me2_input_column_set, XGBoost, max_sequence_length=MAX_SEQUENCE_LENGTH, num_models=1, num_folds=1)
    #plot_sequence_importance(site, me2_input_column_set, XGBoost, max_sequence_length=MAX_SEQUENCE_LENGTH, num_models=1, num_folds=1, flatten=False)

    #plot_sequence_importance(site, me2_input_column_set, DynamicANN, max_sequence_length=MAX_SEQUENCE_LENGTH, num_models=10)
    #plot_sequence_importance(site, me2_input_column_set, DynamicANN, max_sequence_length=MAX_SEQUENCE_LENGTH, num_models=10, flatten=False)

    ### Example usage for the best_***_search functions
    #best_vanilla_network_search(site, me2_input_column_set, stat_interval=[None, 7, 14, 30])
    #best_rnn_search(site, me2_input_column_set, model_class=LSTM)
    #best_rnn_search(site, me2_input_column_set, model_class=RNN)

    ### RNN, LSTM, DynamicANN (stat interval and flatenned) with 7,14,31,90 days and 0.0, 0.01, 0.05 dropout
    # for sl in [7,14,31,90,180]:
    #     for d in [0.0, 0.01, 0.05]:
    #         best_rnn_search(Site.Me2, me2_input_column_set, RNN, sequence_length=sl, dropout=d)
    #         best_rnn_search(Site.Me2, me2_input_column_set, LSTM, sequence_length=sl, dropout=d)
    #     best_vanilla_network_search(Site.Me2, me2_input_column_set, sequence_length=sl, flatten=False)
    #     best_vanilla_network_search(Site.Me2, me2_input_column_set, sequence_length=sl, flatten=True)
    # train_test_eval(DynamicANN,
    #                 site=Site.Me2,
    #                 input_columns=me2_input_column_set,
    #                 num_folds=2,
    #                 epochs=50, 
    #                 layer_dims=(6,4),
    #                 lr=[1e-2],
    #                 batch_size=64,
    #                 #sequence_length=31,
    #                 #hidden_state_size=8,
    #                 #num_layers=1,
    #                 #dropout=0.0,
    #                 #time_series=True,
    #                 num_models=10)


                
if __name__=="__main__":
    main()