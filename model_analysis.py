from torch import nn
import numpy as np

from data_handler import  Site, get_site_vars
from train import train_test_eval, feature_pruning
from model_class import FirstANN, DynamicANN, RNN, LSTM, XGBoost, RandomForest

import matplotlib.pyplot as plt

default_hparams = {DynamicANN: {'layer_dims': (6,6), 'epochs': 300, 'batch_size': 64, 'lr': [0.01, 0.001]},
                   RNN: {'hidden_state_size': [8, 15], 'num_layers': [1,2], 'epochs': 200, 'batch_size': 64, 'lr': 0.01},
                   LSTM: {'hidden_state_size': [8, 15], 'num_layers': 1, 'epochs': 200, 'batch_size': 32, 'lr': 0.01},
                   XGBoost: {'lr': 0.5, 'n_estimators': 1000},
                   RandomForest: {'n_estimators': 1000}}

def test_tte():
    simple_cols = ['P', 'PPFD_IN', 'D_SNOW']
    # This should take little time to run
    
    print("TTE with DynamicANN")
    train_test_eval(DynamicANN, site=Site.Me2, input_columns=simple_cols, layer_dims=[(2,),(2,2)], num_folds=2, epochs=5, lr=1e-2, batch_size=64)
    print("TTE with DynamicANN & multiple stat intervals")
    train_test_eval(DynamicANN, layer_dims=[(2,),(2,2)], num_folds=2, epochs=5, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, stat_interval=[None,7,14])
    print("TTE with DynamicANN & ustar=na")
    train_test_eval(DynamicANN, layer_dims=[(2,),(2,2)], num_folds=2, epochs=5, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, ustar='na')
    # test time series + flatten for linear models
    print("TTE with DynamicANN & flattened time series data")
    train_test_eval(DynamicANN, layer_dims=[(2,),(2,2)], num_folds=2, epochs=5, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, time_series=True, sequence_length=7, flatten=True)
    print("TTE with DynamicANN and seasonal data (summer)")
    train_test_eval(DynamicANN, site=Site.Me2, num_folds=2, input_columns=simple_cols, epochs=2, season='summer')
    
    # test time series data preparation and RNN predictor model
    print("TTE with RNN and match sequence length")
    train_test_eval(RNN, site=Site.Me2, input_columns=simple_cols, num_folds=2, epochs=1, lr=1e-2, batch_size=64, time_series=True, sequence_length=7, match_sequence_length=31)
    print("TTE with RNN")
    train_test_eval(RNN, site=Site.Me2, input_columns=simple_cols, num_folds=2, epochs=1, slr=1e-2, batch_size=64, time_series=True, sequence_length=7)
    print("TTE with RNN and ustar=na")
    train_test_eval(RNN, site=Site.Me2, input_columns=simple_cols, num_folds=2, sequence_length=12, time_series=True, ustar='na')
    print("TTE with RNN and seasonal data (winter)")
    train_test_eval(RNN, site=Site.Me2, input_columns=simple_cols, num_folds=2, sequence_length=12, time_series=True, season='winter')
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
                   RNN: {'hidden_state_size': 15, 'num_layers': 1, 'epochs': 2000, 'batch_size': 64, 'lr': 0.001, 'weight_decay': 0.01}, # [8, 15]
                   LSTM: {'hidden_state_size': 8, 'num_layers': 3, 'epochs': 500, 'batch_size': 64, 'lr': 0.01, 'weight_decay': 0.001, 'dropout':0.001,'momentum': 0.00}, # [8, 15]
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
    
        # TA 1 1 1 has no data until 2007
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
    ### Testing scripts - these usually evoke any bugs present in the project
    #test_sklearn()
    #test_tte()
    plot_sequence_importance(site, me2_input_column_set, RNN, max_sequence_length=31, num_models=50, dropout=0.01)
    plot_sequence_importance(site, me2_input_column_set, LSTM, max_sequence_length=31, num_models=50, dropout=0.01)

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
