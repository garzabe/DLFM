from torch import nn
import numpy as np

from data_handler import  Site, get_site_vars
from train import train_test_eval, feature_pruning
from model_class import FirstANN, DynamicANN, RNN, LSTM, XGBoost, RandomForest



def test_tte():
    simple_cols = ['P', 'PPFD_IN']
    # This should take little time to run
    print("TTE with DynamicANN")
    train_test_eval(DynamicANN, layer_dims=[(2,),(2,2)], num_folds=2, epochs=5, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64)
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
    print("TTE with RNN")
    train_test_eval(RNN, num_folds=2, epochs=1, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, time_series=True, sequence_length=7)
    print("TTE with RNN and ustar=na")
    train_test_eval(RNN, num_folds=2, site=Site.Me2, epochs=2, input_columns=simple_cols, sequence_length=12, time_series=True, ustar='na')
    print("TTE with RNN and seasonal data (winter)")
    train_test_eval(RNN, num_folds=2, site=Site.Me2, epochs=2, input_columns=simple_cols, sequence_length=12, time_series=True, season='winter')
    print("TTE with LSTM")
    train_test_eval(LSTM, num_folds=2, epochs=1, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, time_series=True, sequence_length=50)

def test_sklearn():
    simple_cols=['P','PPFD_IN']

    print("TTE with XGBoost")
    train_test_eval(XGBoost, num_folds=2, site=Site.Me2, input_columns=simple_cols, lr=[0.1, 1], n_estimators=[10,100], sklearn_model=True)
    print("TTE with RandomForest")
    train_test_eval(RandomForest, num_folds=2, site=Site.Me2, input_columns=simple_cols, n_estimators=[10,100], sklearn_model=True)

def search_longest_sequence(input_columns, ustar=None):
    longest_sequence_low = 1
    longest_sequence_high = 1000
    while longest_sequence_low != longest_sequence_high-1:
        sequence_length = (longest_sequence_low + longest_sequence_high)//2
        print(f"Testing {sequence_length}")
        r2 = train_test_eval(RNN, epochs=1, num_folds=2, input_columns=input_columns, site=Site.Me2, time_series=True, sequence_length=sequence_length, skip_eval=True)
        if r2 == -np.inf:
            print("TTE failed to train a model due to sequence length")
            longest_sequence_high=sequence_length
        else:
            print("TTE trained a model with the given sequence length")
            longest_sequence_low=sequence_length
    print(f"The longest sequence found was {longest_sequence_low}")
    return longest_sequence_low

# TODO: count how many 1-day gaps there are

# TODO: do best_rnn_search, iterating on sequence length to observe the change in performance with lost days

# Does an exhaustive search for the best hyperparameter configuration of a vanilla neural network
# we can optionally include multiple stat intervals to search on as well
def best_vanilla_network_search(site, input_columns, sequence_length=None, flatten=False):
    # To define a model architecture with a single hidden layer, you must add a comma after the layer dimension in the tuple
    # or Python will simplify it to an int
    if not flatten:
        train_test_eval(DynamicANN,
                        layer_dims=[(1, ), (4, ), (8, ), (10, ), (4,4), (6,4), (10,4), (6,6), (10,6), (4,4,4)],
                        num_folds=7,
                        epochs=[100,200,300],
                        site=site,
                        input_columns=input_columns,
                        lr=[1e-3, 1e-2],
                        batch_size=[32,64],
                        stat_interval=sequence_length)
    else:
        train_test_eval(DynamicANN,
                        layer_dims=[(1, ), (4, ), (8, ), (10, ), (4,4), (6,4), (10,4), (6,6), (10,6), (4,4,4)],
                        num_folds=7,
                        epochs=[100,200,300],
                        site=site,
                        input_columns=input_columns,
                        lr=[1e-3, 1e-2],
                        batch_size=[32,64],
                        sequence_length=sequence_length, time_series=True, flatten=True)
    
def best_rnn_search(site, input_columns, model_class = LSTM, sequence_length = None, dropout=0.0):
    train_test_eval(model_class,
                    num_folds=7,
                    epochs=[100,200,300],
                    site=site,
                    input_columns=input_columns,
                    lr=[1e-2, 1e-3],
                    batch_size=[32,64],
                    sequence_length=[7,14,31,62,124] if sequence_length is None else sequence_length, # 1 year is too long...
                    hidden_state_size=[4,8,15],
                    num_layers=[1,2,3],
                    dropout=dropout,
                    time_series=True)


def main():
    site = Site.Me2
    me2_input_column_set = [
        'D_SNOW',
        # no data until 2006
        'SWC_1_7_1',
        # 2 7 1 has really spotty data
        #'SWC_2_7_1',
        #'SWC_3_7_1',
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
    train_test_eval(DynamicANN,
                    num_folds=2,
                    epochs=30,
                    site=Site.Me2,
                    input_columns=me2_input_column_set,
                    layer_dims=(6,4),
                    lr=[1e-2],
                    batch_size=64,
                    #sequence_length=31,
                    #hidden_state_size=8,
                    #num_layers=1,
                    #dropout=0.0,
                    #time_series=True,
                    num_models=10)


                
if __name__=="__main__":
    main()