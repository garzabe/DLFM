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

# Does an exhaustive search for the best hyperparameter configuration of a vanilla neural network
# we can optionally include multiple stat intervals to search on as well
def best_vanilla_network_search(site, input_columns, stat_interval=None):
    # To define a model architecture with a single hidden layer, you must add a comma after the layer dimension in the tuple
    # or Python will simplify it to an int
    train_test_eval(DynamicANN,
                    layer_dims=[(1, ), (4, ), (8, ), (10, ), (4,4), (6,4), (10,4), (6,6), (10,6), (4,4,4)],
                    num_folds=7,
                    epochs=[100,200,300],
                    site=site,
                    input_columns=input_columns,
                    lr=[1e-3, 1e-2],
                    batch_size=[32,64],
                    stat_interval=stat_interval)
    
def best_rnn_search(site, input_columns, model_class = LSTM, sequence_length = None):
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
                    dropout=[0.0, 0.01, 0.1],
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

    input_columns = get_site_vars(Site.Me2)
    # these columns are high in NA values and make it hard to find a feature set to train on
    # 
    bad_columns =  ['TIMESTAMP_END', 'G_1_1_1', 'G_6_1_1', 'G_7_1_1', 'G_8_1_1', 'ALB'] +['RECO_PI_F', 'WS_MAX', 'TA_2_2_1', 'TA_2_2_2', 'RH_2_2_1', 'PPFD_IN_2_2_1', 'SW_IN_2_2_1', 'SW_OUT_2_2_1', 'LW_IN_2_2_1', 'LW_OUT_2_2_1', 'NETRAD_2_2_1', 'PA_2_2_1']
    for bad_col in bad_columns:
        input_columns.remove(bad_col)
    #feature_pruning(DynamicANN, site, num_folds=7, epochs=100, batch_size=64, lr=1e-2, layer_dims=(6,6), input_columns=input_columns)
    #train_test_eval(DynamicANN, site=site, input_columns=me2_input_column_set, num_folds=7, epochs=100, batch_size=64, lr=1e-2, layer_dims=(6,6))

    #search_longest_sequence(me2_input_column_set, ustar='na')
    #test_sklearn()
    test_tte()
    #train_test_eval(XGBoost, num_folds=7, site=Site.Me2, input_columns=me2_input_column_set, lr=[0.5, 0.9, 1], n_estimators=[100, 1000, 10000], sklearn_model=True)
    #train_test_eval(RandomForest, num_folds=7, site=Site.Me2, input_columns=me2_input_column_set, n_estimators=[10,100, 1000], sklearn_model=True)

    #best_vanilla_network_search(site, me2_input_column_set, stat_interval=[None, 7, 14, 30])

    #best_rnn_search(site, me2_input_column_set, model_class=LSTM)
    #best_rnn_search(site, me2_input_column_set, model_class=RNN)
    #train_test_eval(RNN, time_series=True, sequence_length=60, num_folds=5, epochs=300, site=site, input_columns=me2_input_column_set, batch_size=64, lr=1e-2, eval_years=2, num_layers=1, ustar='na')

                
if __name__=="__main__":
    main()