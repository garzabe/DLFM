from torch import nn

from data_handler import  Site, get_site_vars
from train import train_test_eval, feature_pruning
from model_class import FirstANN, DynamicANN, RNN, LSTM



def test_tte():
    simple_cols = ['P', 'PPFD_IN']
    # This should take little time to run
    #train_test_eval(DynamicANN, layer_dims=[(2,),(2,2)], num_folds=2, epochs=5, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64)
    #train_test_eval(DynamicANN, layer_dims=[(2,),(2,2)], num_folds=2, epochs=5, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, stat_interval=[None,7,14])
    # test time series + flatten for linear models
    train_test_eval(DynamicANN, layer_dims=[(2,),(2,2)], num_folds=2, epochs=5, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, time_series=True, sequence_length=7, flatten=True)
    # test time series data preparation and RNN predictor model
    train_test_eval(RNN, num_folds=2, epochs=1, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, time_series=True, sequence_length=7)
    train_test_eval(LSTM, num_folds=2, epochs=1, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, time_series=True, sequence_length=50)

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

    test_tte()

    #best_vanilla_network_search(site, me2_input_column_set, stat_interval=[None, 7, 14, 30])

    #best_rnn_search(site, me2_input_column_set, model_class=LSTM)
    #best_rnn_search(site, me2_input_column_set, model_class=RNN)
    #train_test_eval(RNN, time_series=True, sequence_length=31, num_folds=5, epochs=300, site=site, input_columns=me2_input_column_set, batch_size=64, lr=1e-2, eval_years=2, num_layers=1)

                
if __name__=="__main__":
    main()