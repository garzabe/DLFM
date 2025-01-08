from torch import nn

from data_handler import AmeriFLUXDataset, prepare_data, Site
from train import train_test_eval
from model_class import FirstANN, DynamicANN, RNN, LSTM


"""
# Visualize model performance, and
# calculate R-Squared value
r2metric = R2Score(device=device)
full_data, _ = prepare_data(site, no_split=True)
full_df = full_data.df
eval_dataloader = DataLoader(full_data, batch_size=len(full_data), shuffle=False)
X, _y = next(iter(eval_dataloader))
pred : torch.Tensor = model(X)
r2metric.update(input=pred, target=_y)
print(f"The model achieves an r-squared value of {r2metric.compute().item():.4f}")
pred_s = [a[0] for a in pred.detach().numpy()]
y = full_df['NEE'].to_list()
dates = pd.to_datetime(full_df['DAY'], format="%Y%m%d").to_list()
x = [d.date() for d in dates]

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
limit = 365
plt.plot(x[:limit], y[:limit], label='Actual NEE')
plt.plot(x[:limit], pred_s[:limit], label='Predicted NEE')
plt.xticks([x[i] for i in range(0, limit, limit//10)])
plt.gcf().autofmt_xdate()
plt.ylabel("NEE")
plt.legend()
plt.title('NEE Model Predictions')
plt.show()
"""


# A Dyanmic Programming feature pruning algorithm
#  to determine the best feature set for a given model architecture
#  
def DP_feature_pruning(model_class, *model_args, **kwargs):
    num_folds = kwargs.get('num_folds', 5)
    epochs = kwargs.get('epochs', 100)
    site = kwargs.get('site', Site.Me2)

    # .... still has the same shortcoming as feature addition

def test_tte():
    simple_cols = ['P', 'PPFD_IN']
    # This should take little time to run
    #train_test_eval(DynamicANN, layer_dims=[(2,),(2,2)], num_folds=2, epochs=1, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64)
    #train_test_eval(DynamicANN, layer_dims=[(2,),(2,2)], num_folds=2, epochs=1, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, stat_interval=7)
    # test time series data preparation and RNN predictor model
    #train_test_eval(RNN, num_folds=2, epochs=1, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, time_series=True, sequence_length=7)
    train_test_eval(LSTM, num_folds=2, epochs=1, site=Site.Me2, input_columns=simple_cols, lr=1e-2, batch_size=64, time_series=True, sequence_length=7)

# Does an exhaustive search for the best hyperparameter configuration of a vanilla neural network
# we can optionally include multiple stat intervals to search on as well
def best_vanilla_network_search(site, input_columns, stat_interval=None):
    # To define a model architecture with a single hidden layer, you must add a comma after the layer dimension in the tuple
    # or Python will simplify it to an int
    train_test_eval(DynamicANN,
                    layer_dims=[(1, ), (4, ), (8, ), (10, ), (4,4), (6,4), (10,4), (6,6), (10,6), (4,4,4)],
                    num_folds=7,
                    epochs=100,
                    site=site,
                    input_columns=input_columns,
                    lr=[1e-3, 1e-2],
                    batch_size=[32,64],
                    stat_interval=stat_interval)


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

    #best_vanilla_network_search(site, me2_input_column_set)

    test_tte()

    #train_test_eval(LSTM, time_series=True, sequence_length=7, num_folds=5, epochs=100, site=site, input_columns=me2_input_column_set, batch_size=64, lr=1e-2)

                
if __name__=="__main__":
    main()