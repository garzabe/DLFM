from torch import nn

from data_handler import AmeriFLUXDataset, prepare_data, Site
from train import train_test_eval
from model_class import FirstANN, DynamicANN, RNN


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
    # comparing performance of first ann and two layer ann
    train_test_eval(DynamicANN, layer_dims=[(5,5), (10, 10)], num_folds=5, epochs=[50, 100], site=site, input_columns=me2_input_column_set, stat_interval=[3,7,14])
    #first_r2 = train_test_eval(FirstANN)
    #results = [('original', first_r2)]
    """
    layer_sizes = [4,6,8,10,12,14,20]
    for l1 in layer_sizes:
        arch = [l1]
        r2 = train_test_eval(DynamicANN, [arch, nn.ReLU])
        with open('output.txt', 'a+') as f:
                dt = datetime.datetime.now()
                f.write(f"{dt.year}-{dt.month:02}-{dt.day:02} {dt.hour:02}:{dt.minute:02}:{dt.second:02} : Architecture {arch} R-Squared {r2:.4f}\n")        
    for l1 in layer_sizes:
        for l2 in layer_sizes:
            arch = [l1, l2]
            r2 = train_test_eval(DynamicANN, [arch, nn.ReLU])
            with open('output.txt', 'a+') as f:
                dt = datetime.datetime.now()
                f.write(f"{dt.year}-{dt.month:02}-{dt.day:02} {dt.hour:02}:{dt.minute:02}:{dt.second:02} : Architecture {arch} R-Squared {r2:.4f}\n")        
    """
                
if __name__=="__main__":
    main()