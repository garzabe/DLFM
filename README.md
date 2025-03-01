Milestone: The following is the list of ML architectures I plan on training and optimizing:

ANN
Random Forest
XGBoost
RNN
LSTM
xLSTMRN

sequence_length_search with the "default" input columns and ustar=None(drop) found a longest sequence length of 211
"" with ustar='na' found the same longest sequence length
with all hours of data included (no daylight filtering) and ustar='na', the longest sequence length is 236


Hyperparameter options to train on:
- flattened time series data (for not RNN, LSTM, xLSTM)
- stat interval data (for not RNN, LSTM, xLSTM)
- summer/winter seasonal split
- with dropout


Notes from meetings with Loren and Kristen 2/26-27:
- we can determine importance of each day of memory by iterating on the sequence length and observing the change in loss (plot this)
    - once we identify "important" days, we can do a sensitivity analysis on that specific day's inputs for more fine-grained analysis
- seasonal split can be done on first snow to last snow for simple yet accurate splits (better than solstices)
- Loren dropped any day with >1/2 readings
- we can try a linear interpolation on the input data to fill any 1-day gaps. How many of those are there in the dataset?
- how well does a model do on nighttime data, rather than daytime data?
- let's include RMSE in the history as another performance dimension to compare on
- once we can get sets of predictions from multiple models, we can make plots on specific input variables and observe the remaining variance (a form of PCA)

Notes from last Alan Meeting:
- what if we include the previous day NEP (NEE) as an input
- other regularization methods...