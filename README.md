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


|     | ANN | Random Forest | XGBoost | RNN | LSTM | xLSTM |
| --- | --- | ------------- | ------- | --- | ---- | ----- |
| daytime | --- | ------------- | ------- | --- | ---- | ----- |
| nighttime | --- | ------------- | ------- | --- | ---- | ----- |
| all time | --- | ------------- | ------- | --- | ---- | ----- |
| 7 day sequence (& flattened) | --- | ------------- | ------- | --- | ---- | ----- |
| 14 day sequence (& flattened) | --- | ------------- | ------- | --- | ---- | ----- |
| 31 day sequence (& flattened) | --- | ------------- | ------- | --- | ---- | ----- |
| 90 day sequence (& flattened) | --- | ------------- | ------- | --- | ---- | ----- |
| seasonal | --- | ------------- | ------- | --- | ---- | ----- |
| dropout | --- | ------------- | ------- | --- | ---- | ----- |
