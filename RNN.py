from enum import Enum
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import tqdm

global input_column_set
global site

Site = Enum('Site', ['Me2', 'Me6'])

class AmeriFLUXDataset(Dataset):
    def __init__(self, df_X_y : pd.DataFrame):
        # hold onto the original dataframe
        self.df = df_X_y
        _df = df_X_y.reset_index()
        self.inputs : pd.DataFrame = _df.drop(columns=['DAY', 'NEE'])
        self.labels : pd.Series = _df['NEE']

    def __len__(self, ):
        return len(self.labels)

    def __getitem__(self, idx):
        input : np.ndarray = self.inputs.iloc[idx].drop("index").to_numpy(dtype=np.float32)
        label : np.ndarray = np.array([self.labels.iloc[idx]], dtype=np.float32)
        return input, label
    
class AmeriFLUXSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.inputs = X
        self.labels = y

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]
        return input, label

site = Site.Me2

# this is the set for Me-2
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

input_column_set = me2_input_column_set

LAYER_FEATURES = 8

# returns a training dataset and a test dataset with sequences of the given length (half hour intervals)
# for reference, 24 hours == length of 48, 1 week == length 336, 1 month (31 days) == 1488, 1 year == 17520
def prepare_data(site_name : Site, sequence_length : int, no_split = False) -> tuple[AmeriFLUXDataset, AmeriFLUXDataset]:
    filepath = ''
    if site_name == Site.Me2:
        filepath = 'AmeriFLUX Data/AMF_US-Me2_BASE-BADM_19-5/AMF_US-Me2_BASE_HH_19-5.csv'
    elif site_name == Site.Me6:
        filepath = 'AmeriFLUX Data/AMF_US-Me6_BASE-BADM_16-5/AMF_US-Me6_BASE_HH_16-5.csv'
    else:
        print(f"Error: {site_name} not a valid site")

    df = pd.read_csv(filepath, header=2)

    # replace -9999 with NaN
    df = df.replace(-9999, np.nan)

    _nrows = len(df)
    # drop all rows where ustar is not sufficient
    df = df[df['USTAR'] > 0.2]
    #print(f"Dropped {_nrows - len(df)}  rows with USTAR threshold ({len(df)})")
    

    # remove daylight hours?
    #df = df[df['PPFD_IN'] > 4.0]

    _nrows = len(df)

    # reduce the columns to our desired set
    target_col = 'NEE_PI' if 'NEE_PI' in df.columns else 'NEE_PI_F'

    global input_column_set
    df['DATETIME'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
    df = df[['DATETIME', *input_column_set, target_col]]
    df["NEE"] = df[target_col]
    df = df.drop(columns=[target_col])

    X_dataset = []
    y_dataset = []
    # TODO: create the sequences for time-series predictors
    print("Building dataset")
    t = tqdm.tqdm(total = len(df)-sequence_length)
    for i in range(len(df)-sequence_length):
        t.update(1)
        # if there are any gaps, skip this iteration
        if df.iloc[i:i+sequence_length].isna().values.any():
            continue
        df_seq = df.iloc[i:i+sequence_length].dropna()
        #t_0 = df_seq["DATETIME"].iloc[0]
        # get time difference from t_0 in minutes, these will be our new columns?
        #delta_t = (df_seq["DATETIME"] - t_0).apply(lambda td: td.days*48 + td.seconds // 60)
        X_seq = df_seq.to_numpy()
        #print(X_seq)
        y_seq = df_seq[["NEE"]].to_numpy()
        X_dataset.append(X_seq)
        y_dataset.append(y_seq)
    print(f"The final size of the dataset is {len(X_dataset)}")
    train_size = int(len(X_dataset)*0.8)
    X_train = np.array(X_dataset[0:train_size])
    y_train = np.array(y_dataset[0:train_size])
    X_eval = np.array(y_dataset[train_size:])
    y_eval = np.array(y_dataset[train_size:])
    print(X_train.shape)
    print(y_train.shape)
    print(X_eval.shape)
    return AmeriFLUXSequenceDataset(X_train, y_train), AmeriFLUXSequenceDataset(X_eval, y_eval)

prepare_data(Site.Me2, 48)