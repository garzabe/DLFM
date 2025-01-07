import pandas as pd
from enum import Enum
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import tqdm

Site = Enum('Site', ['Me2', 'Me6'])

class AmeriFLUXDataset(Dataset, ABC):
    @abstractmethod
    def __init__(self, ):
        pass

    @abstractmethod
    def get_train_test_idx(self, delta_year : int) -> tuple[list[int], list[int]]:
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

class AmeriFLUXLinearDataset(Dataset):
    def __init__(self, df_X_y : pd.DataFrame):
        # hold onto the original dataframe
        self.df = df_X_y.reset_index()
        self.years = self.df['DAY'].str[:4].unique()
        self.inputs : pd.DataFrame = self.df.drop(columns=['DAY', 'NEE'])
        self.labels : pd.Series = self.df['NEE']

    def __len__(self, ):
        return len(self.labels)
    
    # return the train and test index ranges for a single fold
    # with one year left out for test
    def get_train_test_idx(self, delta_year : int) -> tuple[list[int], list[int]]:
        year = self.years[-1-delta_year]
        test_year_match = self.df['DAY'].str.match(rf'^{year}\d\d\d\d$')
        return self.df[~test_year_match].index.to_list(), self.df[test_year_match].index.to_list()


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

def get_data(site : Site):
    if not isinstance(site, Site):
        raise ValueError("the provided site is invalid")
    filepath = ''
    if site == Site.Me2:
        filepath = 'AmeriFLUX Data/AMF_US-Me2_BASE-BADM_19-5/AMF_US-Me2_BASE_HH_19-5.csv'
    elif site == Site.Me6:
        filepath = 'AmeriFLUX Data/AMF_US-Me6_BASE-BADM_16-5/AMF_US-Me6_BASE_HH_16-5.csv'

    data = pd.read_csv(filepath, header=2).replace(-9999, np.nan)

    return data

def prepare_data(site_name : Site, input_columns : list[str], no_split = False, eval_years : int = 2, **kwargs) -> tuple[AmeriFLUXDataset, AmeriFLUXDataset]:
    df = get_data(site_name)
    stat_interval = kwargs.get('stat_interval', None)

    # reduce the columns to our desired set
    target_col = 'NEE_PI' if 'NEE_PI' in df.columns else 'NEE_PI_F'

    if input_columns is not None:
        df = df[['TIMESTAMP_START', *input_columns, 'USTAR', target_col]]
    df["NEE"] = df[target_col]
    df = df.drop(columns=[target_col])

    rolling_columns = [col + '_rolling_var' for col in input_columns] + [col + '_rolling_avg' for col in input_columns]

    for col in input_columns:
        rolling_series = df[col].rolling(window=48*stat_interval, min_periods=5*stat_interval)
        df[col+'_rolling_var'] = rolling_series.var()
        df[col+'_rolling_avg'] = rolling_series.mean()


    """
    _nrows = len(df)
    temporal_columns = []
    if stat_interval is not None:
        halfhour_interval = stat_interval*48
        print("Adding temporal data...")
        for input_column in input_columns:
            temporal_avg_col = input_column + "_prev_avg"
            temporal_var_col = input_column + '_prev_var'
            temporal_columns.append(temporal_avg_col)
            temporal_columns.append(temporal_var_col)
            df[temporal_avg_col] = pd.Series()
            df[temporal_var_col] = pd.Series()
        t = tqdm.tqdm(total = (_nrows - halfhour_interval)*len(input_columns))
        for i in range(halfhour_interval, len(df)):
            # if there are any gaps, skip this iteration
            if df[[*input_columns]].iloc[i-halfhour_interval:i].isna().values.any():
                t.update(len(input_columns))
            else:
                for input_column in input_columns:
                    temporal_avg_col = input_column + "_prev_avg"
                    temporal_var_col = input_column + "_prev_var"
                    t.update(1)
                    sequence = df[input_column].iloc[i-halfhour_interval:i]
                    print(sequence.mean())
                    print(sequence.var())
                    df.at[i, temporal_avg_col] = sequence.mean()
                    df.at[i, temporal_var_col] = sequence.var()
    """


    # drop all rows where ustar is not sufficient
    df = df[df['USTAR'] > 0.2].drop(columns=['USTAR'])
    #print(f"Dropped {_nrows - len(df)}  rows with USTAR threshold ({len(df)})")
    

    df = df[df['PPFD_IN'] > 4.0]

    _nrows = len(df)

    

    # group into daily averages
    df['DATETIME'] = pd.to_datetime(df['TIMESTAMP_START'], format="%Y%m%d%H%M")
    df['DAY'] = df['DATETIME'].apply(lambda dt: f"{dt.year:04}{dt.month:02}{dt.day:02}")
    df = df.drop(columns=['DATETIME', 'TIMESTAMP_START'])
    
    # the means are the important values, but count helps us identify low-data days
    df_avg = df.groupby('DAY').aggregate('mean').reset_index()
    df_count = df.groupby('DAY').aggregate('count').reset_index()

    # now only include the days where all column counts are above 20??
    # perfect recording is 48 per day
    # with ~9 hours of daylight, the max daylight rows is 18
    _nrows = len(df_avg)
    min_count = 5
    #print(df_count.head())
    min_count_filter = df_count.drop(columns=['DAY']) >= min_count
    #print(df_avg[~min_count_filter])
    df_X_y = df_avg[min_count_filter.all(axis=1)]
    #print(df_X_y.head())
    #df_X_y = df_X_y.drop(columns=['DAY'])
    print(f"Dropped {_nrows - len(df_X_y)} rows from min count filter ({len(df_X_y)})")
    print(df_X_y.head(100))

    # TODO: normalize all data
    # add these columns back at the end
    _df = df_X_y.drop(columns=["DAY", "NEE"])
    _df = (_df - _df.mean())/_df.std()
    _df["DAY"] = df_X_y["DAY"]
    _df["NEE"] = df_X_y["NEE"]

    # simple train-eval and eval 80/20 split
    train_size = int(len(_df)*0.8)
    # TODO: modular design to day of year split
    # determine where the eval set starts
    first_eval_year = int(_df["DAY"].iloc[-1][:4]) - (eval_years - 1)
    eval_year_range = [str(y) for y in range(first_eval_year, 2022)]
    year_match_str = '|'.join(eval_year_range)
    _df_eval = _df[_df["DAY"].str.match(rf'^{year_match_str}\d\d\d\d$')]
    _df = _df[~_df["DAY"].str.match(rf'^{year_match_str}\d\d\d\d$')]
    print(f"The training set has {len(_df)} entries")
    print(f"The eval set has {len(_df_eval)} entries")

    ## TODO: remove this, we are no longer using default train test splitting
    # split in to train, test, validation
    #if not no_split:
    #    df_train, df_test = train_test_split(_df, test_size=0.2)
    #    print(f"The training set has {len(df_train)} entries, and the test set has {len(df_test)} entries")
    #else:
    #    df_train = _df
    #    df_test = None
    #df_train = _df
    #return AmeriFLUXDataset(df_train), AmeriFLUXDataset(df_test) if df_test is not None else None
    return AmeriFLUXLinearDataset(_df), AmeriFLUXLinearDataset(_df_eval)
    

def prepare_timeseries_data(site : Site, input_columns : list[str], sequence_length : int, no_split = False) -> tuple[AmeriFLUXDataset, AmeriFLUXDataset]:
    df = get_data(site)

    _nrows = len(df)
    # drop all rows where ustar is not sufficient
    df = df[df['USTAR'] > 0.2]
    #print(f"Dropped {_nrows - len(df)}  rows with USTAR threshold ({len(df)})")
    

    # remove daylight hours?
    #df = df[df['PPFD_IN'] > 4.0]

    _nrows = len(df)

    # reduce the columns to our desired set
    target_col = 'NEE_PI' if 'NEE_PI' in df.columns else 'NEE_PI_F'

    df['DATETIME'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
    if input_columns is not None:
        df = df[['DATETIME', *input_columns, target_col]]
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
        df_seq = df.iloc[i:i+sequence_length].dropna().drop(columns=['DATETIME'])
        #t_0 = df_seq["DATETIME"].iloc[0]
        # get time difference from t_0 in minutes, these will be our new columns?
        #delta_t = (df_seq["DATETIME"] - t_0).apply(lambda td: td.days*48 + td.seconds // 60)
        X_seq = df_seq.drop(columns=["NEE"]).to_numpy(dtype=np.float32)
        #print(X_seq)
        y = df_seq[["NEE"]].iloc[-1].to_numpy(dtype=np.float32)
        X_dataset.append(X_seq)
        y_dataset.append(y)
    print(f"The final size of the dataset is {len(X_dataset)}")
    train_size = int(len(X_dataset)*0.8)
    X_train = np.array(X_dataset[0:train_size])
    y_train = np.array(y_dataset[0:train_size])
    X_eval = np.array(y_dataset[train_size:])
    y_eval = np.array(y_dataset[train_size:])
    return AmeriFLUXSequenceDataset(X_train, y_train), AmeriFLUXSequenceDataset(X_eval, y_eval)