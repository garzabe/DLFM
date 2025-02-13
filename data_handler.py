import pandas as pd
from enum import Enum
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import tqdm
import bisect

Site = Enum('Site', ['Me2', 'Me6'])

class AmeriFLUXDataset(Dataset, ABC):
    @abstractmethod
    def __init__(self, ):
        pass

    @abstractmethod
    def get_train_test_idx(self, delta_year : int) -> tuple[list[int], list[int]]:
        pass

    @abstractmethod
    def get_dates(self, idx_range : list[int]) -> list:
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def get_X(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_y(self) -> np.ndarray:
        pass

class AmeriFLUXLinearDataset(Dataset):
    def __init__(self, df_X_y : pd.DataFrame):
        # hold onto the original dataframe
        self.df = df_X_y.reset_index()
        self.years = self.df['DAY'].dt.year.unique()
        self.inputs : pd.DataFrame = self.df.drop(columns=['DAY', 'NEE'])
        self.labels : pd.Series = self.df['NEE']

    def __len__(self, ):
        return len(self.labels)
    
    # return the train and test index ranges for a single fold
    # with one year left out for test
    def get_train_test_idx(self, delta_year : int) -> tuple[list[int], list[int]]:
        if delta_year > len(self.years):
            print(f"Warning: delta_year ({delta_year}) is greater than the number of years in the dataset ({len(self.years)})")
            return None, None
        year = self.years[-1-delta_year]
        test_year_match = self.df['DAY'].dt.year == year
        return self.df[~test_year_match].index.to_list(), self.df[test_year_match].index.to_list()
    
    def get_dates(self, idx_range : list[int]):
        return self.df['DAY'].iloc[idx_range].to_list()

    def get_num_years(self):
        return len(self.years)

    def __getitem__(self, idx):
        input : np.ndarray = self.inputs.iloc[idx].drop("index").to_numpy(dtype=np.float32)
        label : np.ndarray = np.array([self.labels.iloc[idx]], dtype=np.float32)
        return input, label
    
    # returns the input data as a numpy array for the use with ML packages other than pytorch
    def get_X(self):
        return self.inputs.to_numpy()
    
    def get_y(self):
        return self.labels.to_numpy()
    
class AmeriFLUXSequenceDataset(Dataset):
    def __init__(self, X, y, dates, years_idx):
        self.inputs = X
        self.labels = y
        self.dates = dates
        self.years_idx = years_idx
        self.years = np.unique(self.years_idx)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]
        return input, label

    def get_num_years(self):
        return len(self.years)
    
    def get_train_test_idx(self, delta_year : int) -> tuple[list[int], list[int]]:
        if delta_year > len(self.years):
            print(f"Warning: delta_year ({delta_year}) is greater than the number of years in the dataset ({len(self.years)})")
            return None, None
        year = self.years[-1-delta_year]
        year_lo = bisect.bisect(self.years_idx, year-1)
        year_hi = bisect.bisect(self.years_idx, year)
        return list(range(0, year_lo)) + list(range(year_hi, len(self.years_idx))), list(range(year_lo, year_hi))
    
    def get_dates(self, idx_range : list[int]):
        return [self.dates[i] for i in idx_range]
    
    def get_X(self):
        return self.inputs
    
    def get_y(self):
        return self.labels

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

def get_site_vars(site : Site):
    if not isinstance(site, Site):
        raise ValueError("the provided site is invali")
    data = get_data(site)
    return data.columns.to_list()

def prepare_data(site_name : Site, eval_years : int = 2, **kwargs) -> tuple[AmeriFLUXDataset, AmeriFLUXDataset]:
    df = get_data(site_name)
    stat_interval = kwargs.get('stat_interval', None)
    input_columns = kwargs.get('input_columns', df.columns)
    time_series = kwargs.get('time_series', False)
    sequence_length = kwargs.get('sequence_length', -1)
    flatten = kwargs.get('flatten', False)

    # if we want time series data, then jump to the time series data preparator instead
    #if time_series:
    #    return prepare_timeseries_data(site_name, input_columns, sequence_length)

    # reduce the columns to our desired set
    target_col = 'NEE_PI' if 'NEE_PI' in df.columns else 'NEE_PI_F'

    if input_columns is not None:
        df = df[['TIMESTAMP_START', *input_columns, 'USTAR', target_col]]
    df["NEE"] = df[target_col]
    df = df.drop(columns=[target_col])
    if stat_interval is not None:
        for col in input_columns:
            rolling_series = df[col].rolling(window=48*stat_interval, min_periods=5*stat_interval)
            df[col+'_rolling_var'] = rolling_series.var()
            df[col+'_rolling_avg'] = rolling_series.mean()


    # drop all rows where ustar is not sufficient
    # TODO: does USTAR apply to just NEE or all input variables @Loren
    df = df[df['USTAR'] > 0.2].drop(columns=['USTAR'])
    #print(f"Dropped {_nrows - len(df)}  rows with USTAR threshold ({len(df)})")
    
    # daylight hours
    df = df[df['PPFD_IN'] > 4.0]

    _nrows = len(df)

    # group into daily averages
    df['DATETIME'] = pd.to_datetime(df['TIMESTAMP_START'], format="%Y%m%d%H%M")
    df['DAY'] = pd.to_datetime(df['DATETIME'].apply(lambda dt: f"{dt.year:04}{dt.month:02}{dt.day:02}"))
    df = df.drop(columns=['DATETIME', 'TIMESTAMP_START'])
    
    # the means are the important values, but count helps us identify low-data days
    df_avg = df.groupby('DAY').aggregate('mean').reset_index()
    df_count = df.groupby('DAY').aggregate('count').reset_index()

    # now only include the days where all column counts are above 20??
    # perfect recording is 48 per day
    # with ~9 hours of daylight, the max daylight rows is 18
    _nrows = len(df_avg)
    # TODO: if there aren't enough readings in a day and its surrounded by good days, impute from the surrounding days
    # is very local linear interp okay @Loren ?
    min_count = 5
    #print(df_count.head())
    min_count_filter = df_count.drop(columns=['DAY']) >= min_count
    #print(df_avg[~min_count_filter])
    df_X_y = df_avg[min_count_filter.all(axis=1)]
    #print(df_X_y.head())
    #df_X_y = df_X_y.drop(columns=['DAY'])
    print(len(df_X_y))
    if len(df_X_y) == 0:
        print("No data after filtering")
        return None, None

    # normalize all data
    # add these columns back at the end
    _df = df_X_y.drop(columns=["DAY", "NEE"])
    _df = (_df - _df.mean())/_df.std()
    _df["DAY"] = df_X_y["DAY"]
    _df["NEE"] = df_X_y["NEE"]


    if time_series:
        X_dataset = []
        y_dataset = []
        years_idx = []
        dates = []
        print("Building dataset")
        # TODO: is there a way to hack the rolling window to quicky generate the sequence data we need?
        #print(pd.DataFrame(df.rolling(window=sequence_length)))
        # This doesn't work
        #print(pd.DataFrame(df.rolling(window=sequence_length)).apply(lambda r: r.dropna(ignore_index=True), axis=1))
        # perhaps iterate over columns and generate the sequences for each column and ~~combine~~
        t = tqdm.tqdm(total = len(_df)-sequence_length)
        for i in range(len(_df)-sequence_length):
            t.update(1)
            # if there are any gaps, skip this iteration
            if _df['DAY'].iloc[i] + pd.Timedelta(sequence_length, 'day') != _df['DAY'].iloc[i+sequence_length]:
                continue
            df_seq = _df.iloc[i:i+sequence_length]
            year = df_seq['DAY'].dt.year.iloc[-1]
            date = df_seq['DAY'].iloc[-1]
            df_seq = df_seq.drop(columns=['DAY'])

            X_seq = df_seq.drop(columns=["NEE"]).to_numpy(dtype=np.float32)
            #print(X_seq)
            y = df_seq[["NEE"]].iloc[-1].to_numpy(dtype=np.float32)
            X_dataset.append(X_seq)
            y_dataset.append(y)
            years_idx.append(year)
            dates.append(date)
        # sequence too long, empty dataset
        if len(X_dataset) == 0:
            return None, None
        years_ref = np.unique(years_idx)
        years_eval = years_ref[-eval_years:]
        eval_idx = bisect.bisect(years_idx, years_eval[0])
        X_train = np.array(X_dataset[0:eval_idx])
        y_train = np.array(y_dataset[0:eval_idx])
        dates_train = np.array(dates[0:eval_idx])
        X_eval = np.array(X_dataset[eval_idx:])
        y_eval = np.array(y_dataset[eval_idx:])
        dates_eval = np.array(dates[eval_idx:])
        train_years_idx = years_idx[0:eval_idx]
        eval_years_idx = years_idx[eval_idx:]
        if flatten:
            train_size = len(X_train)
            eval_size = len(X_eval)
            input_size = len(X_train[0][0])
            X_df_train = pd.DataFrame(X_train.reshape((train_size, sequence_length*input_size)))
            y_df_train = pd.DataFrame(y_train.reshape((train_size, 1)), columns=['NEE'])
            dates_df_train = pd.DataFrame(dates_train.reshape((train_size, 1)), columns=['DAY'])
            df_train = pd.concat([X_df_train, y_df_train, dates_df_train], axis=1)

            X_df_eval = pd.DataFrame(X_eval.reshape((eval_size, sequence_length*input_size)))
            y_df_eval = pd.DataFrame(y_eval.reshape((eval_size, 1)), columns=['NEE'])
            dates_eval = pd.DataFrame(dates_eval.reshape((eval_size, 1)), columns=['DAY'])
            df_eval = pd.concat([X_df_eval, y_df_eval, dates_eval], axis=1)
            print(df_train.head())
            return AmeriFLUXLinearDataset(df_train), AmeriFLUXLinearDataset(df_eval)
        else:
            return AmeriFLUXSequenceDataset(X_train, y_train, dates_train, train_years_idx), AmeriFLUXSequenceDataset(X_eval, y_eval, dates_eval, eval_years_idx)

    else:
        eval_year_range = np.unique(_df['DAY'].dt.year)[-eval_years:]
        _df_eval = _df[_df["DAY"].dt.year.isin(eval_year_range)]
        _df = _df[~(_df["DAY"].dt.year.isin(eval_year_range))]
        return AmeriFLUXLinearDataset(_df), AmeriFLUXLinearDataset(_df_eval)
    

def prepare_timeseries_data(site : Site, input_columns : list[str], sequence_length : int, eval_years : int = 2) -> tuple[AmeriFLUXDataset, AmeriFLUXDataset]:
    if sequence_length < 1:
        raise ValueError(f"Error: The sequence length cannot be less than 1 ({sequence_length})")
    df = get_data(site)

    # reduce the columns to our desired set
    target_col = 'NEE_PI' if 'NEE_PI' in df.columns else 'NEE_PI_F'

    df['DATETIME'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
    if input_columns is not None:
        df = df[['DATETIME', *input_columns, 'USTAR', target_col]]
    df["NEE"] = df[target_col]
    df = df.drop(columns=[target_col])

    X_dataset = []
    y_dataset = []
    years_idx = []
    print("Building dataset")
    # TODO: is there a way to hack the rolling window to quicky generate the sequence data we need?
    #print(pd.DataFrame(df.rolling(window=sequence_length)))
    # This doesn't work
    #print(pd.DataFrame(df.rolling(window=sequence_length)).apply(lambda r: r.dropna(ignore_index=True), axis=1))
    # perhaps iterate over columns and generate the sequences for each column and ~~combine~~
    t = tqdm.tqdm(total = len(df)-sequence_length)
    for i in range(len(df)-sequence_length):
        t.update(1)
        # if there are any gaps or bad readings, skip this iteration
        if df.iloc[i:i+sequence_length].isna().values.any() or \
            (df['USTAR'] <= 0.2).iloc[i:i+sequence_length].any():
            continue
        df_seq = df.iloc[i:i+sequence_length].drop(columns=[ 'USTAR'])
        year = df_seq['DATETIME'].dt.year.iloc[-1]
        df_seq = df_seq.drop(columns=['DATETIME'])

        X_seq = df_seq.drop(columns=["NEE"]).to_numpy(dtype=np.float32)
        #print(X_seq)
        y = df_seq[["NEE"]].iloc[-1].to_numpy(dtype=np.float32)
        X_dataset.append(X_seq)
        y_dataset.append(y)
        years_idx.append(year)
    print(f"The final size of the dataset is {len(X_dataset)}")
    years_ref = np.unique(years_idx)
    years_eval = years_ref[-eval_years:]
    eval_idx = bisect.bisect(years_idx, years_eval[0])
    X_train = np.array(X_dataset[0:eval_idx])
    y_train = np.array(y_dataset[0:eval_idx])
    X_eval = np.array(X_dataset[eval_idx:])
    y_eval = np.array(y_dataset[eval_idx:])
    train_years_idx = years_idx[0:eval_idx]
    eval_years_idx = years_idx[eval_idx:]
    return AmeriFLUXSequenceDataset(X_train, y_train, train_years_idx), AmeriFLUXSequenceDataset(X_eval, y_eval, eval_years_idx)