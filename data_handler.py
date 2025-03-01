import pandas as pd
from enum import Enum
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import tqdm
import bisect
import datetime

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
    ustar = kwargs.get('ustar', 'drop') # defines the method of handling low ustar entries: current possible values are: drop, na
    season = kwargs.get('season', None) # summer, winter
    
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
    if ustar == 'drop'or not time_series:
        df = df[df['USTAR'] > 0.2]
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

    # TODO: if we can interpolate
    if False:
        date_diffs = pd.DataFrame(df_avg['DAY'])
        date_diffs['TIMEDIFF'] = date_diffs['DAY'].diff()
        # Day | timediff between Day and previous row Day
        # if 
        single_day_gaps = (date_diffs['TIMEDIFF'] == pd.Timedelta(days=2))
        # insert the single days between single-day gaps and interpolate values (except NEE)
        prev_days = pd.DataFrame(columns=df_avg.columns)
        prev_days_count = pd.DataFrame(columns=df_count.columns)
        # now we have all missing single-gap days in the form of the df_avg dataframe
        prev_days['DAY'] = date_diffs[single_day_gaps]['DAY'] - pd.Timedelta(days=1)
        prev_days_count['DAY'] = date_diffs[single_day_gaps]['DAY'] - pd.Timedelta(days=1)
        prev_days_count.fillna(1)
        
        df_avg = pd.concat([df_avg, prev_days]).sort_values(by='DAY').interpolate(limit=1)
        df_count = pd.concat([df_count, prev_days_count]).sort_values(by='DAY')


    # now only include the days where all column counts are above 20??
    # perfect recording is 48 per day
    # with ~9 hours of daylight, the max daylight rows is 18
    _nrows = len(df_avg)
    # TODO: if there aren't enough readings in a day and its surrounded by good days, impute from the surrounding days
    # is very local linear interp okay @Loren ?
    min_count = 1
    #print(df_count.head())
    min_count_filter = df_count.drop(columns=['DAY']) >= min_count
    #print(df_avg[~min_count_filter])
    df_X_y = df_avg[min_count_filter.all(axis=1)]
    # remove any NEE values with ustar below threshold
    if ustar=='na':
        df_X_y.loc[df_X_y['USTAR'] <= 0.2, 'NEE'] = np.NaN
    #print(df_X_y.head())
    #df_X_y = df_X_y.drop(columns=['DAY'])
    if len(df_X_y) == 0:
        print("No data after filtering")
        return None, None

    # normalize all data
    # add these columns back at the end
    _df = df_X_y.drop(columns=["DAY", "NEE", "USTAR"])
    _df = (_df - _df.mean())/_df.std()
    _df["DAY"] = df_X_y["DAY"]
    _df["NEE"] = df_X_y["NEE"]


    if time_series:
        X_dataset = []
        y_dataset = []
        years_idx = []
        dates = []
        def in_season(d : datetime.datetime, s : str):
            if s is None:
                return True
            if s == 'winter':
                return d - datetime.datetime(d.year if d.month==12 else d.year-1, 12, 21) <= datetime.timedelta(days=90)
            if s == 'summer':
                return d - datetime.datetime(d.year, 12, 21) <= datetime.timedelta(days=90)
            else:
                return False
        t = tqdm.tqdm(total = len(_df)-sequence_length)
        for i in range(len(_df)-sequence_length):
            t.update(1)
            # if there are any gaps, skip this iteration
            if _df['DAY'].iloc[i] + pd.Timedelta(sequence_length, 'day') != _df['DAY'].iloc[i+sequence_length]:
                continue
            # if the datapoint is not in the desired season, skip
            if not in_season(_df['DAY'].iloc[-1], season):
                continue
            df_seq = _df.iloc[i:i+sequence_length]
            # if the final day NEE is NaN, skip this iteration
            if pd.isna(df_seq['NEE']).iloc[-1]:
                continue
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
        print(f"There are {len(years_ref)} unique years in the dataset")
        if len(years_ref) <= eval_years:
            print(f"There are not enough unique years in the dataset to have {eval_years} evaluation years")
            if len(years_ref)==1:
                print("There is only one year in the dataset. Reverting to a 80-20 train-eval split. WARNING: This model will probably not be good")
                eval_idx = (len(years_idx)*8)//10
            else:
                print(f"Overriding to split the dataset into 1 training year and {len(years_ref)-1} eval years")
                eval_years = len(years_ref)-1
        if len(years_ref)>1:
            years_eval = years_ref[-eval_years:]
            final_training_year = years_ref[-eval_years-1]
            print(f"Using {years_eval} as the evaluation years")
            eval_idx = bisect.bisect(years_idx, final_training_year)
        X_train = np.array(X_dataset[0:eval_idx])
        y_train = np.array(y_dataset[0:eval_idx])
        dates_train = np.array(dates[0:eval_idx])
        X_eval = np.array(X_dataset[eval_idx:])
        y_eval = np.array(y_dataset[eval_idx:])
        dates_eval = np.array(dates[eval_idx:])
        train_years_idx = years_idx[0:eval_idx]
        eval_years_idx = years_idx[eval_idx:]
        print(f"The training set has {len(X_train)} rows and the evaluation set has {len(X_eval)} rows")
        if len(X_eval) <= 1:
            print("The evaluation set does not have enough data to compute an R-squared metric")
            return None, None
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
            return AmeriFLUXLinearDataset(df_train), AmeriFLUXLinearDataset(df_eval)
        else:
            return AmeriFLUXSequenceDataset(X_train, y_train, dates_train, train_years_idx), AmeriFLUXSequenceDataset(X_eval, y_eval, dates_eval, eval_years_idx)

    else:
        if season is not None:
            solstice_mo = 6 if season=='winter' else 12 if season == 'summer' else None
            _df['PREV_SOLSTICE'] = pd.to_datetime(_df['DAY'].apply(lambda dt: f"{dt.year:04}{solstice_mo:02}21"))
            # e.g. day is 1-12-2020 and summer solstice is 6-21-2020, then solstice diff is negative
            # day is 1-21-2020 and winter solstice is 12-21-2020, solstice diff is negative but <90 days (mod 365 days)
            solstice_diff = (_df['DAY'] - _df['PREV_SOLSTICE']).apply(lambda td: td.days) % 365
            _df = _df[solstice_diff < 90].drop(columns=['PREV_SOLSTICE'])

        eval_year_range = np.unique(_df['DAY'].dt.year)[-eval_years:]
        _df_eval = _df[_df["DAY"].dt.year.isin(eval_year_range)]
        _df = _df[~(_df["DAY"].dt.year.isin(eval_year_range))]
        return AmeriFLUXLinearDataset(_df), AmeriFLUXLinearDataset(_df_eval)