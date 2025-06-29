import pandas as pd
from enum import Enum
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import tqdm
import bisect
import datetime
import re
import os

#Site = Enum('Site', ['Me2', 'Me6'])

class AmeriFLUXDataset(Dataset, ABC):
    @abstractmethod
    def __init__(self, ):
        pass

    @abstractmethod
    def get_train_test_idx(self, delta_year : int) -> tuple[list[int], list[int]]:
        pass

    @abstractmethod
    def get_dates(self, idx_range : list[int] = None) -> list[datetime.datetime]:
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
    def __init__(self, df_X_y : pd.DataFrame, means = None, stds = None):
        # hold onto the original dataframe
        self.df = df_X_y.reset_index(drop=True)

        # use season years
        #self.years = self.df['DAY'].dt.year.unique()
        self.years = self.df['SEASON_YEAR'].unique()
        self.years.sort()
        self.vars = self.df.drop(columns=['DAY', 'SEASON_YEAR', 'NEE']).columns.to_list()

        self.inputs : pd.DataFrame = self.df.drop(columns=['DAY', 'SEASON_YEAR', 'NEE'])
        self.labels : pd.Series = self.df['NEE']
        self.means = means
        self.stds = stds

    def __len__(self, ):
        return len(self.labels)
    
    # return the train and test index ranges for a single fold
    # with one year left out for test
    def get_train_test_idx(self, delta_year : int) -> tuple[list[int], list[int]]:
        if delta_year > len(self.years):
            print(f"Warning: delta_year ({delta_year}) is greater than the number of years in the dataset ({len(self.years)})")
            return None, None
        year = self.years[-1-delta_year]
        test_year_match = self.df['SEASON_YEAR'] == year
        return self.df[~test_year_match].index.to_list(), self.df[test_year_match].index.to_list()
    
    def get_dates(self, idx_range : list[int]=None):
        if idx_range is not None:
            return self.df['DAY'].iloc[idx_range].to_list()
        else:
            return self.df['DAY'].to_list()   
        
    def get_var_idx(self, var):
        if var not in self.vars:
            raise ValueError(f"Error: the variable {var} is not present")
        return self.vars.index(var)

    def get_num_years(self):
        return len(self.years)

    def __getitem__(self, idx):
        input : np.ndarray = self.inputs.iloc[idx].to_numpy(dtype=np.float32)
        label : np.ndarray = np.array([self.labels.iloc[idx]], dtype=np.float32)
        return input, label
    
    # returns the input data as a numpy array for the use with ML packages other than pytorch
    def get_X(self):
        return self.inputs.to_numpy()
    
    def get_y(self):
        return self.labels.to_numpy()
    
class AmeriFLUXSequenceDataset(Dataset):
    def __init__(self, X, y, dates, years_idx, vars, means = None, stds = None):
        self.inputs = X
        self.labels = y
        self.dates = dates
        self.years_idx = years_idx
        self.years = np.unique(self.years_idx)
        self.vars = vars
        self.means = means
        self.stds = stds

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]
        return input, label

    def get_num_years(self):
        return len(self.years)
    
    def get_train_test_idx(self, delta_year : int) -> tuple[list[int], list[int]]:
        if delta_year >= len(self.years):
            print(f"Warning: delta_year ({delta_year}) is greater than the number of years in the dataset ({len(self.years)})")
            return None, None
        year = self.years[-1-delta_year]
        year_lo = bisect.bisect(self.years_idx, year-1)
        year_hi = bisect.bisect(self.years_idx, year)
        return list(range(0, year_lo)) + list(range(year_hi, len(self.years_idx))), list(range(year_lo, year_hi))
    
    def get_dates(self, idx_range : list[int] = None):
        if idx_range is not None:
            return [self.dates[i] for i in idx_range]
        else:
            return self.dates
        
    def get_var_idx(self, var):
        if var not in self.vars:
            raise ValueError(f"Error: the variable {var} is not present")
        return self.vars.index(var)
    
    def get_X(self):
        return self.inputs
    
    def get_y(self):
        return self.labels

def get_data(data_filepath : str):
    if not os.path.exists(data_filepath):
        raise ValueError(f"The provided path {data_filepath} is not valid")
    #if not isinstance(site, Site):
    #    raise ValueError("the provided site is invalid")
    #filepath = ''
    #if site == Site.Me2:
    #    filepath = 'AmeriFLUX Data/AMF_US-Me2_BASE-BADM_20-5/AMF_US-Me2_BASE_HH_20-5.csv'
    #elif site == Site.Me6:
    #    filepath = 'AmeriFLUX Data/AMF_US-Me6_BASE-BADM_17-5/AMF_US-Me6_BASE_HH_17-5.csv'

    data = pd.read_csv(data_filepath, header=2).replace(-9999, np.nan)

    return data

def get_site_vars(data_filepath : str):
    #if not isinstance(site, Site):
    #    raise ValueError("the provided site is invalid")
    data = get_data(data_filepath)
    return data.columns.to_list()

def prepare_data(data_filepath : str, input_columns : list[str], eval_years : int = 3, **kwargs) -> tuple[AmeriFLUXDataset, AmeriFLUXDataset]:

    ### Arguments for building time series data

    # Sequence length (in days) of the time series inputs - can range from 1 - ~200 days
    sequence_length = kwargs.get('sequence_length', None)

    # Due to limited data, longer sequence length arguments result in significantly smaller datasets
    # Force smaller sequence length datasets to match datapoints to a longer sequence length
    # Useful for comparing models across sequence lengths
    match_sequence_length = kwargs.get('match_sequence_length', None)

    # Flatten time series data - useful for non-sequence models
    flatten = kwargs.get('flatten', False)
    # Rolling window statistics for use with DynamicANN and other non-sequence models
    stats = kwargs.get('stats', False)

    # defines the method of handling low ustar entries: current possible values are: drop, na
    ustar = kwargs.get('ustar', 'na') 

    # Filter through only one season of data
    # Options are:
    # summer (last snow of winter to first snow of winter in same calendar year)
    # winter (first snow to last snow)
    season = kwargs.get('season', None)

    # Include the target variable from the previous day
    yesterday_NEE = kwargs.get('yesterday_NEE', False)

    # Interpolate 1-day gaps in weather or climate data
    interpolate = kwargs.get('interpolate', True)

    time_series = sequence_length is not None
    peak_NEE = kwargs.get('peak_NEE', False)
    doy = kwargs.get('doy', False)

    # in the case we are using match_sequence_length, first get the set of prediction dates for the target dataset
    match_dates_train = None
    match_dates_eval = None
    if match_sequence_length is not None and sequence_length != match_sequence_length:
        print("Generating the reference dataset for our actual dataset to match")
        match_dataset_train, match_dataset_eval = prepare_data(data_filepath, input_columns, eval_years=eval_years, sequence_length=match_sequence_length,
                                     ustar=ustar, flatten=flatten, season=season, interpolate=interpolate)
        match_dates_train = match_dataset_train.get_dates()
        match_dates_eval = match_dataset_eval.get_dates()

    df = get_data(data_filepath)
    
    # reduce the columns to our desired feature set
    target_col = 'NEE_PI' if 'NEE_PI' in df.columns else 'NEE_PI_F'
    # if no input columns given, use a default set that is (generally) guaranteed to exist in the dataset
    df = df[['TIMESTAMP_START', *input_columns, 'USTAR', target_col]]
    df["NEE"] = df[target_col]
    df = df.drop(columns=[target_col])

    # get rolling window average and variance
    if stats:
        for col in input_columns:
            rolling_series = df[col].rolling(window=48*sequence_length, min_periods=5*sequence_length)
            df[col+'_rolling_var'] = rolling_series.var()
            df[col+'_rolling_avg'] = rolling_series.mean()

    
    # drop all rows where ustar is not sufficient
    if ustar == 'drop' or not time_series:
        df = df[df['USTAR'] > 0.2]
    
    # daylight hours
    df = df[df['PPFD_IN'] > 4.0]

    # group into daily averages
    df['DATETIME'] = pd.to_datetime(df['TIMESTAMP_START'], format="%Y%m%d%H%M")
    df['DAY'] = pd.to_datetime(df['DATETIME'].apply(lambda dt: f"{dt.year:04}{dt.month:02}{dt.day:02}"))

    df_X = df.drop(columns=['NEE'])
    df_y = df[['DATETIME', 'DAY', 'NEE']]

    # only use morning hours 9-11 for ~peak NEE calculation
    if peak_NEE:
        df_y = df_y[(df_y['DATETIME'].dt.hour >= 9) & (df_y['DATETIME'].dt.hour <= 11)]

    df_y = df_y.drop(columns=['DATETIME'])
    df_X = df_X.drop(columns=['DATETIME', 'TIMESTAMP_START'])
    
    # the means are the important values, but count helps us identify low-data days
    df_X_avg = df_X.groupby('DAY').aggregate('mean').reset_index()
    df_X_count = df_X.groupby('DAY').aggregate('count').reset_index()
    df_y_avg = df_y.groupby('DAY').aggregate('mean').reset_index()
    df_y_count = df_y.groupby('DAY').aggregate('count').reset_index()

    # now only include the days where all column counts are above 20??
    # perfect recording is 48 per day
    # with ~9 hours of daylight, the max daylight rows is 18
    min_count =  9  #or 10 for full day, 3 for just the morning
    min_NEE_count = 3 if peak_NEE else 9

    # A mask for missing/low-sample NEE days
    nee_below_threshold = (df_y_avg['NEE'] == np.nan) | (df_y_count['NEE'] < min_NEE_count)

    # Replace missing/low-sample NEE days with NaN
    df_y_avg.loc[nee_below_threshold, 'NEE'] = np.nan

    # drop rows in df_X with NaN values
    X_is_na = df_X_avg.notna().all(axis=1)
    df_X_avg = df_X_avg[X_is_na]
    df_X_count = df_X_count[X_is_na]

    # Mask for low-sample days
    min_count_filter = (df_X_count.drop(columns=['DAY']) >= min_count).all(axis=1)
    df_X_avg = df_X_avg[min_count_filter]
    df_X_y = df_X_avg.merge(df_y_avg, on='DAY', how='left')

    # remove any NEE values with ustar below threshold if doing replace rather than drop
    if ustar=='na':
        df_X_y.loc[df_X_y['USTAR'] <= 0.2, 'NEE'] = np.nan

    # Too many high-gap features can result in empty datasets after pre-processing
    if len(df_X_y) == 0:
        print("No data after filtering")
        return None, None

    if interpolate:
        date_diffs = pd.DataFrame(df_X_y['DAY'])
        date_diffs['TIMEDIFF'] = date_diffs['DAY'].diff()

        # Day | timediff between Day and previous row Day
        single_day_gaps = (date_diffs['TIMEDIFF'] == pd.Timedelta(days=2))

        # insert the single days between single-day gaps and interpolate values (except NEE)
        prev_days = pd.DataFrame(columns=df_X_y.columns)
        
        # now we have all missing single-gap days in the form of the df_avg dataframe
        prev_days['DAY'] = date_diffs[single_day_gaps]['DAY'] - pd.Timedelta(days=1)
        
        # interpolate the input data
        df_interp = pd.concat([df_X_y, prev_days]).reset_index().sort_values(by='DAY')

        # remember where the gaps are to remove NEE after interpolate
        gap_filled = df_interp.isna().any(axis=1)
        df_interp.interpolate(limit=1, inplace=True)
        df_interp.loc[gap_filled, 'NEE'] = np.nan
        df_X_y = df_interp.drop(columns=['index'])

    
    # assign seasons and season years
    # default to winter, and change datapoints to summer as needed
    season_df = df_X_y[['DAY', 'D_SNOW']].assign(SEASON='winter')
    season_df.loc[:, 'YEAR'] = season_df['DAY'].dt.year
    # safe to iterate on years with <20 years of data
    years = season_df['YEAR'].unique()
    years.sort()
    season_df = season_df.assign(SEASON_YEAR=min(years))
    for year in years:
        year_df = season_df[season_df['YEAR']==year]
        # assume the last snow happens before august
        jan_june_df = year_df[year_df['DAY'].dt.month <= 6]
        # assume first snow happens after june
        aug_dec_df = year_df[year_df['DAY'].dt.month >= 7]
        # filter on snow depth > 0
        jan_june_snow = jan_june_df.loc[jan_june_df['D_SNOW'] > 0]
        # filter on snow depth > 0
        aug_dec_snow = aug_dec_df.loc[aug_dec_df['D_SNOW'] > 0]
        last_snow_idx = -1
        first_snow_idx = -1
        if len(jan_june_df) == 0 or len(jan_june_snow) == 0:
            # if there is no snow data for the first half of the year (or no data at all)
            # -> have the last snow index be the index of the first entry of aug_dec_df (assume it starts in summer)
            # aug_dec_df is guaranteed to have data since otherwise this year wouldn't have been iterated on
            last_snow_idx = aug_dec_df.index[0]
        else:
            # Get the index of the last matching row for the last snow
            last_snow_idx = jan_june_snow[-1:].index[0]
        if len(aug_dec_df) == 0 or len(aug_dec_snow) == 0:
            # have first snow be the index after the last row in jan_june
            # again, guaranteed to exist in this case since the year has at least one row in the dataset
            first_snow_idx = jan_june_df.index[-1] + 1
        else:
            # Get the index of the first matching row for the first snow
            first_snow_idx = aug_dec_snow.index[0]

        # Finally, label every row between the discovered indices as summer
        season_df.loc[last_snow_idx+1:first_snow_idx, 'SEASON'] = 'summer'
        # assign each row after the start of this season-year to the current year
        # later years will get updated in later iterations
        season_df.loc[last_snow_idx+1:, 'SEASON_YEAR'] = year
    
    df_X_y = df_X_y.assign(SEASON_YEAR=season_df['SEASON_YEAR'])

    # if training on a specific season, filter out other season
    if season is not None:
        df_X_y = df_X_y[season_df['SEASON'] == season]

        
    # normalize all remaining data
    # add these columns back at the end
    _df = df_X_y.drop(columns=["DAY", "NEE", "SEASON_YEAR", "USTAR"])
    means = _df.mean()
    stds = _df.std()
    _df = (_df - _df.mean())/_df.std()
    _df["DAY"] = df_X_y["DAY"]
    _df["NEE"] = df_X_y["NEE"]
    _df["SEASON_YEAR"] = df_X_y['SEASON_YEAR']
    if doy:
        _df["DOY"] = (_df["DAY"].dt.day_of_year - 183)/105.66

    vars = pd.DataFrame(_df.drop(columns=['SEASON_YEAR', 'DAY', 'NEE'])).columns.to_list()

    if time_series:
        X_dataset = []
        y_dataset = []
        years_idx = []
        dates = []
        t = tqdm.tqdm(total = len(_df)-sequence_length)
        for i in range(len(_df)-sequence_length):
            t.update(1)
            datapoint_date = _df['DAY'].iloc[i+sequence_length]
            # if there are any gaps, skip this iteration
            if _df['DAY'].iloc[i] + pd.Timedelta(sequence_length, 'day') != datapoint_date:
                continue
            # if we are using match_sequence_length and the prediction date is not in the reference dataset, skip
            if match_sequence_length is not None and sequence_length != match_sequence_length and \
                    datapoint_date not in match_dates_train and datapoint_date not in match_dates_eval:
                continue

            df_seq = _df.iloc[i:i+sequence_length]

            # if the final day NEE is NaN, skip this iteration
            if pd.isna(df_seq['NEE']).iloc[-1]:
                continue

            year = df_seq['SEASON_YEAR'].iloc[-1]

            date = df_seq['DAY'].iloc[-1]
            df_seq = df_seq.drop(columns=['DAY', 'SEASON_YEAR'])

            X_seq = df_seq.drop(columns=["NEE"]).to_numpy(dtype=np.float32)
            #print(X_seq)
            y = df_seq[["NEE"]].iloc[-1].to_numpy(dtype=np.float32)
            X_dataset.append(X_seq)
            y_dataset.append(y)
            years_idx.append(year)
            dates.append(date)
        t.close()
        # sequence too long, empty dataset
        if len(X_dataset) == 0:
            return None, None
        years_ref = np.unique(years_idx)
        print(f"Dataset resulted in the following season years: {years_ref}")
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
            eval_idx = bisect.bisect(years_idx, final_training_year)
        X_train = np.array(X_dataset[0:eval_idx])
        y_train = np.array(y_dataset[0:eval_idx])
        dates_train = np.array(dates[0:eval_idx])
        X_eval = np.array(X_dataset[eval_idx:])
        y_eval = np.array(y_dataset[eval_idx:])
        dates_eval = np.array(dates[eval_idx:])
        train_years_idx =np.array(years_idx[0:eval_idx])
        eval_years_idx = np.array(years_idx[eval_idx:])
        print(f"The training set has {len(X_train)} rows and the evaluation set has {len(X_eval)} rows")
        if len(X_eval) <= 1:
            print("The evaluation set does not have enough data to compute an R-squared metric")
            return None, None
        if flatten:
            print("Flattening")
            train_size = len(X_train)
            eval_size = len(X_eval)
            input_size = len(X_train[0][0])
            X_df_train = pd.DataFrame(X_train.reshape((train_size, sequence_length*input_size)))
            y_df_train = pd.DataFrame(y_train.reshape((train_size, 1)), columns=['NEE'])
            dates_df_train = pd.DataFrame(dates_train.reshape((train_size, 1)), columns=['DAY'])
            years_df_train = pd.DataFrame(train_years_idx.reshape((train_size, 1)), columns=['SEASON_YEAR'])
            df_train = pd.concat([X_df_train, y_df_train, dates_df_train, years_df_train], axis=1)

            X_df_eval = pd.DataFrame(X_eval.reshape((eval_size, sequence_length*input_size)))
            y_df_eval = pd.DataFrame(y_eval.reshape((eval_size, 1)), columns=['NEE'])
            dates_eval = pd.DataFrame(dates_eval.reshape((eval_size, 1)), columns=['DAY'])
            years_eval = pd.DataFrame(eval_years_idx.reshape((eval_size, 1)), columns=['SEASON_YEAR'])
            df_eval = pd.concat([X_df_eval, y_df_eval, dates_eval, years_eval], axis=1)
            return AmeriFLUXLinearDataset(df_train), AmeriFLUXLinearDataset(df_eval)
        else:
            return AmeriFLUXSequenceDataset(X_train, y_train, dates_train, train_years_idx, vars=vars, means=means, stds=stds), AmeriFLUXSequenceDataset(X_eval, y_eval, dates_eval, eval_years_idx, vars=vars, means=means, stds=stds)

    else:
        # if we are not generating time series data, we do not want any nans to remain in the dataset
        _df = _df.dropna()

        # only include rows that are found in the reference dataset
        if match_sequence_length is not None:
            _df = _df[_df['DAY'].isin(np.concatenate([match_dates_train, match_dates_eval]))]

        eval_year_range = np.unique(_df['SEASON_YEAR'])[-eval_years:]

        _df_eval = _df[_df["SEASON_YEAR"].isin(eval_year_range)]
        _df = _df[~(_df["SEASON_YEAR"].isin(eval_year_range))]
        return AmeriFLUXLinearDataset(_df), AmeriFLUXLinearDataset(_df_eval)
    

# Determines the difference in dataset size with and without the given col
# Useful for determining a feature set, as some variables have a lot of gaps that reduce
# the dataset size by as much as ~500 datapoints
def get_dataset_size_diff(data_filepath, input_columns: list[str], col:str, sequence_length=None):
    if col not in input_columns:
        print("Error: column not in the input set")
        return
    train, test = prepare_data(data_filepath, input_columns, interpolate=False, sequence_length=sequence_length)
    size_with = len(train) + len(test)

    cols_without = [c for c in input_columns if c != col]

    train, test = prepare_data(data_filepath, cols_without, interpolate=False, sequence_length=sequence_length)
    size_without = len(train) + len(test)

    print(f"Removing {col} increases the dataset by {size_without - size_with} ({size_with} -> {size_without})")

COLUMN_LABELS = {'CO2': {'title':'Carbon Dioxide Content', 'y_label':'Mol fraction (umolCO2 mol-1)'},
                'H20': {'title':'Water Content', 'y_label':'Mol fraction (mmolH2O mol-1)'},
                'CH4': {'title':'Methane Content', 'y_label':'Mol fraction (nmolCH4 mol-1)'},
                'FC': {'title':'Carbon Dioxide Flux', 'y_label':'Flux (umolCO2 m-2 s-1)'},
                'SC': {'title':'Carbon Dioxide Storage Flux', 'y_label':'Flux (umolCO2 m-2 s-1)'},
                'FCH4': {'title':'Methane Flux', 'y_label':'Flux (nmolCH4 m-2 s-1)'},
                'SCH4': {'title':'Methane Storage Flux', 'y_label':'Flux (nmolCH4 m-2 s-1)'},
                'G': {'title':'Soil Heat Flux', 'y_label':'Heat Flux (W m-2)'},
                'H': {'title':'Sensible Heat Flux', 'y_label':'Heat Flux (W m-2)'},
                'LE': {'title':'Latent Heat Flux', 'y_label':'Heat Flux (W m-2)'},
                'SH': {'title':'Air Heat Storage', 'y_label':'Heat Flux (W m-2)'},
                'SLE': {'title':'Latent Heat Storage Flux', 'y_label':'Heat Flux (W m-2)'},
                'WD': {'title':'Wind Direction', 'y_label':'Direction (deg)'},
                'WS': {'title':'Wind Speed', 'y_label':'Speed (m s-1)'},
                'USTAR': {'title':'Friction Velocity', 'y_label':'Speed (m s-1)'},
                'ZL': {'title':'Stability Param', 'y_label':''},
                'PA': {'title':'Atmospheric Pressure', 'y_label':'Pressure (kPa)'},
                'RH': {'title':'Relative Humidity', 'y_label':'Percent Humidity (%)'},
                'TA': {'title':'Air Temperature', 'y_label':'Temperature (deg C)'},
                'VPD': {'title':'Vapor Pressure Deficit', 'y_label':'Pressure (hPa)'},
                'SWC': {'title':'Soil Water Content', 'y_label':'Percent Water Content (%)'},
                'TS': {'title':'Soil Temperature', 'y_label':'Temperature (deg C)'},
                'WTD': {'title':'Water Table Depth', 'y_label':'Depth (m)'},
                'NETRAD': {'title':'Net Radiation', 'y_label':'Radiation (W m-2)'},
                'PPFD_IN': {'title':'Incoming Photon Flux Density', 'y_label':'Flux Density (umolP m-2 s-1)'},
                'PPFD_OUT': {'title':'Outgoing Photon Flux Density', 'y_label':'Flux Density (umolP m-2 s-1)'},
                'SW_IN': {'title':'Incoming Shortwave Radiation', 'y_label':'Flux (W m-2)'},
                'SW_OUT': {'title':'Outgoing Shortwave Radiation', 'y_label':'Flux (W m-2)'},
                'LW_IN': {'title':'Incoming Longwave Radiation', 'y_label':'Flux (W m-2)'},
                'LW_OUT': {'title':'Outgoing Longwave Radiation', 'y_label':r'Flux ($W m^{-2}$)'},
                'P': {'title':'Precipitation', 'y_label':r'Precipitation Height ($mm$)'},
                'NEE': {'title':'Net Ecosystem Exchange', 'y_label':r'Flux Density ($\mu$mol $m^{-2} s^{-1}$)'},
                'NEP': {'title':'Net Ecosystem Productivity', 'y_label':r'NEP ($\mu$mol $m^{-2} s^{-1}$)'},
                'RECO': {'title':'Ecosystem Respiration', 'y_label':r'Flux Density ($\mu$mol$CO_2 m^{-2} s^{-1}$)'},
                'GPP': {'title':'Gross Primary Productivity', 'y_label':r'Flux Density ($\mu$mol$CO_2 m^{-2} s^{-1}$)'},
                'D_SNOW': {'title': 'Snow Depth', 'y_label':r'Snow Depth ($in$)'},
                }

def find_prefix(var_name : str):
    for var_prefix in COLUMN_LABELS.keys():
        #if var_prefix in var_name:
        #    return var_prefix
        if re.match(rf'^{var_prefix}', var_name):
            return var_prefix
    return var_name