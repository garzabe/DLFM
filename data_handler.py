import pandas as pd
from enum import Enum
import numpy as np
from torch.utils.data import Dataset

Site = Enum('Site', ['Me2', 'Me6'])

class AmeriFLUXDataset(Dataset):
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
    