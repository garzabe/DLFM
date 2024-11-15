import pandas as pd
from enum import Enum
import numpy as np

Site = Enum('Site', ['Me2', 'Me6'])

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
    