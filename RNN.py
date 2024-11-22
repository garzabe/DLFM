from enum import Enum

global input_column_set
global site

Site = Enum('Site', ['Me2', 'Me6'])

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

LAYER_FEATURES = 8