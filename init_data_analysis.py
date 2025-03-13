import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


column_labels = {'CO2': {'title':'Carbon Dioxide Content', 'y_label':'Mol fraction (umolCO2 mol-1)'},
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
                'LW_OUT': {'title':'Outgoing Longwave Radiation', 'y_label':'Flux (W m-2)'},
                'P': {'title':'Precipitation', 'y_label':'Precipitation Height (mm)'},
                'NEE': {'title':'Net Ecosystem Exchange', 'y_label':r'Flux Density ($\mu$mol $m^{-2} s^{-1}$)'},
                'NEP': {'title':'Net Ecosystem Productivity', 'y_label':'NEP ($\mu$mol $m^{-2} s^{-1}$)'},
                'RECO': {'title':'Ecosystem Respiration', 'y_label':'Flux Density (umolCO2 m-2 s-1)'},
                'GPP': {'title':'Gross Primary Productivity', 'y_label':'Flux Density (umolCO2 m-2 s-1)'},
                'D_SNOW': {'title': 'Snow Depth', 'y_label':'Snow Depth (in)'},
                }

def main():

    nan_density_threshold = 0.3
    me2_data = pd.read_csv('AmeriFLUX Data/AMF_US-Me2_BASE-BADM_19-5/AMF_US-Me2_BASE_HH_19-5.csv', header=2)
    me6_data = pd.read_csv('AmeriFLUX Data/AMF_US-Me6_BASE-BADM_16-5/AMF_US-Me6_BASE_HH_16-5.csv', header=2)

    me2_data.replace(-9999, np.nan, inplace=True)
    me6_data.replace(-9999, np.nan, inplace=True)


    #print(me6_data[pd.notna(me6_data['PPFD_IN'])][['TIMESTAMP_START', 'PPFD_IN']])
    # what portion of each column is NA?


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
    #plot_daily_avg(me2_data, 'SWC_1_7_1', 'Me-2', daytime_only=True)
    #plot_daily_avg(me2_data, 'SWC_3_7_1', 'Me-2', daytime_only=True)

    #plot_daily_avg(me2_data, 'PPFD_IN', 'Me-2 Daytime', daytime_only=True)
    #plot_daily_avg(me2_data, 'GPP', 'Me-2')
    #plot_daily_avg(me2_data, 'RECO', 'Me-2')
    #for column in me6_input_column_set:
    #    plot_annual_avg(me6_data, column, 'Me-6')
    # More in depth, over time what are the NaN densities? (per month)
    #plot_annual_avg(me2_data, 'SWC_\d_1', 'Me-2')
    
    #plot_annual_avg(me2_data, 'SWC_\d_7', 'Me-2')

    plot_annual_avg(me2_data, 'P', '')
    

    #lot_annual_avg(me2_data, 'RH', 'Me-2')
    #plot_annual_avg(me2_data, 'SWC_\d_2','Me-2')
    #plot_annual_avg(me2_data, 'TS', 'Me-2')
    #plot_annual_avg(me2_data, 'P', 'Me-2', precision='MONTH')
    #plot_annual_avg(me6_data, 'SWC_1', 'Me-6')
    #plot_annual_avg(me6_data, 'TS', 'Me-6')
    #plot_annual_avg(me6_data, 'P', 'Me-6')
    #plot_annual_avg(me2_data, 'G', precision='MONTH')
    exit()
    month_datapoints = 1440
    monthly_nan_densities : dict[str, list[float]] = {}
    for m in range(0, len(me2_data)//1440):
        month_data = me2_data.iloc[m:min(m+1440, len(me2_data))]
        month_nan_densities = get_nan_densities(month_data)
        for c, d in month_nan_densities.items():
            monthly_nan_densities.setdefault(c, []).append(d)
    
    
    current_prefix = ''
    for c, density_data in []: #monthly_nan_densities.items():
        # lets plot cols with the same prefix on the same graph
        c_prefix = c.split('_')[0]
        if c_prefix != current_prefix:
            # show the previous graph
            if current_prefix != '':
                plt.legend()
                plt.show()
            
            # set up the next graph
            plt.clf()
            current_prefix = c_prefix
            plt.title(f"{current_prefix} not-NaN Density")
            plt.ylim(-0.1, 1.1)
            plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            plt.xlabel('T - start [months]')
        
        plt.plot(range(len(density_data)), [1-d for d in density_data], label=c)
    
        

    """
    # what does the data look like when we remove the high-NaN columns? threshold of 90%
    me2_data.drop(columns=me2_drop_columns, inplace=True)
    me6_data.drop(columns=me6_drop_columns, inplace=True)

    # what do the densities look like after these columns are dropped
    print('\nMe-2 after columns dropped')
    print_nan_densities(me2_data)
    print('\nMe-6 after columns dropped')
    print_nan_densities(me6_data)

    me2_data.dropna(inplace=True)
    me6_data.dropna(inplace=True)

    print(me2_data.head())
    print(len(me2_data))
    print(me6_data.head())
    print(len(me6_data))
    plot_continuous_sequences(me2_data)
    plot_continuous_sequences(me6_data)"""

def plot_continuous_sequences(df: pd.DataFrame):
    # find the time intervals with no gaps
    df_nogaps = get_continuous_sequences(df.index.to_list())
    df_nogap_lengths = [end-begin for begin, end in df_nogaps]
    df_nogap_intervals = [0.5*length/24 for length in df_nogap_lengths]

    print(f"longest no-gap interval is {max(df_nogap_intervals)} days")

    fig, axs = plt.subplots()
    fig.suptitle('Density of continuous sequence lengths')
    axs.hist(df_nogap_intervals, bins=100)
    axs.set_title('Me-2')
    axs.set_yscale('log')
    axs.set_xlabel('Interval [days]')
    plt.show()

def plot_daily_avg(df : pd.DataFrame, col_category: str, title_prefix : str, daytime_only=False):
    # get the matching columns
    cols = df.columns[df.columns.str.contains(f'^{col_category}_') | df.columns.str.contains(f'^{col_category}$')].to_list()
    col_title = column_labels[col_category]['title'] if col_category in column_labels.keys() else col_category
    col_ylabel = column_labels[col_category]['y_label'] if col_category in column_labels.keys() else ''
    ### reduce to daily averages
    df['DATETIME'] = pd.to_datetime(df['TIMESTAMP_START'], format="%Y%m%d%H%M")
    df['DAY'] = df['DATETIME'].apply(lambda dt: f"{dt.year:04}{dt.month:02}{dt.day:02}")
    print(df.head())

    # remove nighttime rows
    if daytime_only:
        df = df[df['PPFD_IN'] > 1.0]

    # simply strip the year, hours and minutes from each timestamp
    df_col = df[["DAY", *cols]]
    # get the daily mean for each included column
    means = df_col.groupby("DAY").aggregate(['mean','count']).reset_index()
    # sort 
    means.sort_values("DAY", ascending=True,axis=0)
    dates = pd.to_datetime(means['DAY'], format="%Y%m%d").to_list()
    x = [d.date() for d in dates]
    for i in range(10):
        print(x[i], means.iloc[i])
    fig, ax1 = plt.subplots()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    #X = means["DAY"].to_list()
    for col in cols:
        y = means[col]['mean'].to_list()
        ax1.plot(x, y, label=col)
    plt.xticks([x[i] for i in range(0, len(x), len(x)//10)])
    fig.autofmt_xdate()
    ax1.set_ylabel(col_ylabel)
    plt.suptitle(f'{title_prefix} - {col_title} - Daily Averages')
    ax1.legend()
    plt.show()

def plot_annual_avg(df: pd.DataFrame, col_category: str, title_prefix : str, precision : str = 'DAY'):
    # get the matching columns
    # filter out nighttime data
    df = df[df['PPFD_IN'] > 4.0]
    df = df[df['USTAR'] > 0.2]
    df.dropna(subset='NEE_PI_F', inplace=True)
    _col = col_category
    if col_category=='NEP':
        _col = 'NEE'
    cols = df.columns[df.columns.str.contains(f'^{_col}_') | df.columns.str.contains(f'^{_col}$')].to_list()
    col_title = column_labels[col_category]['title'] if col_category in column_labels.keys() else col_category
    col_ylabel = column_labels[col_category]['y_label'] if col_category in column_labels.keys() else ''
    ### reduce to daily averages
    df['DATETIME'] = pd.to_datetime(df['TIMESTAMP_START'], format="%Y%m%d%H%M")
    if precision=='MONTH':
        df['MONTH'] = df['DATETIME'].apply(lambda dt: f"{dt.month:02}")
    else:
        df['DAY'] = df['DATETIME'].apply(lambda dt: f"{dt.month:02}{dt.day:02}")
    print(df.head())
   
    # simply strip the year, hours and minutes from each timestamp
    df_col = df[[precision, *cols, 'NEE_PI_F']]
    # get the daily mean for each included column
    means = df_col.groupby(precision).aggregate(['mean']).reset_index()
    variances = df_col.groupby(precision).aggregate(['var']).reset_index()
    quart25 = df_col.groupby(precision).quantile(0.25).reset_index()
    quart75 = df_col.groupby(precision).quantile(0.75).reset_index()

    # sort 
    means.sort_values(precision, ascending=True,axis=0)
    fig, ax1 = plt.subplots()
    
    
    X = means[precision].to_list()
    
    ax2 = ax1.twinx()
    y = means['NEE_PI_F']['mean'].to_list()
    q25 = quart25['NEE_PI_F'].to_list()
    q75 = quart75['NEE_PI_F'].to_list()
    y = np.array([-_y for _y in y])
    q25 = np.array([-_q for _q in q25])
    q75 = np.array([-_q for _q in q75])
    ax2.plot(X,y,label='NEP', alpha=0.5, color='tab:blue')
    ax2.fill_between(X, q25, q75, alpha=0.1, color='tab:blue')
    ax2.set_ylabel(column_labels['NEP']['y_label'])
    
    for col in cols:
        y = means[col]['mean'].to_list()
        q25 = quart25[col].to_list()
        q75 = quart75[col].to_list()
        if col_category == 'NEP':
            y = np.array([-_y for _y in y])
            q25 = np.array([-_q for _q in q25])
            q75 = np.array([-_q for _q in q75])
        ax1.plot(X, y, label='Precipitation', color='tab:red')
        ax1.fill_between(X, q25, q75, alpha=0.2, color='tab:red')
    if precision=='MONTH':
        ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    else:
        ax1.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax1.set_ylim((0, 0.8))
    ax1.set_ylabel(column_labels['P']['y_label'])
    ax1.grid(color='0.95')
    
    plt.title(f'Precipitation Annual Average at US-Me2')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()


def get_nan_densities(df: pd.DataFrame) -> dict[str, float]:
    nan_densities = {}
    for column in df.columns:
        num_na = len(df[df[column].isna()])
        frac_na = num_na/len(df[column])
        nan_densities[column] = frac_na
    return nan_densities

def print_nan_densities(df: pd.DataFrame) -> None:
    densities = get_nan_densities(df)
    for column, density in densities.items():
        print(f"{column} is {density:.3%} NaN")

def get_continuous_sequences(arr: list[int]) -> list[tuple[int, int]]:
    continuous_sequences = []
    i = 0
    while i < len(arr):
        #print(f"{i}: {arr[i]}")
        length = 1
        begin = arr[i]
        for j in range(i+1, len(arr)):
            #print(f"\t{j}: {arr[j]}")
            if arr[j] != begin + (j-i) or j == len(arr)-1:
                # end of the continuous subarray
                end = arr[j-1]
                continuous_sequences.append((begin, end))
                # save some time by skipping all of the elements within the sequence we just saw
                i = j-1
                break
        i+=1

    return continuous_sequences
            

if __name__=='__main__':
    main()
