from data_handler import Site, get_data
import pandas as pd
import matplotlib.pyplot as plt


me2 = get_data(Site.Me6)

# remove all data between timestamp "YYYMMDD0500" and "YYYMMDD1700" (roughly daytime hours)
me2['DATETIME'] = pd.to_datetime(me2['TIMESTAMP_START'], format="%Y%m%d%H%M")
me2['DAY'] = me2['DATETIME'].apply(lambda dt: dt.date())
me2['TIME'] = me2['DATETIME'].apply(lambda dt: dt.time())

#me2 = me2[(me2['HOUR'] < 5) | (me2['HOUR'] > 22)]
#print(me2.head())

# look for any photon data over ~5
#for ppfd_threshold in range(1, 50, 4):
#    nighttime_light_data = me2[me2['PPFD_IN'] > ppfd_threshold]
#    print(f"Threshold of {ppfd_threshold}: {len(nighttime_light_data)} outliers")



# when we use ppfd to determine daytime hours, what are the first data points for each day?
photon_data = me2[me2['PPFD_IN'] > 5][['DAY','TIME','PPFD_IN']]

photon_data_grouped = photon_data.groupby(by=['DAY'])

photon_data_earliest = photon_data_grouped.min().reset_index()
photon_data_latest = photon_data_grouped.max().reset_index()

# observe the earliest high-light times
days = photon_data_earliest['DAY']
earliest_light_time = photon_data_earliest['TIME'].apply(lambda dtt: dtt.hour*60 + dtt.minute)
latest_light_time = photon_data_latest['TIME'].apply(lambda dtt: dtt.hour*60 + dtt.minute)
plt.scatter(days, earliest_light_time, label='Earliest PPFD_IN>5 reading', s=10)
plt.scatter(days, latest_light_time, label='Latest PPFD_IN>5 reading', s=10)
plt.xlabel("Date")
plt.ylabel("Minute of Day")
plt.yticks(range(0, 60*24, 3*60), [f"{minute//60}:00" for minute in range(0, 60*24, 3*60)])
plt.ylim((0, 60*24))
plt.legend()
plt.title("Earliest and Latest PPFD_IN > 5 readings for each day - Me-6")
plt.show()