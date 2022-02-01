import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#Stanfod Open Policing Project dataset
ri = pd.read_csv('police.csv')
ri.head(3)

#Locating missing values
ri.isnull().sum()
ri.shape

#Drop NAs
ri.dropna(subset=["county_name", "state", 'driver_gender'], axis="columns", inplace=True)

#Check types
ri.dtypes
ri.is_arrested.head()
ri['is_arrested'] = ri.is_arrested.astype(bool)
ri["is_arrested"].dtype

#Fix index by making it a time-series
combined = ri.stop_date.str.cat(ri.stop_time, sep=" " )
ri['stop_datetime'] = pd.to_datetime(combined)
ri.dtypes
ri.set_index("stop_datetime", inplace=True)
ri.index
ri.columns

# Count the unique values in 'violation'
print(ri["violation"].value_counts())

print(ri["violation"].value_counts(normalize = True))

# Create a DataFrame of female drivers
female = ri[ri.driver_gender == 'F']
# Create a DataFrame of male drivers
male = ri[ri.driver_gender == 'M']
print(female.violation.value_counts(normalize=True))
print(male.violation.value_counts(normalize=True))
female_and_speeding = ri[(ri.driver_gender == "F") & (ri.violation == "Speeding")]
male_and_speeding = ri[(ri.driver_gender == "M") & (ri.violation == "Speeding")]
print(female_and_speeding.stop_outcome.value_counts(normalize=True))

print(male_and_speeding.stop_outcome.value_counts(normalize=True))

print(ri.search_conducted.dtype)

print(ri.search_conducted.value_counts(normalize=True))

print(ri.search_conducted.mean())

print(ri.groupby("driver_gender").search_conducted.mean())
print(ri.groupby(by=["driver_gender", "violation"]).search_conducted.mean())
print(ri.search_type.value_counts())
ri['frisk'] = ri.search_type.str.contains('Protective Frisk', na=False)
print(ri["frisk"].dtype)
print(ri["frisk"].sum())
# DataFrame of stops in which a search was conducted
searched = ri[ri.search_conducted == True]

#Overall frisk rate by taking the mean of 'frisk'
print(searched.frisk.mean())
print(searched.groupby("driver_gender").frisk.mean())
print(ri.is_arrested.mean())
print(ri.groupby(ri.index.hour).is_arrested.mean())
hourly_arrest_rate = ri.groupby(ri.index.hour).is_arrested.mean()
plt.plot(hourly_arrest_rate)
plt.xlabel("Hour")
plt.ylabel("Arrest Rate")
plt.title('Arrest Rate by Time of Day')

plt.show()

print(ri.drugs_related_stop.resample("A").mean())
annual_drug_rate = ri.drugs_related_stop.resample("A").mean()
plt.plot(annual_drug_rate)
plt.show()

#Annual search rate
annual_search_rate = ri.search_conducted.resample("A").mean()
annual = pd.concat([annual_drug_rate, annual_search_rate], axis=1)
annual.plot(subplots=True)
plt.show()

print(pd.crosstab(ri["district"], ri["violation"]))
all_zones = pd.crosstab(ri["district"], ri["violation"])
print(all_zones.loc["Zone K1":"Zone K3"])
k_zones = all_zones.loc["Zone K1":"Zone K3"]
k_zones.plot( kind="bar")
plt.show()
k_zones.plot(kind="bar", stacked=True)
plt.show()

print(ri.stop_duration.unique())
mapping = {'0-15 Min': 8,'16-30 Min':23,'30+ Min':45}
ri['stop_minutes'] = ri.stop_duration.map(mapping)
print(ri.stop_minutes.unique())
print(ri.groupby("violation_raw").stop_minutes.mean())

stop_length = ri.groupby("violation_raw").stop_minutes.mean()
stop_length.sort_values().plot(kind="barh")
plt.show()

#####
weather = pd.read_csv("weather.csv")
weather.head(3)

# 'weather.csv' into a DataFrame named 'weather'
weather = pd.read_csv("weather.csv")

print(weather[["TMIN", "TAVG", "TMAX"]].describe())
weather.plot(kind='box')

plt.show()
weather["TDIFF"] = weather["TMAX"] - weather["TMIN"]

print(weather["TDIFF"].describe())
weather.TDIFF.hist(bins=20)
plt.show()
WT = weather.loc[:,"WT01":"WT22"]
weather['bad_conditions'] = WT.sum(axis = "columns")
weather['bad_conditions'] = weather.bad_conditions.fillna(0).astype('int')
weather["bad_conditions"].plot(kind="hist")
plt.show()
print(weather.bad_conditions.value_counts().sort_index())
mapping = {0:'good', 1:'bad', 2:'bad', 3:"bad", 4:"bad", 5:"worse", 6:"worse", 7:"worse", 8:"worse", 9:"worse"}
weather['rating'] = weather.bad_conditions.map(mapping)
print(weather["rating"].value_counts())

cats = ["good", "bad", "worse"]

weather['rating'] = weather.rating.astype("category", ordered=True, categories = cats)

print(weather["rating"].head())
ri.reset_index(inplace=True)
print(ri.head())
weather_rating = weather[["DATE", "rating"]]

print(weather_rating.head())

print(ri.shape)
ri_weather = pd.merge(left=ri, right=weather_rating, left_on='stop_date', right_on='DATE', how='left')

print(ri_weather.shape)
ri_weather.set_index('stop_datetime', inplace=True)
overall arrest rate
print(ri_weather.is_arrested.mean())

print(ri_weather.groupby("rating").is_arrested.mean())

print(ri_weather.groupby(by=["violation", "rating"]).is_arrested.mean())

arrest_rate = ri_weather.groupby(['violation', 'rating']).is_arrested.mean()

print(arrest_rate)

print(arrest_rate.loc["Moving violation", "bad"])

print(arrest_rate.loc["Speeding"])

print(arrest_rate.unstack())
print(ri_weather.pivot_table(index='DATE', columns='violation', values='is_arrested'))

