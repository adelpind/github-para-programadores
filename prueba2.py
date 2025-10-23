from sklearn import preprocessing
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv", nrows = 100000)

df = df[["ArrDelay", "DepDelay", "Distance", "AirTime"]].dropna()

X_scaled= preprocessing.scale(df)

print(X_scaled)

print(X_scaled.mean(axis = 0))
print(X_scaled.std(axis = 0))

print(df.iloc[2])

print(X_scaled[2])

#min_max_scaler = preprocessing.MinMaxScaler([0,10])
#X_train_minmax = min_max_scaler.fit_transform(df)

#print(X_train_minmax)

df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv", nrows = 100)
print(pd.get_dummies(df["Origin"]))

df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv", nrows = 1e5)
newdf = df[["AirTime", "DepDelay"]].dropna()

newdf = df[["AirTime", "Distance", "TaxiOut", "ArrDelay", "DepDelay"]].dropna()

kmeans = KMeans(n_clusters=4, random_state=0).fit(newdf)
print(kmeans.labels_)

np.unique(kmeans.labels_,return_counts=True)

plt.scatter(x = newdf["AirTime"], y = newdf["DepDelay"],c= kmeans.labels_)
plt.show()

