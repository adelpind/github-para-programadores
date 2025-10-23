from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv")

df = df.dropna(subset=["ArrDelay"])

df.sample(frac=1).head(100000)

Y = df["ArrDelay"]
X = df[["DepDelay"]]

print(df.columns)

regr = linear_model.LinearRegression()
regr.fit(X,Y)

print("Coeficientes: ",regr.coef_)
Y_pred = regr.predict(X)
print("R Cuadrado: ",r2_score(Y, Y_pred))

plt.scatter(X[1:10000],Y[1:10000], color = 'black')
plt.plot(X[1:10000],Y_pred[1:10000], color = 'blue')
plt.show()

X = df[["AirTime","Distance","TaxiIn","TaxiOut"]]
df["Month"] = df["Month"].apply(str)
df["DayofMonth"] = df["DayofMonth"].apply(str)
df["DayOfWeek"] = df["DayOfWeek"].apply(str)

#dummies = pd.get_dummies(data = df[["Month","DayofMonth","DayOfWeek","Origin","Dest"]])
#X = dummies.add(X, fill_value=0)
#print(X.columns)

#regr = linear_model.LinearRegression()
#regr.fit(X,Y)

#print("Coeficientes: ",regr.coef_)
#Y_pred = regr.predict(X)
#print("R Cuadrado: ",r2_score(Y, Y_pred))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv")

df = df.dropna(subset=["ArrDelay"])

df.sample(frac=1).head(100000)

Y = df["ArrDelay"] < 30 # Vuelos retrasados más de 30 minutos
X = df[["DepDelay"]]

logreg = LogisticRegression()
logreg.fit(X, Y)
Y_pred = logreg.predict(X)

np.round(logreg.predict_proba(X), 3)

np.mean(Y_pred == Y)
np.mean(Y)

confusion_matrix = confusion_matrix(Y, Y_pred)
print(confusion_matrix)

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv")

df.sample(frac=1).head(500000)
df = df.dropna(subset=["ArrDelay"])
Y = df["ArrDelay"] > 0

df["Month"] = df["Month"].apply(str)
df["DayofMonth"] = df["DayofMonth"].apply(str)
df["DayOfWeek"] = df["DayOfWeek"].apply(str)
df["TailNum"] = df["TailNum"].apply(str)

X = pd.get_dummies(data = df[["Month","DayofMonth","TailNum", "DayOfWeek","Origin","Dest","UniqueCarrier"]])
print(X.head())

clf = BernoulliNB()
#clf = MultinomialNB()
clf.fit(X, Y)
Y_pred = clf.predict(X)

print(np.mean(Y == Y_pred))
print(1 - np.mean(Y))

