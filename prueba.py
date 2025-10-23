import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv", nrows = 100000)

print(df.head(10))

print(df.tail(10))

print(df.columns)

print(df.dtypes)

print(df.DepTime)

print(df.values)

print(df["ArrDelay"].head())

print(df[df["ArrDelay"] < 60].head())
print(df[df["ArrDelay"] > 60].head())

print(df[df.Origin.isin(["HOU","ATL"])].head())

print(len(df[pd.isna(df["ArrDelay"])]))

df["HoursDelay"] = round(df["ArrDelay"] / 60)

print(df["HoursDelay"].head())

del(df["HoursDelay"])

#df = df.drop(["Dicerted", "Cancelled", "Year"], axis = 1)

dfATL = df[df.Origin == "ATL"]
dfHOU = df[df.Origin == "HOU"]

newdf = dfATL._append(dfHOU)

print(newdf.Origin)

print(df.groupby(by="DayOfWeek")["ArrDelay"].max())
print(df.groupby(by="DayOfWeek")["ArrDelay"].mean())
print(df.groupby(by="DayOfWeek")["ArrDelay"].min())
print(df.groupby(by="DayOfWeek")["ArrDelay"].describe())

dfATLHOU = df[df.Origin.isin(["HOU","ATL"])]

print(dfATLHOU.groupby(by=["DayOfWeek","Origin"])["ArrDelay"].mean())

dfclean = df.drop_duplicates()

print(len(df) == len(newdf), len(df), len(newdf))
print(len(df) == len(dfclean))

print(df.dropna(thresh = len(df)-2))

valoraciones = np.array([[8,7,6,5],[1,6,8,9], [6,5,2,1]])
print(valoraciones[0][1])

print(np.mean(valoraciones, axis = 1))

print(np.random.rand(4,4))

df.dropna(inplace=True, subset=["ArrDelay","DepDelay"])

print(np.corrcoef(df["ArrDelay"],df["DepDelay"]))

df.drop(inplace=True, columns=["Year","Cancelled","Diverted"])

print(df.corr)

df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv")

np.random.seed(0)
df = df[df["Origin"].isin(["HOU","ATL","IND"])]
df.sample(frac=1)
df = df[0:10000]

df["BigDelay"] = df["ArrDelay"] > 30
observados = pd.crosstab(index= df["BigDelay"], columns=df["Origin"], margins=True)
print(observados)

from scipy.stats import chi2_contingency

test = chi2_contingency(observados)

test

esperados = pd.DataFrame(test[3])

esperados

esperados_rel = round(esperados.apply(lambda r: r/len(df)*100, axis=1), 2)
observados_rel = round(observados.apply(lambda r: r/len(df)*100, axis=1), 2)

print(esperados_rel)

print(observados_rel)

print(test[1])

df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv", nrows=100000)

x = df["ArrDelay"].dropna()
Q1 = np.percentile(x,25)
Q3 = np.percentile(x,75)
rangointer = Q3 - Q1

umbralsuperior = Q3 + 1.5*rangointer
umbralinferior = Q1 - 1.5*rangointer

print("umbralsuperior:", umbralsuperior)
print("umbralinferior:", umbralinferior)

np.mean(x > umbralsuperior)
np.mean(x < umbralinferior)

from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination= .01)

for i in range(3):
    pass
print(i)

data = [(1,"Angel","del Pino",59,1,"boligrafo",0.5,0.3,0.2),
        (1,"Angel","del Pino",59,2,"cuaderno",2.6,1.2,1.4),
        (1,"Angel","del Pino",59,3,"Ratón",15.5,4.3,11.2),
        (2,"Judith","Gutierrez",49,1,"boligrafo",0.5,0.3,0.2),
        (2,"Judith","Gutierrez",49,2,"cuaderno",2.6,1.2,1.4),
        (2,"Judith","Gutierrez",49,3,"Ratón",15.5,4.3,11.2),
        (3,"Lucia","del Pino",20,1,"boligrafo",0.5,0.3,0.2),
        (3,"Lucia","del Pino",20,2,"cuaderno",2.6,1.2,1.4)]

labels = ["Comprador_id","Nombre","Apellido","Edad","Product_id","Nombre_producto","Precio","Coste","Margen"]

df = pd.DataFrame.from_records(data, columns=labels)

print(df)

df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv", nrows = 100000)

data = np.unique(df.Cancelled, return_counts=True)

df = df.sample(frac=1).head(100)

print(df.head())

plt.scatter(x = df.DayofMonth, y = df.ArrDelay, s = df.Distance, alpha = .3, c = df.DayOfWeek.isin([6,7]))
plt.title("Retrasos en EEUU")
#plt.xlabel(s= "Dia del Mes", loc = 'center')
#plt.ylabel(s= "Retraso al llegar", loc = 'center')
plt.show()


df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv")
df.dropna(inplace= True, subset= ["ArrDelay","DepDelay","Distance"])

sns.kdeplot(df["ArrDelay"])
sns.kdeplot(df["DepDelay"])
plt.xlim([-300,300])
plt.show()

df2 = df[df["Origin"].isin(["ATL","HOU","IND"])].sample(frac = 1).head(500)
sns.boxplot(x = "DepDelay", y="Origin", data=df2)
plt.xlim([-20,180])
plt.show()

df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv")
df.dropna(inplace= True, subset= ["ArrDelay","DepDelay","Distance","AirTime"])
sns.set_theme(rc = {'figure.figsize':(15,10)})
df2 = df[df["Origin"].isin(["ATL","HOU","IND"])].sample(frac = 1).head(1000)
sns.jointplot(x = df2["DepDelay"],y = df2["ArrDelay"])
plt.show()

df3 = df2[np.abs(df2["DepDelay"]<40)]
df3 = df3[np.abs(df2["ArrDelay"]<40)]
sns.jointplot(x = df3["DepDelay"],y = df3["ArrDelay"], kind = "hex")
plt.show()

sns.jointplot(x = df3["DepDelay"],y = df3["ArrDelay"], kind = "kde")
plt.show()

gb_df = pd.DataFrame(df2.groupby(["Origin","Month"], as_index=False)["DepDelay"].mean())
print(gb_df.head())
data1 = gb_df.pivot(columns=["Month","Origin","DepDelay"])
print(data1)

#sns.set_theme(rc = {'figure.figsize':(15,8)})
#sns.heatmap(data = data1, annot=True)
#plt.show()
sns.heatmap()
