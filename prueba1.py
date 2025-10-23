import pandas as pd
import numpy as np
from joblib import Parallel, delayed


df = pd.read_csv("C:/Users/angel.delpinodiaz/downloads/base_datos_2008.csv", nrows = 100000)

df_sub = df[["CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay"]]

print(df.head())

def retraso_maximo(fila):
    if not np.isnan(fila).any():
        names = ["CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay"]
        return names[fila.index(max(fila))]
    else:
        return "None"

results = []
for fila in df_sub.values.tolist():
    results.append(retraso_maximo(fila))

print(results)

result = Parallel(n_jobs=2, backend = "multiprocessing")(map(delayed(retraso_maximo), df_sub.values.tolist()))

