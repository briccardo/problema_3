import pandas
from pandas import DataFrame
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

datos = pandas.read_csv('data/problema_3.csv')
df = DataFrame(datos)

x = df[['manual']]  # variable(s) independiente(s)
y = df['automática']  # variable dependiente
nitrato = 100

lineal = linear_model.LinearRegression()
lineal.fit(x, y)

y_adjusted = lineal.predict(x)
y_predicted = lineal.predict([[nitrato]])

rmse = np.sqrt(mean_squared_error(y, y_adjusted))
r2 = r2_score(y, y_adjusted)

print(f'Resultado regresión: {round(y_predicted[0], 2)}')
print(f'R2: {round(r2, 2)}')
print(f'Error medio: ±{round(rmse, 2)}')

