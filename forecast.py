import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar datos desde un archivo Excel
data = pd.read_excel('C:/Users/Dell G/Desktop/DB_CRM.xlsx', engine='openpyxl')

# Aplicar una condición a una columna (por ejemplo, seleccionar filas con 'ColumnaCondicion' > 10)
condicion = data['Estado'] == "Reposo"
datos_filtrados = data[condicion]

# Extraer las columnas de características (X) y etiquetas (y)
X = datos_filtrados['Valor'].values.reshape(-1, 1)
y = datos_filtrados['Probabilidad'].values

# Crear un modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos filtrados
model.fit(X, y)

# Realizar predicciones
y_pred = model.predict(X)

# Visualizar los resultados
plt.scatter(X, y, label='Datos', color='blue')
plt.plot(X, y_pred, label='Predicciones', color='red')
plt.xlabel('Pronostico de Ventas Q')
plt.ylabel('Probabilidad de venta % ')
plt.legend()
plt.title('Regresión Lineal con Condiciones desde Excel')
plt.show()

# Coeficiente de regresión y término independiente
print(f'Coeficiente de regresión (pendiente): {model.coef_[0]}')
print(f'Término independiente (intercept): {model.intercept_}')
