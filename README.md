Respuesta esperada del script
El script entrenará un modelo de regresión lineal para predecir el costo de un trayecto en un sistema de transporte basado en dos variables:
1. Tiempo de viaje.
2. Demanda (número de pasajeros por hora).

Después de entrenar el modelo con un conjunto de datos de ejemplo, el script realizará predicciones sobre un conjunto de prueba. Los resultados principales que obtendrás al ejecutar el script son:

Error Cuadrático Medio (Mean Squared Error, MSE): Un valor indica el error promedio que el modelo tiene al predecir los costos. Un MSE más bajo significa que el modelo es más preciso.
Coeficientes del modelo: Son los pesos que el modelo ha asignado a cada una de las variables (tiempo de viaje y demanda). Estos coeficientes indican cuánto cambia el costo del trayecto cuando una de las variables cambia.

1. Carga de bibliotecas y dataset:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

pandas: Se utiliza para manipular datos en forma de DataFrame.
train_test_split: Esta función divide el conjunto de datos en dos partes: una para entrenar el modelo (80%) y otra para probarlo (20%).
LinearRegression: Es el modelo de regresión lineal que utilizaremos para predecir el costo del transporte.
mean_squared_error: Evalúa el modelo midiendo la diferencia entre los valores reales de costo y las predicciones del modelo.

2. Dataset
   data = {
    'start_station': ['A', 'A', 'B', 'C', 'E'],
    'end_station': ['B', 'C', 'D', 'F', 'F'],
    'transport_mode': ['bus', 'metro', 'metro', 'bus', 'bicicleta'],
    'cost': [2.0, 1.5, 3.0, 5.0, 2.0],
    'travel_time': [15, 10, 20, 25, 30],
    'demand': [120, 100, 200, 150, 50]
}
df = pd.DataFrame(data)

Se crea un conjunto de datos ficticio donde cada fila representa un trayecto entre estaciones, el medio de transporte, el costo, el tiempo de viaje y la demanda de pasajeros.

3. Preparación del dataset:
   X = df[['travel_time', 'demand']]  # Variables predictoras
y = df['cost']  # Variable objetivo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X: Contiene las variables predictoras (tiempo de viaje y demanda).
y: Contiene la variable objetivo (costo).
train_test_split: Divide los datos en un conjunto de entrenamiento y uno de prueba. El 80% de los datos se utilizan para entrenar el modelo, y el 20% restante para evaluarlo.

4. Entrenamiento del modelo:
   model = LinearRegression()
model.fit(X_train, y_train)

Se crea una instancia del modelo de regresión lineal y se entrena (fit) con los datos de entrenamiento.

5. Predicción y evaluación:
   y_pred = model.predict(X_test)  # Realiza predicciones sobre el conjunto de prueba
mse = mean_squared_error(y_test, y_pred)  # Calcula el error cuadrático medio (MSE)
print(f"Mean Squared Error: {mse}")
print(f"Coeficientes: {model.coef_}")  # Muestra los coeficientes del modelo

   model.predict: Realiza predicciones basadas en los datos de prueba.
mean_squared_error: Compara las predicciones del modelo con los valores reales y calcula el error cuadrático medio.
model.coef_: Muestra los coeficientes del modelo, que indican el impacto de cada variable en la predicción del costo.
