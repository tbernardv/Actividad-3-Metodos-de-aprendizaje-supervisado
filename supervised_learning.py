# Librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dataset de ejemplo
data = {
    'start_station': ['A', 'A', 'B', 'C', 'E'],
    'end_station': ['B', 'C', 'D', 'F', 'F'],
    'transport_mode': ['bus', 'metro', 'metro', 'bus', 'bicicleta'],
    'cost': [2.0, 1.5, 3.0, 5.0, 2.0],
    'travel_time': [15, 10, 20, 25, 30],
    'demand': [120, 100, 200, 150, 50]
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Variables predictoras y variable objetivo
X = df[['travel_time', 'demand']]
y = df['cost']

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Mostrar coeficientes del modelo
print(f"Coeficientes: {model.coef_}")
