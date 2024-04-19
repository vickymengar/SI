import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Cargar los datos
data = pd.read_csv("Delitos.csv")

# Definir las variables predictoras y la variable objetivo
X = data[["seccion", "mes"]]
y = data["tipodelito"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear un modelo Random Forest
model = RandomForestClassifier(n_estimators=100)

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir los delitos en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = model.score(X_test, y_test)
print("Precisión:", accuracy)

# Predecir los delitos para una sección y mes específicos
seccion = "Primera"
mes = "01"
prediccion = model.predict([[seccion, mes]])
print("Predicción:", prediccion[0])
