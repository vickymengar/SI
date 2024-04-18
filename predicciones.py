import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

# Cargar datos desde el archivo CSV
ruta_archivo_csv = "datos_prediccion_delitos.csv"
df = pd.read_csv('Delitos.csv')

# Preprocesamiento de datos (opcional)
# ... (agregar código de preprocesamiento si es necesario)

# Extraer las variables de interés
X = df[['No. de Delitos', 'Sección', 'Mes', 'Año']].values  # Ajustar nombres de columnas según tu archivo
y = df['Tipo de Delito'].values  # Ajustar nombre de columna según tu archivo

# Convertir variables categóricas (opcional)
if df['Sección'].dtype == 'object':
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    X[:, 1] = encoder.fit_transform(X[:, 1])

if df['Mes'].dtype == 'object':
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    X[:, 2] = encoder.fit_transform(X[:, 2])

# Construir el modelo de regresión logística
classifier = LogisticRegression(random_state=0)
classifier.fit(X, y)

# Definir una función para solicitar datos al usuario
def solicitar_datos_usuario():
    no_de_delitos = float(input("Ingrese el número de delitos en el área: "))
    seccion = input("Ingrese la sección del municipio: ")  # Ajustar tipo de entrada si es necesario
    mes = input("Ingrese el mes (en número): ")  # Ajustar tipo de entrada si es necesario
    año = int(input("Ingrese el año: "))

    # Preprocesar datos de entrada (si es necesario)
    if df['Sección'].dtype == 'object':
        seccion = encoder.transform([seccion])[0]

    if df['Mes'].dtype == 'object':
        mes = encoder.transform([mes])[0]

    # Convertir datos de entrada a formato adecuado
    datos_entrada = np.array([no_de_delitos, seccion, mes, año]).reshape(1, -1)
    return datos_entrada

# Realizar la predicción para el usuario
datos_entrada = solicitar_datos_usuario()
prediccion = classifier.predict(datos_entrada)
probabilidad = classifier.predict_proba(datos_entrada)

# Mostrar el resultado de la predicción
tipo_delito_predicho = df['Tipo de Delito'].unique()[prediccion[0]]
print(f"Se predice el siguiente tipo de delito: {tipo_delito_predicho}")
print(f"Probabilidad de que sea {tipo_delito_predicho}:", probabilidad[0, prediccion[0]])

# Definir una función para graficar las regiones de decisión y los datos de entrenamiento (opcional)
def plot_decision_regions(X, y, classifier):
    markers = ('s', 'x', 'o', '^', 'v')  # Definir marcadores para cada clase
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  # Definir colores para cada clase
    cmap = ListedColormap(colors[:len(np.unique(y))])  # Crear un mapa de colores

    # Definir los límites de los ejes x e y
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))

    # Crear una malla de puntos para la predicción
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # Colorear las regiones de decisión
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap='coolwarm')
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
