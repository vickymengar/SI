import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Cargar datos desde el archivo CSV
ruta_archivo_csv = "/content/delitooos.csv"  # Ajusta el nombre del archivo CSV si es diferente
# Usar header=0 para indicar que la primera fila es el encabezado
df = pd.read_csv(ruta_archivo_csv, header=0)

# Verificar el contenido del DataFrame
print(df.head())
print(df.columns)

# Preprocesamiento de datos (opcional)
# ... (agregar código de preprocesamiento si es necesario)

# Extraer las variables de interés
X = df[['carpetas', 'delito', 'fecha']].values  # Ajustar nombres de columnas según tu archivo
y = df['id_delito'].values  # Ajustar nombre de columna según tu archivo

# Convertir variables categóricas
encoder = LabelEncoder()
X[:, 1] = encoder.fit_transform(X[:, 1])  # Codificar la variable 'delito'
X[:, 2] = encoder.fit_transform(X[:, 2])  # Codificar la variable 'fecha'

# Construir el modelo de regresión logística
classifier = LogisticRegression(random_state=0)
classifier.fit(X, y)

# Verificar categorías únicas en las variables categóricas de los datos de entrenamiento
categorias_delito_entrenamiento = df['delito'].unique()
categorias_fecha_entrenamiento = df['fecha'].unique()

# Definir una función para solicitar datos al usuario
def solicitar_datos_usuario():
    # Solicitar datos al usuario
    no_de_delitos = float(input("Ingrese el número de delitos en el área: "))
    delito = input("Ingrese el tipo de delito: ")
    fecha = input("Ingrese la fecha (en formato YYYY-MM): ")

    # Verificar si las categorías proporcionadas por el usuario están presentes en los datos de entrenamiento
    if delito not in categorias_delito_entrenamiento:
        print("El tipo de delito ingresado no es válido.")
        return None
    if fecha not in categorias_fecha_entrenamiento:
        print("La fecha ingresada no es válida.")
        return None

    # Preprocesar datos de entrada (si es necesario)
    delito = encoder.transform([delito])[0]
    fecha = encoder.transform([fecha])[0]

    # Convertir datos de entrada a formato adecuado
    datos_entrada = np.array([no_de_delitos, delito, fecha]).reshape(1, -1)
    return datos_entrada

# Realizar la predicción para el usuario
datos_entrada = solicitar_datos_usuario()

if datos_entrada is not None:
    prediccion = classifier.predict(datos_entrada)
    probabilidad = classifier.predict_proba(datos_entrada)

    # Mostrar el resultado de la predicción
    tipo_delito_predicho = df['id_delito'].unique()[prediccion[0]]
    print(f"Se predice el siguiente tipo de delito: {tipo_delito_predicho}")
    print(f"Probabilidad de que sea {tipo_delito_predicho}:", probabilidad[0, prediccion[0]])
