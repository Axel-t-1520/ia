# Importar las librerías y montar Google Drive
from google.colab import drive
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import io

drive.mount('/content/drive')
ruta_archivo = '/content/drive/MyDrive/cleaned_star_data.csv'
datos = pd.read_csv(ruta_archivo)
#print("Vista previa del dataset:")
#print(datos)

# --------------------------------------------------------------------------------------------------
# Pre-procesamiento: Aseguramos que los campos numéricos sean correctos y eliminamos registros incompletos.
# --------------------------------------------------------------------------------------------------
# Convertir las columnas a formato numérico (las entradas problemáticas se convierten en NaN)
columnas_numericas = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']
for col in columnas_numericas:
    datos[col] = pd.to_numeric(datos[col], errors='coerce')

# Filtrar registros que tengan datos completos
datos_limpios = datos.dropna().copy()
print("\nDataset después de depurar valores faltantes:")
print(datos_limpios)

# --------------------------------------------------------------------------------------------------
# Convertir variables categóricas a valores numéricos.
# --------------------------------------------------------------------------------------------------


# Se codifican las columnas 'Star color' y 'Spectral Class'
for campo in ['Star color', 'Spectral Class']:
    codificador = LabelEncoder()
    datos_limpios[campo] = codificador.fit_transform(datos_limpios[campo])

print("\nEjemplo de datos codificados:")
print(datos_limpios[['Star color', 'Spectral Class']])

# --------------------------------------------------------------------------------------------------
# Definir variables: Separamos la variable objetivo (Star type) y las predictoras.
# --------------------------------------------------------------------------------------------------
datos_limpios['Star type'] = datos_limpios['Star type'].astype(int)
X = datos_limpios.drop(columns=['Star type'])
y = datos_limpios['Star type']

# --------------------------------------------------------------------------------------------------
# División en entrenamiento y prueba
# --------------------------------------------------------------------------------------------------


X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------------------------------------------------
# Entrenamiento: Modelo Random Forest para la clasificación.
# --------------------------------------------------------------------------------------------------


modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_entrena, y_entrena)


# --------------------------------------------------------------------------------------------------
# Evaluación: Generamos un reporte sobre el desempeño del modelo.
# --------------------------------------------------------------------------------------------------
from sklearn.metrics import classification_report

predicciones = modelo_rf.predict(X_prueba)
reporte = classification_report(y_prueba, predicciones)
print("\nReporte de desempeño:")
print(reporte)
