import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Cargar datos
datos = load_wine()
X = datos.data[:, :4] # Usamos solo las 4 primeras características
y = datos.target

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Pipeline profesional pero simple
modelo_pipeline = Pipeline([
    ('escalador', StandardScaler()),
    ('clasificador', LogisticRegression(max_iter=1000))
])

# 4. Entrenar y guardar
modelo_pipeline.fit(X_train, y_train)

with open("modelo_completo.pkl", "wb") as archivo:
    pickle.dump(modelo_pipeline, archivo)

print("Modelo entrenado y 'modelo_completo.pkl' creado con éxito.")
