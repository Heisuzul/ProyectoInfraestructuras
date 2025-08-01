# Entrenamiento paralelo con Ray
import ray
import time
import pandas as pd
import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

ray.init(ignore_reinit_error=True)

@ray.remote
def entrenar_y_evaluar(n_estimators, X_train, X_test, y_train, y_test):
    modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    return (n_estimators, precision, modelo)

def ejecutar_entrenamiento(parametros):
    data = fetch_openml(name='credit-g', version=1, as_frame=True)
    X = pd.get_dummies(data.data)
    y = LabelEncoder().fit_transform(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    inicio = time.time()
    futuros = [entrenar_y_evaluar.remote(n, X_train, X_test, y_train, y_test) for n in parametros]
    resultados = ray.get(futuros)

    mejor_precision = 0
    mejor_modelo = None
    todos_los_modelos = []

    for n, precision, modelo in resultados:
        nombre_modelo = f"entrenamiento/modelo_paralelo_{n}.pkl"
        joblib.dump(modelo, nombre_modelo)
        todos_los_modelos.append({
            "n_estimators": n,
            "precision": precision,
            "ruta_modelo": nombre_modelo
        })
        print(f"Precisión con {n} árboles:", precision)

        if precision > mejor_precision:
            mejor_precision = precision
            mejor_modelo = modelo

    fin = time.time()

    # Guardar el mejor modelo
    ruta_modelo = "entrenamiento/modelo_paralelo.pkl"
    joblib.dump(mejor_modelo, ruta_modelo)

    # Guardar muestras para predicción
    idx = np.random.choice(len(X_train), size=10, replace=False)
    muestras_aleatorias = X_train.iloc[idx]
    joblib.dump(muestras_aleatorias, "entrenamiento/muestras_validas.pkl")

    return {
        "mejor_precision": mejor_precision,
        "ruta_modelo": ruta_modelo,
        "duracion": round(fin - inicio, 2),
        "modelos_entrenados": todos_los_modelos
    }
