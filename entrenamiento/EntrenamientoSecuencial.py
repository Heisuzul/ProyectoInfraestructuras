#Entrenamiento secuencial 
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
import joblib
import numpy as np

def ejecutar_entrenamiento_secuencial(parametros):
    # Cargar y preparar los datos
    data = fetch_openml(name='credit-g', version=1, as_frame=True)
    X = pd.get_dummies(data.data)
    y = LabelEncoder().fit_transform(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    inicio = time.time()

    mejor_precision = 0
    mejor_modelo = None
    modelos_entrenados = []

    for n_estimators in parametros:
        modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=1)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        print(f"Precisión con {n_estimators} árboles:", precision)

        # Guardar cada modelo individualmente
        ruta_individual = f"entrenamiento/modelo_secuencial_{n_estimators}.pkl"
        joblib.dump(modelo, ruta_individual)
        modelos_entrenados.append({
            "n_estimators": n_estimators,
            "precision": precision,
            "ruta_modelo": ruta_individual
        })

        if precision > mejor_precision:
            mejor_precision = precision
            mejor_modelo = modelo

    fin = time.time()

    # Guardar el mejor modelo
    ruta_modelo = "entrenamiento/modelo_secuencial.pkl"
    joblib.dump(mejor_modelo, ruta_modelo)

    # Guardar 10 muestras aleatorias
    idx = np.random.choice(len(X_train), size=10, replace=False)
    muestras_aleatorias = X_train.iloc[idx]
    joblib.dump(muestras_aleatorias, "entrenamiento/muestras_validas.pkl")

    return {
        "mejor_precision": mejor_precision,
        "ruta_modelo": ruta_modelo,
        "duracion": round(fin - inicio, 2),
        "modelos_entrenados": modelos_entrenados
    }