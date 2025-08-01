# entrenamiento/PrediccionParalela.py
import joblib
import ray
import numpy as np

@ray.remote
def predecir_remoto(muestra, ruta_modelo):
    try:
        modelo = joblib.load(ruta_modelo)
        muestra_np = np.array(muestra).reshape(1, -1)
        return modelo.predict(muestra_np).tolist()[0]
    except Exception as e:
        return f"Error en predicción: {str(e)}"

def predecir_en_paralelo(lista_muestras, modo="paralelo"):
    rutas = {
        "paralelo": "entrenamiento/modelo_paralelo.pkl",
        "secuencial": "entrenamiento/modelo_secuencial.pkl"
    }

    ruta_modelo = rutas.get(modo)
    if not ruta_modelo:
        return {"error": f"Modo de predicción no válido: {modo}"}

    tareas = [predecir_remoto.remote(m, ruta_modelo) for m in lista_muestras]
    resultados = ray.get(tareas)
    return {"predicciones": resultados}
