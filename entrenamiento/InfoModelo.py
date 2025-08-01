# entrenamiento/InfoModelo.py
import joblib

def info_modelo(modo="paralelo"):
    rutas = {
        "paralelo": "entrenamiento/modelo_paralelo.pkl",
        "secuencial": "entrenamiento/modelo_secuencial.pkl"
    }

    ruta_modelo = rutas.get(modo)
    if not ruta_modelo:
        return {"error": f"Modo no válido: {modo}"}

    try:
        modelo = joblib.load(ruta_modelo)
        return {
            "modo": modo,
            "tipo_modelo": type(modelo).__name__,
            "parametros": modelo.get_params(),
            "n_estimators": getattr(modelo, "n_estimators", "N/A")
        }
    except Exception as e:
        return {"error": str(e)}
