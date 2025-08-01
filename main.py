# main.py
import ray
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from entrenamiento.EntrenamientoParalelo import ejecutar_entrenamiento
from entrenamiento.EntrenamientoSecuencial import ejecutar_entrenamiento_secuencial
from entrenamiento.Prediccion import predecir_en_paralelo
from entrenamiento.InfoModelo import info_modelo
import joblib

app = FastAPI()

class ListaMuestras(BaseModel):
    muestras: list[list[float]]

class ListaParametros(BaseModel):
    parametros: list[int]

@serve.deployment
@serve.ingress(app)
class APIModelos:

    # Entrenamiento
    @app.post("/entrenar-paralelo")
    def entrenar_paralelo(self, entrada: ListaParametros):
        return ejecutar_entrenamiento(entrada.parametros)

    @app.post("/entrenar-secuencial")
    def entrenar_secuencial(self, entrada: ListaParametros):
        return ejecutar_entrenamiento_secuencial(entrada.parametros)

    # Predicción
    @app.get("/predecir-paralelo")
    def predecir_paralelo(self, n: int = 5):
        return self._predecir_con_modo(n, "paralelo")

    @app.get("/predecir-secuencial")
    def predecir_secuencial(self, n: int = 5):
        return self._predecir_con_modo(n, "secuencial")

    def _predecir_con_modo(self, n: int, modo: str):
        try:
            muestras = joblib.load("entrenamiento/muestras_validas.pkl")
            muestras_seleccionadas = muestras[:n].values.tolist()
            resultado = predecir_en_paralelo(muestras_seleccionadas, modo=modo)
            return {"modo": modo, "predicciones": resultado["predicciones"]}
        except Exception as e:
            return {"error": str(e)}

    # Información del modelo
    @app.get("/info-modelo-paralelo")
    def info_paralelo(self):
        return info_modelo(modo="paralelo")

    @app.get("/info-modelo-secuencial")
    def info_secuencial(self):
        return info_modelo(modo="secuencial")

if __name__ == "__main__":
    ray.shutdown()
    ray.init()
    serve.run(APIModelos.bind())
