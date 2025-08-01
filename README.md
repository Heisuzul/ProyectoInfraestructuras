# Proyecto de Entrenamiento Paralelo con Ray

Este proyecto demuestra cómo utilizar [Ray](https://www.ray.io/) para paralelizar el entrenamiento de modelos de machine learning en Python.

## Requisitos

- Python 3.10+
- Entorno virtual configurado
- Paquetes: `ray`, `pandas`, `scikit-learn`, entre otros (ver `requirements.txt` si existe)

## Pasos para ejecutar

1. Crear entorno virtual
  python -m venv .venv

2. **Activar el entorno virtual**  
   En Windows:
   ```bash
   .\.venv\Scripts\activate

3. Instalar dependencias
  pip install -r requirements.txt

4. Iniciar Ray localmente
  ray start --head

5. Ejecutar el servicio API
  python main.py

