FROM python:3.10-slim

# Instala dependencias del sistema (si necesitas)
RUN apt-get update && apt-get install -y build-essential

# Copia los archivos del proyecto
WORKDIR /app
COPY . .

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Da permisos de ejecución al script
RUN chmod +x start.sh

# Expone el puerto de la API
EXPOSE 8000

# Comando final: ejecuta el script que inicia Ray y luego main.py
CMD ["./start.sh"]
