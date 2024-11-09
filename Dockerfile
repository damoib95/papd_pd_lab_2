# Imagen base de Python
FROM python:3.12-slim

# Determinamos nuestro directorio de trabajo
WORKDIR /app

# Dependencias y librerias
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean

# Agregamos nuestros archivos
COPY . /app

# Instalar requirements
RUN pip install --no-cache-dir -r requirements.txt

# Como se ejecuta
EXPOSE 8000

CMD ["python", "main.py"]

