# papd_pd_lab_2
## Laboratorio 2: Despliegue de AutoML en Docker (Automated Machine Learning)

Este repositorio contiene el código y los materiales utilizados para el **Laboratorio 2** del curso **Product Development** del postgrado en **Análisis y Predicción de Datos** de la **Maestría en Data Science** en la **Universidad Galileo**.

En este laboratorio, se ha implementado un flujo de trabajo de **AutoML** que se despliega en un contenedor Docker. Se proporcionan dos modos de ejecución: **Batch Prediction** y **API Prediction**.

---

### Prerrequisitos

Antes de comenzar, asegúrate de tener lo siguiente:

- **Docker** instalado en tu máquina.
- Un archivo de datos de entrada en el directorio local especificado como `localpath/data` el cual será utilizado para entrenar y optimizar el modelo.
- Los archivos de configuración `.env` para cada modo de ejecución.

---

### Construcción del contenedor Docker

Primero, debes construir la imagen Docker utilizando el siguiente comando. Este comando crea la imagen **auto-ml** con la última versión:

```bash
docker build -t auto-ml:latest .
```

Este comando buscará el archivo `Dockerfile` en el directorio actual y construirá la imagen Docker. Asegúrate de ejecutar este comando en el directorio raíz del proyecto, donde se encuentra el `Dockerfile`.

---

### Ejecución del contenedor Docker

Una vez que la imagen se haya construido, puedes ejecutar el contenedor en uno de los dos modos disponibles: **Batch Prediction** o **API Prediction**.

#### Modo Batch Prediction

En el modo **Batch Prediction**, el contenedor ejecutará el modelo en un lote de datos. En esta modalidad, el programa se queda a la espera de archivos nuevos ubicados en la ruta `/app/data/input` para aplicar el preprocesamiento y predicción del modelo optimizado. Una vez finaliza de procesar los archivos, los mueve a la ubicación `/app/data/input/processed` y busca en intervalos de 10 segundos por archivos nuevos. Si no se reciben archivos nuevos durante 60 segundos, entonces, el programa finaliza automáticamente.

Asegúrate de tener el archivo de configuración adecuado para este modo (`batch_prediction.env`), que debe contener las variables de entorno necesarias.

Para ejecutar el contenedor en modo Batch Prediction, usa el siguiente comando:

```bash
docker run --env-file batch_prediction.env -v "localpath/data":/app/data auto-ml:latest
```

Explicación del comando:
- `--env-file batch_prediction.env`: Carga las variables de entorno del archivo `batch_prediction.env`.
- `-v "localpath/data":/app/data`: Monta el directorio `localpath/data` de tu máquina local dentro del contenedor en `/app/data`.
- `auto-ml:latest`: Especifica la imagen Docker que acabas de construir.

---

#### Modo API Prediction

En el modo **API Prediction**, el contenedor ejecutará un servidor de API que permite realizar predicciones individuales a través de peticiones HTTP. Para este modo, necesitas tener el archivo de configuración adecuado (`api_prediction.env`).

Para ejecutar el contenedor en modo API Prediction, usa el siguiente comando:

```bash
docker run --env-file api_prediction.env -v "localpath/data":/app/data -p 8000:8000 auto-ml:latest
```

Explicación del comando:
- `--env-file api_prediction.env`: Carga las variables de entorno del archivo `api_prediction.env`.
- `-v "localpath/data":/app/data`: Monta el directorio `localpath/data` de tu máquina local dentro del contenedor en `/app/data`.
- `-p 8000:8000`: Expone el puerto 8000 del contenedor en el puerto 8000 de tu máquina local.
- `auto-ml:latest`: Especifica la imagen Docker que acabas de construir.

Este modo te permite hacer predicciones en tiempo real mediante una API RESTful. Después de ejecutar el contenedor, puedes realizar solicitudes HTTP `POST` a `http://localhost:8000/predict` para obtener predicciones.

---

### Archivos y variables de entorno

#### `batch_prediction.env`

Este archivo debe contener las variables de entorno necesarias para la ejecución en modo **Batch Prediction**. Un ejemplo de archivo podría ser:

```
DATASET=data/dataset.parquet
TARGET=Churn
MODEL=NaiveBayes
TRIALS=2
DEPLOYMENT_TYPE=Batch

INPUT_FOLDER=data/input
OUTPUT_FOLDER=data/output
```

#### `api_prediction.env`

Este archivo debe contener las variables de entorno necesarias para la ejecución en modo **API Prediction**. Un ejemplo de archivo podría ser:

```
DATASET=data/dataset.parquet
TARGET=Churn
MODEL=GradientBoosting
TRIALS=2
DEPLOYMENT_TYPE=API

PORT=8000
```

---

### Uso de la API

Una vez que el contenedor esté en ejecución en modo API, puedes hacer peticiones `POST` a la API para obtener predicciones. Aquí tienes un ejemplo de cómo hacer una solicitud mediante **Python**:

```python
import pandas as pd
import requests
import json

df = pd.read_parquet('data/input/test_1.parquet')
data_json = df.to_dict(orient='records')

url = 'http://localhost:8000/predict'
response = requests.post(url, json={"data": data_json})

print(json.dumps(response.json(), indent=2))
```

Esta solicitud devolverá una respuesta JSON con la predicción.
