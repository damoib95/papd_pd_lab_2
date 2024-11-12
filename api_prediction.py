import os
from dotenv import load_dotenv
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict
from model import train_model
import logging

class PredictionInput(BaseModel):
    data: List[Dict]

logging.basicConfig(level=logging.INFO)

def run_api():
    app = FastAPI(
        title="API para Predicciones",
        description="Servicio para realizar predicciones utilizando un modelo previamente entrenado",
        version="1.0.0"
    )

    logging.info('Iniciando modelamiento')
    model = train_model()

    @app.get('/health', summary='Health check', description='Verificar el estado de la API')
    async def health_check():
        logging.info('Health check')
        return {'status': 'ok'}

    @app.post('/predict', summary='Predicción', description='Realizar predicciones a partir de datos enviados')
    async def predict(input_data: PredictionInput):
        logging.info('Petición de predicción recibida')

        try:
            df = pd.DataFrame(input_data.data)

            if df.empty:
                raise HTTPException(status_code=400, detail="Los datos JSON están vacíos.")

            if os.path.exists('preprocessor.pkl'):
                preprocessor = joblib.load('preprocessor.pkl')
                X_input = preprocessor.transform(df)
            else:
                raise FileNotFoundError("El preprocesador entrenado no se encuentra.")

            predictions = model.predict_proba(X_input)
            
            predictions_formatted = [
                {f"Clase{i+1}": prob for i, prob in enumerate(prob_row)}
                for prob_row in predictions
            ]

            return {"predictions": predictions_formatted}

        except Exception as e:
            logging.info(f'Error al realizar la predicción: {str(e)}')
            raise HTTPException(status_code=500, detail=str(e))

    load_dotenv()
    port = os.getenv("PORT", 8000)
    import uvicorn
    logging.info('Iniciando API...')
    uvicorn.run(app, host="0.0.0.0", port=port)

