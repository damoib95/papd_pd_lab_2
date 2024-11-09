import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from model import train_model, preprocess_data

def run_api():
    app = FastAPI(
        title="API para Predicciones",
        description="Servicio para realizar predicciones utilizando un modelo previamente entrenado",
        version="1.0.0"
    )

    model = train_model()

    @app.get('/health', summary='Health check', description='Verificar el estado de la API')
    async def health_check():
        print('Health check')
        return {'status': 'ok'}

    @app.post('/predict', summary='Predicción', description='Realizar predicciones a partir de datos enviados')
    async def predict(file: UploadFile = File(...)):
        print('Petición de predicción recibida')

        try:
            df = pd.read_parquet(file.file)

            if df.empty:
                raise HTTPException(status_code=400, detail="El archivo Parquet está vacío.")

            X_input, _ = preprocess_data(df, fit=False)

            predictions = model.predict_proba(X_input)
            
            predictions_formatted = [
                {f"Clase{i+1}": prob for i, prob in enumerate(prob_row)}
                for prob_row in predictions
            ]

            return {"predictions": predictions_formatted}

        except Exception as e:
            print(f'Error al realizar la predicción: {str(e)}')
            raise HTTPException(status_code=500, detail=str(e))

    import uvicorn
    print('Iniciando API...')
    uvicorn.run(app, host="0.0.0.0", port=8000)

