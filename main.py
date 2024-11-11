import os
from dotenv import load_dotenv
from batch_prediction import run_batch
from api_prediction import run_api
import logging

logging.basicConfig(level=logging.INFO)

def main():
    load_dotenv()
    deployment_type = os.getenv("DEPLOYMENT_TYPE")
    
    if deployment_type == "Batch":
        logging.info('Ejecutando predicción Batch')
        run_batch()
    elif deployment_type == "API":
        logging.info('Ejecutando predicción API')
        run_api()
    else:
        raise ValueError("DEPLOYMENT_TYPE no válido (Batch/API)")

if __name__ == "__main__":
    main()
