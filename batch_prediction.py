import os
import time
import shutil
import pandas as pd
from model import train_model, preprocess_data
import joblib
import logging

logging.basicConfig(level=logging.INFO)

def run_batch():
    logging.info('Iniciando modelamiento')
    model = train_model()

    INPUT_FOLDER = os.getenv("INPUT_FOLDER")
    OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
    PROCESSED_FOLDER = os.path.join(INPUT_FOLDER, 'processed')
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    logging.info(f"Esperando nuevos archivos en: {INPUT_FOLDER}")

    inactivity_counter = 0
    MAX_INACTIVITY_CYCLES = 6

    while True:
        archivos_nuevos = [file for file in os.listdir(INPUT_FOLDER) if file.endswith('.parquet')]

        if not archivos_nuevos:
            logging.warning("No se encontraron archivos nuevos. Esperando...")
            inactivity_counter += 1
            if inactivity_counter >= MAX_INACTIVITY_CYCLES:
                logging.warning("No se encontraron archivos nuevos durante 1 minuto. Finalizando el proceso...")
                break
            time.sleep(10)
            continue
        
        inactivity_counter = 0

        for file_name in archivos_nuevos:
            logging.info(f"Procesando archivo: {file_name}")
            input_file = os.path.join(INPUT_FOLDER, file_name)
            input_df = pd.read_parquet(input_file)

            X_input, _ = preprocess_data(input_df, target_column=None, fit=False)

            probabilities = model.predict_proba(X_input)

            predictions = [{f"Clase{i+1}": prob for i, prob in enumerate(prob_row)} for prob_row in probabilities]

            output_df = pd.DataFrame(predictions)
            output_file = os.path.join(OUTPUT_FOLDER, f"predictions_{file_name}")
            output_df.to_parquet(output_file)

            logging.info(f"Predicciones guardadas en: {output_file}")

            shutil.move(input_file, os.path.join(PROCESSED_FOLDER, file_name))
            logging.info(f"Archivo procesado movido a: {PROCESSED_FOLDER}")
