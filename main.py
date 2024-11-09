import os
from dotenv import load_dotenv
from batch_prediction import run_batch
from api_prediction import run_api

def main():
    load_dotenv()
    deployment_type = os.getenv("DEPLOYMENT_TYPE")
    
    if deployment_type == "Batch":
        run_batch()
    elif deployment_type == "API":
        run_api()
    else:
        raise ValueError("DEPLOYMENT_TYPE no válido (Batch/API)")

if __name__ == "__main__":
    main()
