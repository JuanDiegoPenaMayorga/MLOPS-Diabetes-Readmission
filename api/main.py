import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from prometheus_fastapi_instrumentator import Instrumentator

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME = "diabetes-readmission-predictor"
MODEL_STAGE = "Production"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = None
model_columns = None

app = FastAPI(title="Diabetes Readmission Predictor API", version="1.0.0")
Instrumentator().instrument(app).expose(app)
@app.on_event("startup")
def load_production_model():
    """Carga el modelo de MLflow y las columnas al iniciar."""
    global model, model_columns
    print(f"Cargando modelo '{MODEL_NAME}' en stage '{MODEL_STAGE}'...")
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        client = mlflow.tracking.MlflowClient()
        latest_version_info = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
        run_id = latest_version_info.run_id
        local_path = client.download_artifacts(run_id, "model_columns.json")
        with open(local_path, "r") as f:
            model_columns = json.load(f)["columns"]
        print("Modelo y columnas cargados con éxito.")
    except Exception as e:
        print(f"Error CRÍTICO al cargar el modelo: {e}")
        
class PredictionRequest(BaseModel):
    race: str; gender: str; age: str; time_in_hospital: int
    num_lab_procedures: int; num_procedures: int; num_medications: int
    diag_1: str; diag_2: str; diag_3: str

@app.get("/")
def read_root():
    return {"message": "API de predicción está activa."}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None: raise HTTPException(status_code=503, detail="Modelo no disponible. Error en el arranque.")
    try:
        input_data = pd.DataFrame([request.dict()])
        input_data_transformed = pd.get_dummies(input_data, drop_first=True, dtype=float)
        final_input = input_data_transformed.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(final_input)[0]
        prediction_proba = model.predict_proba(final_input)[0]
        result = "YES" if prediction == 1 else "NO"
        probability = float(prediction_proba[prediction])
        return {"prediction": result, "probability": f"{probability:.4f}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar: {e}")
