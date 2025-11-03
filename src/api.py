from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(
    title="Restaurant Premium Membership Predictor",
    description="Predicts if a customer will purchase premium membership",
    version="1.0.0"
)

model = None
feature_names = None

@app.on_event("startup")
def load_model():
    global model, feature_names
    
    model_path = Path("models/best_model_GradientBoosting.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    
    sample_data = pd.read_parquet("data/processed/restaurante_clean.parquet")
    feature_names = [col for col in sample_data.columns if col != 'membresia_premium_Sí']
    
    print(f"Model loaded successfully: {model_path.name}")
    print(f"Features expected: {len(feature_names)}")


class CustomerInput(BaseModel):
    edad: float
    frecuencia_visita: int
    promedio_gasto_comida: float
    ingresos_mensuales: int
    estrato_socioeconomico: int
    genero_Masculino: int = 0
    ciudad_residencia_Bogota: int = 0
    ciudad_residencia_Cali: int = 0
    ciudad_residencia_Medellin: int = 0
    ocio_Cine: int = 0
    ocio_Deportes: int = 0
    ocio_Lectura: int = 0
    ocio_Musica: int = 0
    ocio_Videojuegos: int = 0
    consume_licor_Sí: int = 0
    preferencias_alimenticias_Pescetariana: int = 0
    preferencias_alimenticias_Vegana: int = 0
    preferencias_alimenticias_Vegetariana: int = 0
    tipo_de_pago_mas_usado_Efectivo: int = 0
    tipo_de_pago_mas_usado_Tarjeta: int = 0


@app.get("/")
def root():
    return {
        "message": "Restaurant Premium Membership Prediction API",
        "status": "active",
        "model": "GradientBoosting",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(feature_names) if feature_names else 0
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_dict = customer.dict()
        input_df = pd.DataFrame([input_dict])
        
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[feature_names]
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        return {
            "prediction": "Premium" if prediction == 1 else "Regular",
            "probability_regular": float(probability[0]),
            "probability_premium": float(probability[1]),
            "recommendation": "Target for premium campaign" if probability[1] > 0.5 else "Not recommended"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/features")
def get_features():
    if feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "features": feature_names,
        "count": len(feature_names)
    }