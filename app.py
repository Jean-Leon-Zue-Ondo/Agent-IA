
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# ✅ Charger ton modèle entraîné
with open("gold_signal_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Gold Signal Prediction API")

# ✅ Schéma d'entrée
class Features(BaseModel):
    rsi: float
    ema_9: float
    ema_21: float
    macd_line: float

# ✅ Endpoint de prédiction
@app.post("/predict")
def predict_signal(features: Features):
    data = pd.DataFrame([features.dict()])
    prediction = int(model.predict(data)[0])
    proba = float(model.predict_proba(data)[0][1])

    return {
        "prediction": prediction,
        "probability_of_increase": round(proba * 100, 2)
    }
