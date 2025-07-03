from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd


# ✅ Charger ton modèle entraîné
with open("gold_future_close_regressor.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Définir l'application
app = FastAPI(
    title="Gold Future Close Predictor",
    description="API to predict the expected close price of Gold (XAU/USD) in the next 12 hours given hourly technical features.",
    version="1.0"
)

# ✅ Schéma d'entrée EXACT des features du modèle
class Features(BaseModel):
    rsi: float
    ema_9: float
    ema_21: float
    ema_distance: float
    macd_line: float
    atr: float
    volatility_close_std: float
    ema_9_slope: float
    ema_21_slope: float

# ✅ Endpoint racine
@app.post("/predict")
def predict(features: Features):
    data = pd.DataFrame([features.dict()])
    predicted_close = model.predict(data)[0]
    
    # Optionnel : logique de seuil / signal côté backend
    last_close = features.close_dernier
    atr = features.atr
    volatility = features.volatility_close_std
    seuil_utilise = atr
    
    if predicted_close > last_close + seuil_utilise:
        signal = "BUY"
    elif predicted_close < last_close - seuil_utilise:
        signal = "SELL"
    else:
        signal = "NO_TRADE"
    
    return {
        "signal": signal,
        "predicted_close": round(predicted_close, 2),
        "last_close": round(last_close, 2),
        "seuil_utilisé": round(seuil_utilise, 2),
        "atr": round(atr, 2),
        "volatility_close_std": round(volatility, 2)
    }
