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
@app.get("/")
def home():
    return {"message": "API is live. Use POST /predict to get the future_close prediction."}

# ✅ Endpoint de prédiction
@app.post("/predict")
def predict_future_close(features: Features):
    # ✅ Transformer en dataframe
    df_input = pd.DataFrame([features.dict()])

    # ✅ Vérifier l'ordre des colonnes
    expected_cols = list(model.feature_names_in_)
    df_input = df_input[expected_cols]

    # ✅ Faire la prédiction
    prediction = float(model.predict(df_input)[0])

    # ✅ Retour structuré
    return {
        "predicted_future_close": round(prediction, 2)
    }
