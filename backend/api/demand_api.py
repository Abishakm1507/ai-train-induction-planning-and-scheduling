import os
import numpy as np
import pandas as pd
import joblib
import requests
from fastapi import APIRouter
import google.generativeai as genai

# -------------------------------------------------
# Path configuration (IMPORTANT)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# -------------------------------------------------
# Load ML model and features
# -------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "demand_forecast_model.pkl"))
features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))

# -------------------------------------------------
# Constants
# -------------------------------------------------
TRAIN_CAPACITY = 1000

STATIONS = [
    "Aluva", "Pulinchodu", "Companypady", "Ambattukavu",
    "Muttom", "Kalamassery", "CUSAT", "Edappally",
    "Kaloor", "MG Road", "Maharaja’s", "Ernakulam South"
]

# -------------------------------------------------
# Weather API integration
# -------------------------------------------------
def get_weather_data(city: str = "Kochi"):
    api_key = os.getenv("WEATHER_API_KEY")

    if not api_key:
        return {"temp": 28, "rain_mm": 0, "condition": "Clear"}

    url = (
        f"http://api.weatherapi.com/v1/current.json"
        f"?key={api_key}&q={city}&aqi=no"
    )

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        return {
            "temp": data["current"]["temp_c"],
            "rain_mm": data["current"]["precip_mm"],
            "condition": data["current"]["condition"]["text"]
        }
    except Exception:
        return {"temp": 28, "rain_mm": 0, "condition": "Clear"}

# -------------------------------------------------
# Weather demand impact
# -------------------------------------------------
def weather_demand_multiplier(weather: dict) -> float:
    factor = 1.0

    if weather["rain_mm"] > 5:
        factor += 0.20
    if "Heavy" in weather["condition"]:
        factor += 0.10
    if weather["temp"] > 34:
        factor -= 0.05

    return factor

# -------------------------------------------------
# LLM (Gemini) configuration
# -------------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.5-flash")

def generate_llm_explanation(demand: int, weather: dict, is_peak: int):
    """
    Natural language explanation for dashboard (optional UI use)
    """
    prompt = f"""
Explain the metro passenger demand situation.

Context:
- Estimated passenger demand: {demand}
- Weather: {weather['condition']}
- Rainfall: {weather['rain_mm']} mm
- Temperature: {weather['temp']}°C
- Peak hour: {"Yes" if is_peak else "No"}

Rules:
- Simple professional language
- Max 2–3 sentences
- Do not mention AI or algorithms
"""

    try:
        return llm.generate_content(prompt).text.strip()
    except Exception:
        return "Passenger demand is influenced by peak-hour travel patterns and prevailing weather conditions."

# -------------------------------------------------
# Demand prediction logic
# -------------------------------------------------
def predict_passenger_demand(input_data: dict):
    """
    Time-series + ML based passenger demand prediction
    """

    # Prepare input dataframe
    df = pd.DataFrame([input_data])

    # Ensure correct feature order
    df = df.reindex(columns=features, fill_value=0)

    # Base ML prediction
    base_demand = int(model.predict(df)[0])

    # Weather adjustment
    weather = get_weather_data()
    weather_factor = weather_demand_multiplier(weather)

    predicted_demand = int(base_demand * weather_factor)

    # Optional explanation (for UI)
    explanation = generate_llm_explanation(
        predicted_demand,
        weather,
        input_data.get("is_peak_hour", 0)
    )

    return {
        "predicted_demand": predicted_demand,
        "weather": weather,
        "explanation": explanation
    }

# -------------------------------------------------
# FastAPI Router
# -------------------------------------------------
router = APIRouter()

@router.post("/predict")
async def predict_demand(payload: dict):
    """
    POST /api/demand/predict
    """
    return predict_passenger_demand(payload or {})
