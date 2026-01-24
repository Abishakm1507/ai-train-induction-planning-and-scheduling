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
    "Muttom", "Kalamassery", "Cochin University", "Edapally",
    "Kaloor", "MG Road", "Maharajas College", "Ernakulam South"
]

# -------------------------------------------------
# Weather API integration
# -------------------------------------------------
def get_weather_data(city: str = "Kochi"):
    api_key = "4bbb090bd7be475fabb71537262301"

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
genai.configure(api_key="AIzaSyCdVsXfVyuBpOVserT_qnE3tt7CUy3CuX0")
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

@router.get("/heatmap")
async def get_heatmap_data(day_type: str = "weekday", start_hour: int = 6, end_hour: int = 22):
    """
    GET /api/demand/heatmap
    Dynamically generates station-wise demand predictions using the ML model.
    """
    try:
        # Load processed data
        data_path = os.path.join(BASE_DIR, "data", "processed", "processed-data.csv")
        df_csv = pd.read_csv(data_path)

        # Basic Preprocessing (similar to notebook)
        df_csv['hour'] = df_csv['arrival_time'].astype(str).str.split(':').str[0].astype(int) % 24
        is_weekend = 1 if day_type == "weekend" else 0

        # Calculate service frequency (trains per hour per stop)
        train_freq = (
            df_csv.groupby(['stop_name', 'hour'])
            .agg(trains_per_hour=('trip_id', 'nunique'))
            .reset_index()
        )

        # Predictions storage
        hours_range = list(range(start_hour, end_hour + 1))
        heatmap_values = []

        for station in STATIONS:
            station_predictions = []
            for hour in hours_range:
                # Find service frequency for this station and hour
                freq_row = train_freq[(train_freq['stop_name'] == station) & (train_freq['hour'] == hour)]
                tph = freq_row['trains_per_hour'].values[0] if not freq_row.empty else 1
                
                # Prepare features
                is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0
                
                input_df = pd.DataFrame([{
                    'hour': hour,
                    'is_weekend': is_weekend,
                    'is_peak_hour': is_peak,
                    'trains_per_hour': tph,
                    'direction_id': 0 # Default to inbound/outbound average (0)
                }])
                
                # Ensure feature order
                input_df = input_df.reindex(columns=features, fill_value=0)
                
                # Predict
                pred = model.predict(input_df)[0]
                station_predictions.append(int(pred))
            
            heatmap_values.append(station_predictions)

        return {
            "stations": STATIONS,
            "hours": hours_range,
            "values": heatmap_values
        }

    except Exception as e:
        print(f"Heatmap API Error: {str(e)}")
        return {"error": str(e)}

@router.get("/line/status")
async def get_line_status(hour: int = 9, day_type: str = "weekday"):
    """
    GET /api/demand/line/status
    Predicts passenger demand per station and determines congestion status.
    """
    try:
        is_weekend = 1 if day_type == "weekend" else 0
        is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0
        
        # Load processed data to get station-specific trains_per_hour
        data_path = os.path.join(BASE_DIR, "data", "processed", "processed-data.csv")
        df_csv = pd.read_csv(data_path)
        df_csv['hour_val'] = df_csv['arrival_time'].astype(str).str.split(':').str[0].astype(int) % 24
        
        results = []
        for station in STATIONS:
            # Find typical trains per hour for this station and hour
            station_hour_data = df_csv[(df_csv['stop_name'] == station) & (df_csv['hour_val'] == hour)]
            tph = station_hour_data['trip_id'].nunique() if not station_hour_data.empty else 5
            
            input_df = pd.DataFrame([{
                'hour': hour,
                'is_weekend': is_weekend,
                'is_peak_hour': is_peak,
                'trains_per_hour': tph,
                'direction_id': 0
            }])
            input_df = input_df.reindex(columns=features, fill_value=0)
            
            predicted_demand = int(model.predict(input_df)[0])
            
            # Categorize status
            if predicted_demand < 400:
                status = "Normal"
            elif predicted_demand <= 650:
                status = "Moderate"
            else:
                status = "Congested"
                
            results.append({
                "station": station,
                "passengers": predicted_demand,
                "status": status
            })
            
        return results

    except Exception as e:
        print(f"Line Status API Error: {str(e)}")
        return {"error": str(e)}

@router.get("/trend")
async def get_demand_trend(start_hour: int = 8, day_type: str = "weekday"):
    """
    GET /api/demand/trend
    Predicts aggregate demand for the next 12 hours.
    """
    try:
        is_weekend = 1 if day_type == "weekend" else 0
        
        # Load processed data for TPH averages
        data_path = os.path.join(BASE_DIR, "data", "processed", "processed-data.csv")
        df_csv = pd.read_csv(data_path)
        df_csv['hour_val'] = df_csv['arrival_time'].astype(str).str.split(':').str[0].astype(int) % 24

        hours = []
        demand_values = []
        
        for i in range(12):
            hour = (start_hour + i) % 24
            is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0
            
            # Sum predictions across all stations for this hour
            total_hour_demand = 0
            for station in STATIONS:
                station_hour_data = df_csv[(df_csv['stop_name'] == station) & (df_csv['hour_val'] == hour)]
                tph = station_hour_data['trip_id'].nunique() if not station_hour_data.empty else 5
                
                input_df = pd.DataFrame([{
                    'hour': hour,
                    'is_weekend': is_weekend,
                    'is_peak_hour': is_peak,
                    'trains_per_hour': tph,
                    'direction_id': 0
                }])
                input_df = input_df.reindex(columns=features, fill_value=0)
                total_hour_demand += model.predict(input_df)[0]
                
            hours.append(f"{hour:02d}:00")
            demand_values.append(int(total_hour_demand))
            
        return {
            "hours": hours,
            "demand": demand_values
        }

    except Exception as e:
        print(f"Demand Trend API Error: {str(e)}")
        return {"error": str(e)}
