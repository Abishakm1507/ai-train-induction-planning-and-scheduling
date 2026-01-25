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
# Load processed data once at startup
# -------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed-data.csv")
try:
    df_cached = pd.read_csv(DATA_PATH)
    # Pre-calculate hour to speed up filtering
    df_cached['hour_val'] = df_cached['arrival_time'].astype(str).str.split(':').str[0].astype(int) % 24
except Exception as e:
    print(f"Warning: Could not load processed data: {e}")
    df_cached = pd.DataFrame()

# -------------------------------------------------
# Constants
# -------------------------------------------------
TRAIN_CAPACITY = 1000

STATIONS = [
    "Aluva", "Pulinchodu", "Companypady", "Ambattukavu", "Muttom", 
    "Kalamassery", "Pathadipalam", "Cochin University", "Edapally", 
    "Changampuzha Park", "Palarivattom", "JLN Stadium", "Kaloor", 
    "Town Hall", "MG Road", "Maharajas College", "Ernakulam South", 
    "Kadavanthra", "Elamkulam", "Vyttila", "Thykoodam", "Pettah", 
    "Vadakkekotta", "SN Junction", "Tripunithura"
]

# -------------------------------------------------
# Weather API integration
# -------------------------------------------------
def get_weather_data(city: str = "Kochi"):
    api_key = os.getenv("WEATHER_API_KEY", "4bbb090bd7be475fabb71537262301")

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCdVsXfVyuBpOVserT_qnE3tt7CUy3CuX0")
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-1.5-flash")

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
        if df_cached.empty:
            raise Exception("Processed data not available")

        # Basic Preprocessing
        is_weekend = 1 if day_type == "weekend" else 0
        hours_range = list(range(start_hour, end_hour + 1))

        # Calculate service frequency once
        train_freq = (
            df_cached.groupby(['stop_name', 'hour_val'])
            .agg(trains_per_hour=('trip_id', 'nunique'))
            .reset_index()
        )

        # 1. Prepare ALL rows for batch prediction
        rows = []
        for station in STATIONS:
            for hour in hours_range:
                freq_row = train_freq[(train_freq['stop_name'] == station) & (train_freq['hour_val'] == hour)]
                tph = freq_row['trains_per_hour'].values[0] if not freq_row.empty else 1
                is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0
                
                rows.append({
                    'hour': hour,
                    'is_weekend': is_weekend,
                    'is_peak_hour': is_peak,
                    'trains_per_hour': tph,
                    'direction_id': 0,
                    'station': station, # Keep track for mapping later
                    'hour_idx': hour
                })

        batch_df = pd.DataFrame(rows)
        # Ensure only model features are used
        input_df = batch_df.reindex(columns=features, fill_value=0)
        
        # 2. Single batch prediction (VERY FAST)
        predictions = model.predict(input_df)
        batch_df['prediction'] = predictions.astype(int)

        # 3. Format into the [station][hour] 2D array the frontend expects
        heatmap_values = []
        for station in STATIONS:
            station_preds = batch_df[batch_df['station'] == station].sort_values('hour_idx')['prediction'].tolist()
            heatmap_values.append(station_preds)

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
        if df_cached.empty:
            raise Exception("Processed data not available")
            
        is_weekend = 1 if day_type == "weekend" else 0
        is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0
        
        results = []
        for station in STATIONS:
            # Find typical trains per hour for this station and hour
            station_hour_data = df_cached[(df_cached['stop_name'] == station) & (df_cached['hour_val'] == hour)]
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

@router.get("/historical")
async def get_historical_demand(day_type: str = "weekday", start_hour: int = 6, end_hour: int = 22):
    """
    GET /api/demand/historical
    Returns aggregate historical passenger demand per hour.
    """
    try:
        if df_cached.empty:
            raise Exception("Processed data not available")

        is_weekend = 1 if day_type == "weekend" else 0
        
        # Filter by day type
        df_filtered = df_cached[df_cached['is_weekend'] == is_weekend]
        
        # Group by hour and sum passengers, then divide by number of unique trips to get average per hour
        # Actually, for total aggregate demand we just sum by hour
        historical_trend = (
            df_filtered.groupby('hour_val')['passengers']
            .sum()
            .reset_index()
        )
        
        # Filter by hour range
        historical_trend = historical_trend[
            (historical_trend['hour_val'] >= start_hour) & 
            (historical_trend['hour_val'] <= end_hour)
        ]
        
        # Sort by hour
        historical_trend = historical_trend.sort_values('hour_val')
        
        return {
            "hours": historical_trend['hour_val'].tolist(),
            "historical": historical_trend['passengers'].tolist(),
            "success": True
        }

    except Exception as e:
        print(f"Historical API Error: {str(e)}")
        return {"error": str(e), "success": False}

@router.get("/trend")
async def get_demand_trend(start_hour: int = 8, day_type: str = "weekday"):
    """
    GET /api/demand/trend
    Predicts aggregate demand for the next 12 hours.
    """
    try:
        if df_cached.empty:
            raise Exception("Processed data not available")
            
        is_weekend = 1 if day_type == "weekend" else 0
        trend_hours = [(start_hour + i) % 24 for i in range(12)]

        # Prepare all station/hour combos for batch
        rows = []
        for hour in trend_hours:
            is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0
            for station in STATIONS:
                station_hour_data = df_cached[(df_cached['stop_name'] == station) & (df_cached['hour_val'] == hour)]
                tph = station_hour_data['trip_id'].nunique() if not station_hour_data.empty else 5
                
                rows.append({
                    'hour': hour,
                    'is_weekend': is_weekend,
                    'is_peak_hour': is_peak,
                    'trains_per_hour': tph,
                    'direction_id': 0,
                    'trend_hour_val': hour
                })

        batch_df = pd.DataFrame(rows)
        # Ensure only model features are used
        input_df = batch_df.reindex(columns=features, fill_value=0)
        
        # Batch predict
        batch_df['prediction'] = model.predict(input_df)

        # Aggregate demand by hour
        hourly_demand = batch_df.groupby('trend_hour_val')['prediction'].sum()
        
        hours_labels = []
        demand_values = []
        for hour in trend_hours:
            hours_labels.append(f"{hour:02d}:00")
            demand_values.append(int(hourly_demand[hour]))
            
        return {
            "hours": hours_labels,
            "demand": demand_values
        }

    except Exception as e:
        print(f"Demand Trend API Error: {str(e)}")
        return {"error": str(e)}
