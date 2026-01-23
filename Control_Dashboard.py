import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import requests
import google.generativeai as genai

genai.configure(api_key="...")
llm = genai.GenerativeModel("gemini-2.5-flash")

model = joblib.load("./model/demand_forecast_model.pkl")
features = joblib.load("./model/model_features.pkl")

try:
    q_table = joblib.load("./model/rl_q_table.pkl")
    rl_ready = True
except:
    q_table = {}
    rl_ready = False

TRAIN_CAPACITY = 1000
MIN_TRAINS = 2
MAX_TRAINS = 10

STATIONS = [
    "Aluva", "Pulinchodu", "Companypady", "Ambattukavu",
    "Muttom", "Kalamassery", "CUSAT", "Edappally",
    "Kaloor", "MG Road", "Maharaja’s", "Ernakulam South"
]
def get_weather_data(city="Kochi"):
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"

    try:
        r = requests.get(url, timeout=5)
        d = r.json()
        return {
            "temp": d["current"]["temp_c"],
            "rain_mm": d["current"]["precip_mm"],
            "condition": d["current"]["condition"]["text"]
        }
    except:
        return {"temp": 28, "rain_mm": 0, "condition": "Clear"}

def weather_demand_multiplier(w):
    factor = 1.0
    if w["rain_mm"] > 5:
        factor += 0.20
    if "Heavy" in w["condition"]:
        factor += 0.10
    if w["temp"] > 34:
        factor -= 0.05
    return factor

def get_demand_level(demand):
    if demand < 3000:
        return 0
    elif demand < 6000:
        return 1
    else:
        return 2

def rl_train_induction(predicted_demand, is_peak):
    state = (get_demand_level(predicted_demand), is_peak)
    actions = list(range(MIN_TRAINS, MAX_TRAINS + 1))
    q_vals = [q_table.get((state, a), 0) for a in actions]
    return actions[np.argmax(q_vals)]

def generate_llm_explanation(demand, trains, weather, is_peak):
    prompt = f"""
Explain the metro operational decision.

Context:
- Passenger demand: {demand}
- Trains deployed: {trains}
- Weather: {weather['condition']}, Rain {weather['rain_mm']} mm, {weather['temp']}°C
- Peak hour: {"Yes" if is_peak else "No"}

Rules:
- Simple professional language
- Max 3 sentences
- Do not mention AI or algorithms
"""
    return llm.generate_content(prompt).text.strip()

st.set_page_config(page_title="KMRL AI Operations Dashboard", layout="wide")
st.title("🚆 Kochi Metro – AI Operations Control Dashboard")
st.caption("Weather-Aware Demand Forecasting & Train Induction")

c1, c2, c3, c4, c5 = st.columns(5)
hour = c1.slider("Hour", 0, 23, 9)
day_type = c2.selectbox("Day Type", ["Weekday", "Weekend"])
trains_per_hour = c3.slider("Current Trains / Hour", 1, 10, 5)
direction_id = c4.selectbox("Direction", [0, 1])
run_ai = c5.button("🔮 Run AI")

is_weekend = 1 if day_type == "Weekend" else 0
is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0

predicted_demand = 0
recommended_trains = 0
weather = {"temp": 0, "rain_mm": 0, "condition": "-"}
weather_factor = 1.0

if run_ai:
    input_df = pd.DataFrame([{
        "hour": hour,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak,
        "trains_per_hour": trains_per_hour,
        "direction_id": direction_id
    }])[features]

    base_demand = model.predict(input_df)[0]

    weather = get_weather_data()
    weather_factor = weather_demand_multiplier(weather)

    predicted_demand = int(base_demand * weather_factor)
    recommended_trains = rl_train_induction(predicted_demand, is_peak)

left, center, right = st.columns([2.2, 3, 2])

with left:
    st.subheader("🗺️ Metro Line Status")
    congestion = np.clip(predicted_demand / 8000, 0, 1)

    for i, s in enumerate(STATIONS):
        color = "🟢" if congestion < 0.5 else "🟡" if congestion < 0.8 else "🔴"
        icon = "🚆" if i == hour % len(STATIONS) else "➖"
        st.markdown(f"{color} **{s}** {icon}")

with right:
    st.subheader("📊 KPIs")
    if run_ai:
        avg_wait = max(2, 12 - recommended_trains)
        load_pct = min( 100, int((predicted_demand / (recommended_trains * TRAIN_CAPACITY)) * 100))
        energy = max(60, 100 - recommended_trains * 4)
        comfort = max(50, 100 - load_pct)

        st.metric("⏱ Waiting Time", f"{avg_wait} min")
        st.metric("👥 Load", f"{load_pct}%")
        st.metric("⚡ Energy", f"{energy}%")
        st.metric("🙂 Comfort", f"{comfort}%")
    else:
        st.metric("⏱ Waiting Time", "-")
        st.metric("👥 Load", "-")
        st.metric("⚡ Energy", "-")
        st.metric("🙂 Comfort", "-")

with center:
    st.subheader("🤖 AI Recommendation")

    if run_ai:
        st.markdown(f"""
### 🚆 Recommended Trains: **{recommended_trains}**
**Predicted Demand:** {predicted_demand} passengers  
**Direction:** {"Aluva → Ernakulam" if direction_id == 0 else "Ernakulam → Aluva"}

### 🌦️ Weather Impact on Demand
- **Condition:** {weather['condition']}
- **Rainfall:** {weather['rain_mm']} mm
- **Temperature:** {weather['temp']} °C
- **Demand Multiplier:** {round(weather_factor, 2)}×

---
""")

        st.info(generate_llm_explanation(
            predicted_demand,
            recommended_trains,
            weather,
            is_peak
        ))
    else:
        st.info("Run AI to generate recommendation")

st.subheader("📈 Passenger Demand Trend")

history_hours = list(range(24))
historical = np.random.randint(2000, 9000, size=24)
predicted = historical.copy()
predicted[hour] = predicted_demand

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history_hours, historical, label="Historical Demand")
ax.plot(history_hours, predicted, label="AI Predicted", linestyle="--")
ax.axvline(hour, linestyle=":", label="Selected Hour")
ax.set_xlabel("Hour")
ax.set_ylabel("Passengers")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

st.caption("🚀 Weather-Driven Smart Metro Operations")