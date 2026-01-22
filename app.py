import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("./model/demand_forecast_model.pkl")
features = joblib.load("./model/model_features.pkl")

TRAIN_CAPACITY = 1000
SAFETY_BUFFER = 0.20
MIN_TRAINS = 2
MAX_TRAINS = 10

def induction_decision(predicted_demand, is_peak, is_weekend):

    base_trains = np.ceil(predicted_demand / TRAIN_CAPACITY)
    
    required_trains = base_trains + (1 if is_peak else 0)
    
    if is_weekend:
        required_trains = max(1, required_trains - 1)

    return int(min(max(required_trains, MIN_TRAINS), MAX_TRAINS))

st.set_page_config(page_title="KMRL AI Train Induction", layout="centered")

st.title("🚆 AI-Driven Train Induction System")
st.subheader("Kochi Metro Rail Limited (KMRL)")

st.markdown("Enter operational inputs to predict passenger demand and train requirements.")

hour = st.slider("Hour of Day", 0, 23, 9)

day_type = st.selectbox(
    "Day Type",
    ["Weekday", "Weekend"]
)

is_weekend = 1 if day_type == "Weekend" else 0

is_peak_hour = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0

trains_per_hour = st.slider("Current Trains per Hour", 1, 10, 5)

direction_id = st.selectbox("Direction", [0, 1])

if st.button("🔮 Predict Demand & Induction"):
    
    input_df = pd.DataFrame([{
        'hour': hour,
        'is_weekend': is_weekend,
        'is_peak_hour': is_peak_hour,
        'trains_per_hour': trains_per_hour,
        'direction_id': direction_id
    }])

    input_df = input_df[features]

    predicted_demand = model.predict(input_df)[0]

    recommended_trains = induction_decision(
        predicted_demand,
        is_peak_hour,
        is_weekend
    )

    st.success("Prediction Complete ✅")

    st.metric(
        label="📊 Predicted Passenger Demand",
        value=int(predicted_demand)
    )

    st.metric(
        label="🚆 Recommended Trains",
        value=recommended_trains
    )

    if recommended_trains >= 7:
        st.warning("⚠️ Increase train frequency")
    elif recommended_trains >= 4:
        st.info("ℹ️ Maintain current service")
    else:
        st.error("⬇️ Reduce train frequency")
