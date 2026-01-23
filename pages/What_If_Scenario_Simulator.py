import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(
    page_title="What-If Scenario Simulator",
    layout="wide"
)

st.title("🔮 What-If Scenario Simulator")
st.caption("Strategic AI simulation for metro operational planning")

# =====================================================
# CONSTANTS (same logic as demand forecasting)
# =====================================================

STATIONS = [
    "Aluva", "Pulinchodu", "Companypady", "Ambattukavu",
    "Muttom", "Kalamassery", "CUSAT", "Edappally",
    "Kaloor", "MG Road", "Maharaja’s", "Ernakulam South"
]

HOURS = list(range(6, 23))  # peak operational hours

BASE_TRAINS = 12
TRAIN_CAPACITY = 1000
BASE_HEADWAY = 4  # minutes

# =====================================================
# DEMAND PREDICTION (REUSED LOGIC)
# =====================================================

def predict_demand(selected_date):
    """
    Predict demand based on the same logic used
    in the demand forecasting dashboard
    """

    # Make prediction deterministic per date
    np.random.seed(int(selected_date.strftime("%Y%m%d")))

    is_weekend = 1 if selected_date.weekday() >= 5 else 0
    total_demand = 0

    for _ in STATIONS:
        base = np.random.randint(200, 600)

        for hour in HOURS:
            peak_factor = 1.6 if (8 <= hour <= 10 or 17 <= hour <= 20) else 1.0
            weekend_factor = 0.8 if is_weekend else 1.0

            demand = base * peak_factor * weekend_factor * np.random.uniform(0.85, 1.15)
            total_demand += demand

    return int(total_demand)

# =====================================================
# LAYOUT
# =====================================================

left, right = st.columns([2, 3])

# =====================================================
# SCENARIO INPUTS (DATE ADDED HERE ✅)
# =====================================================

with left:
    st.subheader("⚙️ Scenario Inputs")

    with st.container(border=True):

        selected_date = st.date_input(
            "📅 Select Scenario Date",
            value=date.today()
        )

        # Auto-predict demand from date
        BASE_DEMAND = predict_demand(selected_date)

        demand_increase = st.slider(
            "Passenger Demand Increase (%)",
            0, 100, 20
        )

        unavailable_trains = st.slider(
            "Trains Unavailable",
            0, 10, 2
        )

        st.markdown("### 🌦️ Special Events")
        rain = st.toggle("Rain")
        festival = st.toggle("Festival / City Event")

        run_simulation = st.button(
            "🔁 Run Simulation",
            use_container_width=True
        )

# =====================================================
# SHOW AUTO-PREDICTED DEMAND
# =====================================================

with right:
    st.info(
        f"""
        📅 **Scenario Date:** {selected_date}  
        👥 **AI-Predicted Base Demand:** {BASE_DEMAND}
        """
    )

# =====================================================
# SIMULATION LOGIC
# =====================================================

def simulate_scenario(demand_inc, trains_down, rain, festival):
    demand = BASE_DEMAND * (1 + demand_inc / 100)

    if rain:
        demand *= 1.10
    if festival:
        demand *= 1.25

    available_trains = max(2, BASE_TRAINS - trains_down)
    load_factor = demand / (available_trains * TRAIN_CAPACITY)

    waiting_time = round(BASE_HEADWAY * load_factor, 1)
    energy_use = round(available_trains * 1.2, 1)

    if load_factor > 1.1:
        risk = "High"
    elif load_factor > 0.85:
        risk = "Medium"
    else:
        risk = "Low"

    return demand, available_trains, waiting_time, energy_use, risk

# =====================================================
# RESULTS
# =====================================================

with right:
    st.subheader("📊 Simulation Results")

    if run_simulation:
        sim_demand, sim_trains, sim_wait, sim_energy, sim_risk = simulate_scenario(
            demand_increase,
            unavailable_trains,
            rain,
            festival
        )

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "👥 Passenger Demand",
                f"{int(sim_demand)}",
                f"+{int(sim_demand - BASE_DEMAND)}"
            )

            c2.metric(
                "🚆 Trains Available",
                sim_trains,
                f"-{unavailable_trains}"
            )

            c3.metric(
                "⏱ Avg Waiting Time",
                f"{sim_wait} min",
                f"+{round(sim_wait - BASE_HEADWAY, 1)}"
            )

            if sim_risk == "Low":
                c4.success("🟢 Overcrowding: Low")
            elif sim_risk == "Medium":
                c4.warning("🟡 Overcrowding: Medium")
            else:
                c4.error("🔴 Overcrowding: High")

        st.markdown("### 📈 Before vs After Comparison")

        labels = ["Demand", "Waiting Time", "Energy Usage"]
        before = [BASE_DEMAND, BASE_HEADWAY, BASE_TRAINS * 1.2]
        after = [sim_demand, sim_wait, sim_energy]

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labels))

        ax.bar(x - 0.2, before, width=0.4, label="Normal")
        ax.bar(x + 0.2, after, width=0.4, label="Scenario")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    else:
        st.info("Adjust scenario parameters and click **Run Simulation**")

# =====================================================
# UI POLISH
# =====================================================

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
