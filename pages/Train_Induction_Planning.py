import streamlit as st
import numpy as np
import joblib
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Train Induction Planning",
    layout="wide"
)

st.title("🚆 AI-Assisted Train Induction Planning")
st.caption("Decision-support system for metro scheduling & headway optimization")

# -------------------------------
# Load RL Q-table (if available)
# -------------------------------
Q_TABLE_PATH = "./model/rl_q_table.pkl"

if os.path.exists(Q_TABLE_PATH):
    q_table = joblib.load(Q_TABLE_PATH)
    rl_available = True
else:
    q_table = {}
    rl_available = False

# -------------------------------
# UI Layout
# -------------------------------
left, right = st.columns([2, 3])

with left:
    st.subheader("⚙️ Planning Inputs")

    with st.container(border=True):
        available_trains = st.slider(
            "Available Trains",
            min_value=2,
            max_value=20,
            value=10
        )

        peak_mode = st.toggle("Peak Hour Mode", value=True)

        generate = st.button("🤖 Generate AI Plan", use_container_width=True)

# -------------------------------
# RL Decision Logic
# -------------------------------
def rl_train_planner(predicted_demand, peak, available_trains):
    """
    RL-based train induction planner
    Uses trained Q-table
    """

    # Discretize demand
    if predicted_demand < 2000:
        demand_level = 0
    elif predicted_demand < 5000:
        demand_level = 1
    else:
        demand_level = 2

    state = (demand_level, peak, available_trains)

    # Allowed actions
    actions = list(range(2, available_trains + 1))

    # Select best action from Q-table
    q_values = [q_table.get((state, a), 0) for a in actions]
    deploy = actions[np.argmax(q_values)]

    # Estimate headway (simple operational model)
    headway = round(max(2.0, 12 - deploy), 1)

    expected_wait = round(headway / 2, 1)

    # Overcrowding risk
    if peak and demand_level == 2 and deploy < available_trains * 0.7:
        risk = "High"
    elif peak:
        risk = "Medium"
    else:
        risk = "Low"

    return deploy, headway, expected_wait, risk

# -------------------------------
# Output Panel
# -------------------------------
with right:
    st.subheader("📊 AI Output Preview")

    if generate:

        # Simulated demand (replace with ML model later)
        predicted_demand = np.random.randint(1500, 6500)

        deploy, headway, wait_time, risk = rl_train_planner(
            predicted_demand,
            peak_mode,
            available_trains
        )

        with st.container(border=True):
            st.markdown("### 🤖 AI Deployment Recommendation")

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            col1.metric("🚆 Trains to Deploy", deploy)
            col2.metric("⏱ Headway (min)", f"{headway}")
            col3.metric("⌛ Expected Waiting Time", f"{wait_time} min")

            if risk == "Low":
                col4.success("🟢 Overcrowding Risk: Low")
            elif risk == "Medium":
                col4.warning("🟡 Overcrowding Risk: Medium")
            else:
                col4.error("🔴 Overcrowding Risk: High")

            st.markdown("---")

            st.info(
                f"""
                **AI Insight:**  
                A reinforcement learning agent evaluated multiple deployment actions
                and selected **{deploy} trains** to balance passenger waiting time,
                operational efficiency, and congestion risk under
                {'peak-hour' if peak_mode else 'off-peak'} conditions.
                """
            )

            if not rl_available:
                st.warning(
                    "⚠️ RL model not trained yet. "
                    "Using default Q-values. Train RL agent for optimal performance."
                )

    else:
        st.info("Adjust inputs and click **Generate AI Plan** to view recommendations")
