import os
import joblib
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

# -------------------------------------------------
# Path configuration
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# -------------------------------------------------
# Load RL Q-table
# -------------------------------------------------
try:
    q_table = joblib.load(os.path.join(MODEL_DIR, "rl_q_table.pkl"))
    rl_ready = True
except Exception:
    q_table = {}
    rl_ready = False

# -------------------------------------------------
# Constants
# -------------------------------------------------
MIN_TRAINS = 2
MAX_TRAINS = 10

# -------------------------------------------------
# Pydantic models
# -------------------------------------------------
class InductionRequest(BaseModel):
    predicted_demand: int
    is_peak_hour: int

class InductionResponse(BaseModel):
    recommended_trains: int
    confidence: int
    policy: str

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def get_demand_level(demand: int) -> int:
    """
    Discretize continuous demand into RL states
    """
    if demand < 3000:
        return 0
    elif demand < 6000:
        return 1
    else:
        return 2

def fallback_policy(demand_level: int, is_peak: int) -> int:
    """
    Rule-based fallback if RL model is unavailable
    """
    if is_peak and demand_level == 2:
        return MAX_TRAINS
    if demand_level == 2:
        return MAX_TRAINS - 1
    if demand_level == 1:
        return (MIN_TRAINS + MAX_TRAINS) // 2
    return MIN_TRAINS

# -------------------------------------------------
# FastAPI Router
# -------------------------------------------------
router = APIRouter()

@router.post("/recommend", response_model=InductionResponse)
def recommend_trains(data: InductionRequest):
    """
    POST /api/induction/recommend

    RL State  : (demand_level, is_peak_hour)
    RL Action : number of trains to deploy
    """

    demand_level = get_demand_level(data.predicted_demand)
    state = (demand_level, data.is_peak_hour)

    actions = list(range(MIN_TRAINS, MAX_TRAINS + 1))

    if rl_ready:
        q_values = [q_table.get((state, action), 0.0) for action in actions]
        best_action = actions[int(np.argmax(q_values))]
        confidence = 92
        policy = "reinforcement-learning"
    else:
        best_action = fallback_policy(demand_level, data.is_peak_hour)
        confidence = 78
        policy = "rule-based-fallback"

    return {
        "recommended_trains": best_action,
        "confidence": confidence,
        "policy": policy
    }
