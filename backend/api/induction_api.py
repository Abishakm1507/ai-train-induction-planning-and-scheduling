import os
import joblib
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List, Optional
import google.generativeai as genai

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
# Gemini Configuration
# -------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCdVsXfVyuBpOVserT_qnE3tt7CUy3CuX0")
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------------------------------
# Constants
# -------------------------------------------------
MIN_TRAINS = 2
MAX_TRAINS = 20

# -------------------------------------------------
# Pydantic models
# -------------------------------------------------
class InductionRequest(BaseModel):
    predicted_demand: int
    is_peak_hour: int
    available_trains: int = 10

class InductionResponse(BaseModel):
    recommended_trains: int
    confidence: int
    policy: str
    headway: float
    expected_waiting_time: float
    overcrowding_risk: str
    q_values: Optional[Dict[int, float]] = None
    explanation: str

class InductionDetailedResponse(BaseModel):
    recommended_trains: int
    confidence: int
    policy: str
    headway: float
    expected_waiting_time: float
    overcrowding_risk: str
    demand_level: int
    state: tuple
    q_values: Dict[int, float]
    all_actions: List[int]
    rl_model_loaded: bool
    explanation: str

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

def calculate_headway(trains_deployed: int) -> float:
    """
    Calculate headway (time between trains) in minutes
    Headway = 60 minutes / number of trains deployed
    """
    if trains_deployed == 0:
        return float('inf')
    return round(60.0 / trains_deployed, 1)

def calculate_waiting_time(headway: float) -> float:
    """
    Expected waiting time = headway / 2 (passengers arrive randomly)
    """
    return round(headway / 2, 1)

def assess_overcrowding_risk(demand_level: int, is_peak: int, trains_deployed: int) -> str:
    """
    Assess overcrowding risk based on demand and deployment
    """
    demand_threshold = trains_deployed * 800  # ~800 passengers per train
    
    if is_peak:
        if demand_level == 2:  # High demand
            if trains_deployed < 6:
                return "High"
            elif trains_deployed < 8:
                return "Medium"
            else:
                return "Low"
        elif demand_level == 1:  # Medium demand
            return "Medium" if trains_deployed < 5 else "Low"
        else:  # Low demand
            return "Low"
    else:
        if demand_level == 2:
            return "Medium"
        elif demand_level == 1:
            return "Low"
        else:
            return "Low"

def generate_llm_explanation(demand: int, trains: int, is_peak: int, policy: str):
    """
    Generate natural language explanation using Gemini
    """
    prompt = f"""
Explain the metro operational decision.

Context:
- Predicted passenger demand: {demand}
- Trains recommended: {trains}
- Peak hour: {"Yes" if is_peak else "No"}
- Policy used: {policy}

Rules:
- Simple professional language
- Max 2–3 sentences
- Do not mention AI or "the model" or specific algorithm names like Q-learning
- Focus on the balance between service availability and passenger load.
"""
    try:
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return f"Based on the current demand of {demand} passengers, we recommend deploying {trains} trains to maintain optimal headways during {'peak' if is_peak else 'off-peak'} hours."

def generate_explanation(demand_level: int, is_peak: int, trains_deployed: int, 
                        headway: float, policy: str) -> str:
    """
    Generate human-readable explanation of recommendation
    """
    demand_text = ["low", "medium", "high"][demand_level]
    peak_text = "peak-hour" if is_peak else "off-peak"
    
    explanation = (
        f"Based on {demand_text} demand levels during {peak_text} conditions, "
        f"the {policy} model recommends deploying {trains_deployed} trains. "
        f"This provides a {headway}-minute headway, ensuring efficient service."
    )
    return explanation

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
    
    Returns detailed recommendation with operational metrics
    """
    demand_level = get_demand_level(data.predicted_demand)
    # The Q-table was trained with numpy types in some fields
    # State format in Q-table: (demand_level, np.int64(peak), available_trains)
    # Action format in Q-table: np.int64(action)
    
    actions = list(range(MIN_TRAINS, 21))
    
    if rl_ready:
        # Try different type combinations to match the Q-table keys
        # Based on inspection: ((int, np.int64, int), np.int64)
        q_values = {}
        for a in actions:
            # Construct the exact key used during training
            key = ((demand_level, np.int64(data.is_peak_hour), data.available_trains), np.int64(a))
            # Use a very low fallback for negative Q-values
            q_values[a] = q_table.get(key, -9999.0)

        # If all lookups failed (-9999), use fallback policy
        if all(v == -9999.0 for v in q_values.values()):
            best_action = fallback_policy(demand_level, data.is_peak_hour)
            confidence = 70
            policy = "rule-based-fallback (no Q-match)"
        else:
            best_action = actions[int(np.argmax([q_values[a] for a in actions]))]
            confidence = 92
            policy = "reinforcement-learning"
    else:
        best_action = fallback_policy(demand_level, data.is_peak_hour)
        confidence = 78
        policy = "rule-based-fallback"
        q_values = {action: 0.0 for action in actions}

    # Calculate operational metrics
    headway = calculate_headway(best_action)
    waiting_time = calculate_waiting_time(headway)
    risk = assess_overcrowding_risk(demand_level, data.is_peak_hour, best_action)
    
    # Generate Gemini-powered explanation
    explanation = generate_llm_explanation(
        data.predicted_demand, 
        best_action, 
        data.is_peak_hour, 
        policy
    )

    return {
        "recommended_trains": best_action,
        "confidence": confidence,
        "policy": policy,
        "headway": headway,
        "expected_waiting_time": waiting_time,
        "overcrowding_risk": risk,
        "q_values": q_values,
        "explanation": explanation
    }


@router.post("/detailed", response_model=InductionDetailedResponse)
def recommend_trains_detailed(data: InductionRequest):
    """
    POST /api/induction/detailed

    Extended endpoint providing full Q-table analysis and debug information
    Useful for monitoring and understanding RL model decisions
    """
    demand_level = get_demand_level(data.predicted_demand)
    actions = list(range(MIN_TRAINS, 21))

    if rl_ready:
        q_values = {}
        for a in actions:
            key = ((demand_level, np.int64(data.is_peak_hour), data.available_trains), np.int64(a))
            q_values[a] = float(q_table.get(key, -9999.0))
            
        if all(v == -9999.0 for v in q_values.values()):
            best_action = fallback_policy(demand_level, data.is_peak_hour)
            confidence = 70
            policy = "rule-based-fallback (no Q-match)"
        else:
            best_action = actions[int(np.argmax([q_values[a] for a in actions]))]
            confidence = 92
            policy = "reinforcement-learning"
    else:
        best_action = fallback_policy(demand_level, data.is_peak_hour)
        confidence = 78
        policy = "rule-based-fallback"
        q_values = {action: 0.0 for action in actions}

    # Calculate operational metrics
    headway = calculate_headway(best_action)
    waiting_time = calculate_waiting_time(headway)
    risk = assess_overcrowding_risk(demand_level, data.is_peak_hour, best_action)
    
    # Generate Gemini-powered explanation
    explanation = generate_llm_explanation(
        data.predicted_demand, 
        best_action, 
        data.is_peak_hour, 
        policy
    )

    return {
        "recommended_trains": best_action,
        "confidence": confidence,
        "policy": policy,
        "headway": headway,
        "expected_waiting_time": waiting_time,
        "overcrowding_risk": risk,
        "demand_level": demand_level,
        "state": state,
        "q_values": q_values,
        "all_actions": actions,
        "rl_model_loaded": rl_ready,
        "explanation": explanation
    }


@router.get("/status")
def induction_system_status():
    """
    GET /api/induction/status
    
    Returns health status of the RL induction planning system
    """
    return {
        "status": "operational",
        "rl_model_loaded": rl_ready,
        "min_trains": MIN_TRAINS,
        "max_trains": MAX_TRAINS,
        "q_table_size": len(q_table) if rl_ready else 0,
        "demand_levels": 3,  # Low, Medium, High
        "policies": ["reinforcement-learning", "rule-based-fallback"]
    }
