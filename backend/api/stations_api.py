from fastapi import APIRouter
from datetime import datetime
import random

router = APIRouter()

# -------------------------------------------------
# Static station list (can be DB-driven later)
# -------------------------------------------------
STATIONS = [
    "Aluva", "Pulinchodu", "Companypady", "Ambattukavu",
    "Muttom", "Kalamassery", "CUSAT", "Edappally",
    "Kaloor", "MG Road", "Maharaja’s", "Ernakulam South"
]

# -------------------------------------------------
# Station status generator (mock logic)
# -------------------------------------------------
@router.get("/status")
def get_station_status():
    # return your status data
    statuses = []

    for station in STATIONS:
        status = random.choices(
            ["Operational", "Minor Delay", "Crowded"],
            weights=[0.75, 0.15, 0.10]
        )[0]

        statuses.append({
            "station": station,
            "status": status,
            "last_updated": datetime.now().strftime("%H:%M")
        })

    return statuses

# -------------------------------------------------
# API Endpoint
# -------------------------------------------------
@router.get("/status")
def station_status():
    """
    GET /api/stations/status
    """
    return {
        "line": "Kochi Metro",
        "direction": "Both",
        "stations": get_station_status()
    }
