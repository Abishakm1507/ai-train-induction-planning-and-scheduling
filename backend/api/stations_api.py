from fastapi import APIRouter

# -------------------------------------------------
# Stations metadata
# -------------------------------------------------
STATIONS = [
     "Aluva",
    "Pulinchodu",
    "Companypady",
    "Ambattukavu",
    "Muttom",
    "Kalamassery",
    "Pathadipalam",
    "Cochin University",
    "Edapally",
    "Changampuzha Park",
    "Palarivattom",
    "JLN Stadium",
    "Kaloor",
    "Town Hall",
    "MG Road",
    "Maharaja’s College",
    "Ernakulam South"
]

# -------------------------------------------------
# FastAPI Router
# -------------------------------------------------
router = APIRouter()

@router.get("/list")
def get_stations():
    """
    GET /api/stations/list
    Returns list of all metro stations
    """
    return {
        "stations": STATIONS,
        "count": len(STATIONS)
    }

@router.get("/{station_id}")
def get_station(station_id: int):
    """
    GET /api/stations/{station_id}
    Returns details of a specific station
    """
    if station_id < 0 or station_id >= len(STATIONS):
        return {"error": "Station not found"}
    
    return {
        "id": station_id,
        "name": STATIONS[station_id]
    }
