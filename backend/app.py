from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import sys

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# -------------------------------------------------
# Load environment variables (.env)
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Import routers
# -------------------------------------------------
from api.stations_api import router as stations_router
from api.demand_api import router as demand_router
from api.induction_api import router as induction_router

# -------------------------------------------------
# FastAPI app configuration
# -------------------------------------------------
app = FastAPI(
    title="KMRL AI Backend",
    description="Passenger Demand Forecasting and RL-based Train Induction System",
    version="1.0.0"
)

app.include_router(
    stations_router,
    prefix="/api/stations",
    tags=["Stations"]
)


# -------------------------------------------------
# CORS configuration (frontend access)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Register API routers
# -------------------------------------------------
app.include_router(
    demand_router,
    prefix="/api/demand",
    tags=["Demand Forecasting"]
)

app.include_router(
    induction_router,
    prefix="/api/induction",
    tags=["Train Induction"]
)

# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "KMRL AI Backend",
        "version": "1.0.0"
    }
