# Frontend Integration Guide
**Status:** ✅ COMPLETED - All Backend APIs Connected to Frontend

---

## Overview
The frontend `index.html` dashboard has been fully integrated with the backend FastAPI services. The dashboard now communicates with:
- **Demand Forecasting API** (`demand_api.py`)
- **Train Induction API** (`induction_api.py`)
- **Surge Detection** (`surge_detection.py`)
- **Weather Integration** (via external WeatherAPI)

---

## Architecture

### Frontend Components Connected

#### 1. **Operations Control Tab (Tab 1)** ✅
**File:** [Frontend/index.html](Frontend/index.html)

**Backend Integration:**
- `runAI()` function calls:
  - `fetchDemandForecast()` → `/api/demand/forecast` (POST)
  - `fetchTrainRecommendation()` → `/api/induction/recommend` (POST)
  - `detectSurge()` → Local surge detection logic
  - `getWeatherData()` → External WeatherAPI

**Output Display:**
- Recommended trains with confidence level
- Policy used (Reinforcement Learning vs Rule-Based)
- Predicted passenger demand
- Weather impact analysis
- Surge alerts (if demand threshold exceeded)
- KPI metrics (waiting time, load, energy, comfort)
- Demand trend chart

#### 2. **Demand Analytics Tab (Tab 2)** ✅

**Backend Integration:**
- `runDemandAnalytics()` function:
  - Calls `fetchDemandForecast()` for each hour (6 AM - 10 PM)
  - Generates station-wise demand heatmap
  - Color-coded intensity (Green < Yellow < Red)

**Output Display:**
- Interactive heatmap showing passenger demand by station and hour
- Real-time demand data from the ML model

#### 3. **Train Planning Tab (Tab 3)** ✅

**Backend Integration:**
- `generatePlan()` function:
  - Calls `fetchDemandForecast()` and `fetchTrainRecommendation()` for hours 6-22
  - Supports peak hour mode toggle
  - Respects available train constraints

**Output Display:**
- Tabular scheduling plan with:
  - Hour
  - Predicted passenger demand
  - Recommended trains per hour
  - Headway (minutes between trains)

#### 4. **Scenario Simulator Tab (Tab 4)** ✅

**Backend Integration:**
- `runSimulation()` function:
  - Base demand at peak hour (9:00 AM)
  - Applies demand multipliers:
    - User demand increase: 0-100%
    - Rain: +20%
    - Festival: +35%
  - Unavailable trains constraint
  - Calls `fetchTrainRecommendation()` for adjusted demand

**Output Display:**
- Scenario feasibility (Manageable / Capacity Exceeded)
- Comparison: Base vs Scenario demand
- Train configuration impact
- Impact metrics:
  - Average waiting time
  - Crowding level
  - Surge status

---

## API Endpoints Used

### 1. Demand Forecasting API
**Endpoint:** `POST /api/demand/forecast`

**Request Body:**
```json
{
  "hour": 9,
  "is_weekend": 0,
  "is_peak_hour": 1,
  "trains_per_hour": 5,
  "direction_id": 0
}
```

**Response:**
```json
{
  "estimated_demand": 5234,
  "weather_factor": 1.1,
  "station": "Aluva"
}
```

**Frontend Usage:**
```javascript
const demandResult = await fetchDemandForecast(hour, isWeekend, isPeak, trainsPerHour, direction);
```

---

### 2. Train Induction API (RL-Based)
**Endpoint:** `POST /api/induction/recommend`

**Request Body:**
```json
{
  "predicted_demand": 5234,
  "is_peak_hour": 1
}
```

**Response:**
```json
{
  "recommended_trains": 6,
  "confidence": 92,
  "policy": "reinforcement-learning"
}
```

**Frontend Usage:**
```javascript
const inductionResult = await fetchTrainRecommendation(predictedDemand, isPeak);
```

---

### 3. Surge Detection (Local)
**Function:** `detectSurge(predictedDemand, threshold = 7000)`

**Logic:**
- **High:** Demand ≥ 7000 passengers
- **Medium:** Demand ≥ 5600 passengers (80% of threshold)
- **Low:** Demand < 5600 passengers

**Frontend Usage:**
```javascript
const surgeStatus = detectSurge(predictedDemand);
if (surgeStatus.surge) {
  // Show warning alert
}
```

---

## Configuration

### API Base URL
```javascript
const API_BASE = "http://localhost:8000/api";
```

**Update location:** [Frontend/index.html - Line ~350](Frontend/index.html#L350)

To change backend server:
1. Modify `API_BASE` in the JavaScript section
2. Ensure backend FastAPI is running on the specified URL

---

## Error Handling

All API calls include try-catch blocks with user-friendly error messages:

```javascript
catch (error) {
  console.error("API Error:", error);
  document.getElementById("aiRecommendation").innerHTML = `
    <div class="error-box">
      ❌ Error: Backend service unavailable. 
      Please ensure FastAPI server is running at ${API_BASE}
    </div>
  `;
}
```

---

## Key Functions Reference

### 1. `runAI()` - Operations Control Flow
**What it does:**
- Fetches real-time weather data
- Calls demand forecast API
- Calls train induction API
- Detects surges
- Updates dashboard with recommendations

**Triggered by:** "Run AI" button in Operations Control tab

---

### 2. `runDemandAnalytics()` - Analytics Heatmap
**What it does:**
- Iterates through 6 AM - 10 PM
- Fetches demand for each hour
- Generates color-coded heatmap
- Displays station congestion patterns

**Triggered by:** "Run" button in Demand Analytics tab

---

### 3. `generatePlan()` - Train Scheduling
**What it does:**
- Calls APIs for each hour (6 AM - 10 PM)
- Respects peak hour mode toggle
- Respects available train constraints
- Calculates headway for each slot

**Triggered by:** "Generate AI Plan" button in Train Planning tab

---

### 4. `runSimulation()` - Scenario Testing
**What it does:**
- Applies demand multipliers (% increase, rain, festival)
- Accounts for unavailable trains
- Determines scenario feasibility
- Calculates impact metrics

**Triggered by:** "Run Simulation" button in Scenario Simulator tab

---

## Testing Checklist

- [ ] Backend FastAPI server is running (`python -m uvicorn backend.app:app --reload`)
- [ ] Frontend can access `http://localhost:8000/api/`
- [ ] CORS is enabled on backend (✅ Already configured in [backend/app.py](backend/app.py))
- [ ] Weather API key is set in `.env` file (optional; falls back to defaults)
- [ ] All model files exist:
  - `model/demand_forecast_model.pkl` ✅
  - `model/model_features.pkl` ✅
  - `model/rl_q_table.pkl` (optional for RL policy)

---

## Running the Dashboard

### 1. Start Backend
```bash
cd backend
python -m uvicorn app:app --reload --port 8000
```

### 2. Open Frontend
```bash
# Option 1: Open file directly in browser
open Frontend/index.html

# Option 2: Serve via Python
cd Frontend
python -m http.server 8080
# Then visit http://localhost:8080/index.html
```

### 3. Interact with Dashboard
- Switch between tabs using navigation buttons
- Adjust sliders and dropdowns
- Click action buttons to trigger API calls
- Monitor loading indicators and error messages

---

## Response Display Examples

### Operations Control - AI Recommendation
```
✓ AI Decision Generated
Recommended Trains/Hour: 6
Confidence Level: 92%
Policy Used: 🤖 Reinforcement Learning

Predicted Passenger Demand: 5234 passengers
Direction: Aluva → Ernakulam

Weather Impact:
• Condition: Clear
• Rainfall: 0mm
• Temperature: 28°C
• Demand Multiplier: 1.0×

Headway (minutes): 10
Peak Hour: Yes
```

### Demand Analytics - Heatmap
- Interactive grid showing demand intensity per station per hour
- Color intensity: 🟢 Green (Low) → 🟡 Yellow (Medium) → 🔴 Red (High)

### Train Planning - Schedule Table
| Hour | Passenger Demand | Recommended Trains | Headway (min) |
|------|------------------|--------------------|---------------|
| 6:00 | 2500             | 2                  | 30            |
| 9:00 | 7200             | 8                  | 7             |
| 17:00| 6800             | 8                  | 7             |

### Scenario Simulator - Impact Report
- Scenario feasibility (✓ Manageable / ❌ Capacity Exceeded)
- Demand comparison (base vs scenario with multipliers)
- Train configuration impact
- Waiting time, crowding level, surge status

---

## Customization

### Change Demand Threshold for Surge Detection
Edit [Frontend/index.html](Frontend/index.html) - `detectSurge()` function:
```javascript
function detectSurge(predictedDemand, threshold = 7000) {  // Change 7000 here
```

### Change Peak Hour Definition
Edit [Frontend/index.html](Frontend/index.html):
```javascript
const isPeak = (hour >= 8 && hour <= 10) || (hour >= 17 && hour <= 20);  // Modify ranges
```

### Change Station List
Edit [Frontend/index.html](Frontend/index.html) - `runDemandAnalytics()` function:
```javascript
const stations = [
  "Aluva", "Pulinchodu", ...  // Add/remove stations
];
```

---

## Known Limitations

1. **Weather API**: Falls back to defaults (28°C, 0mm rain, "Clear") if external API unavailable
2. **RL Model**: Uses rule-based fallback if Q-table not found
3. **Heatmap**: Currently shows 12 stations × variable hours
4. **Time**: Displays forecasts up to 10 PM (22:00)

---

## Support

For issues:
1. Check browser console for JavaScript errors: `F12 → Console tab`
2. Verify backend is running: `http://localhost:8000/`
3. Check network requests: `F12 → Network tab`
4. Review error messages in dashboard (red boxes)

---

**Last Updated:** January 23, 2026
**Frontend File:** [Frontend/index.html](Frontend/index.html)
**Backend Directory:** [backend/](backend/)
