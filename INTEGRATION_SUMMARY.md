# Backend-Frontend Integration Summary

## ✅ Completion Status: COMPLETE

All backend functionalities have been successfully connected to the frontend dashboard.

---

## Connected Components

### 1. **Demand Forecasting API** ✅
**File:** [backend/api/demand_api.py](backend/api/demand_api.py)

**Endpoint:** `POST /api/demand/predict`

**Request Fields:**
- `hour` (int): Hour of day (0-23)
- `is_weekend` (int): 0=weekday, 1=weekend
- `is_peak_hour` (int): 0=off-peak, 1=peak
- `trains_per_hour` (int): Current train frequency
- `direction_id` (int): 0=direction A, 1=direction B

**Returns:**
```json
{
  "predicted_demand": 5234,
  "weather": {
    "temp": 28.5,
    "rain_mm": 0.5,
    "condition": "Partly cloudy"
  },
  "explanation": "Passenger demand is influenced by peak-hour travel patterns..."
}
```

**Frontend Integration:**
- ✅ `fetchDemandForecast()` function [Line ~500]
- ✅ Used in: Operations Control, Demand Analytics, Train Planning, Scenario Simulator

---

### 2. **Train Induction API (RL-Based)** ✅
**File:** [backend/api/induction_api.py](backend/api/induction_api.py)

**Endpoint:** `POST /api/induction/recommend`

**Request Fields:**
- `predicted_demand` (int): Forecasted passenger count
- `is_peak_hour` (int): 0=off-peak, 1=peak

**Returns:**
```json
{
  "recommended_trains": 6,
  "confidence": 92,
  "policy": "reinforcement-learning"
}
```

**Fallback Policy:** Rule-based if RL model unavailable

**Frontend Integration:**
- ✅ `fetchTrainRecommendation()` function [Line ~520]
- ✅ Used in: Operations Control, Train Planning, Scenario Simulator

---

### 3. **Surge Detection** ✅
**File:** [backend/utils/surge_detection.py](backend/utils/surge_detection.py)

**Logic (Local Implementation):**
```javascript
function detectSurge(predictedDemand, threshold = 7000) {
  if (predictedDemand >= 7000) return { surge: true, severity: "High" };
  if (predictedDemand >= 5600) return { surge: true, severity: "Medium" };
  return { surge: false, severity: "Low" };
}
```

**Frontend Integration:**
- ✅ `detectSurge()` function [Line ~570]
- ✅ Displays alerts in Operations Control tab when active

---

### 4. **Weather Integration** ✅
**External API:** WeatherAPI.com

**Function:** `getWeatherData(city = "Kochi")`

**Returns:**
```json
{
  "temp": 28,
  "rain_mm": 0,
  "condition": "Clear"
}
```

**Fallback:** Returns defaults if external API unavailable

**Demand Multiplier:**
- Rain > 5mm: +20% demand
- Heavy weather: +10% demand
- Temperature > 34°C: -5% demand

---

## Frontend Tabs & Integrations

### Tab 1: Operations Control 🚆
**Purpose:** Real-time AI decision making

**APIs Called:**
1. Weather API (external) → Current conditions
2. Demand API → `/api/demand/predict` → Passenger forecast
3. Induction API → `/api/induction/recommend` → Train recommendation
4. Surge Detection → Local function → Alert if threshold exceeded

**User Inputs:**
- Hour (0-23)
- Day Type (weekday/weekend)
- Current Trains/Hour (1-10)
- Direction (0=A→B, 1=B→A)

**Outputs Displayed:**
- ✅ Recommended trains (with confidence %)
- ✅ RL policy or rule-based fallback
- ✅ Predicted demand
- ✅ Weather conditions & multiplier
- ✅ Headway calculation
- ✅ Surge alert (if applicable)
- ✅ KPI metrics (waiting time, load, energy, comfort)
- ✅ Demand trend chart

**Function:** `runAI()` [Line ~700]

---

### Tab 2: Demand Analytics 📊
**Purpose:** Historical demand patterns & station congestion

**APIs Called:**
1. Demand API → `/api/demand/predict` → Called for each hour (6 AM - 10 PM)

**User Inputs:**
- Date selector
- Time range (start/end hour)
- Day type (weekday/weekend)
- Compare previous week (toggle)

**Outputs Displayed:**
- ✅ Station-wise demand heatmap
- ✅ Color-coded intensity (🟢 Green = Low, 🟡 Yellow = Medium, 🔴 Red = High)
- ✅ Hourly demand values

**Function:** `runDemandAnalytics()` [Line ~850]

---

### Tab 3: Train Planning 🚆
**Purpose:** Multi-hour scheduling optimization

**APIs Called:**
1. Demand API → `/api/demand/predict` → For hours 6-22
2. Induction API → `/api/induction/recommend` → For each hour

**User Inputs:**
- Available trains (2-20)
- Peak mode toggle (on/off)

**Outputs Displayed:**
- ✅ Table with hour-by-hour schedule
- ✅ Predicted demand per hour
- ✅ Recommended trains
- ✅ Headway (minutes between trains)

**Function:** `generatePlan()` [Line ~900]

---

### Tab 4: Scenario Simulator 🔮
**Purpose:** Contingency planning & what-if analysis

**APIs Called:**
1. Demand API → `/api/demand/predict` → Base demand at 9 AM
2. Induction API → `/api/induction/recommend` → Adjusted demand
3. Surge Detection → Local function

**User Inputs:**
- Scenario date
- Demand increase % (0-100%)
- Unavailable trains (0-10)
- Special events (Rain, Festival)

**Multipliers Applied:**
- User demand increase: 1 + (increase / 100)
- Rain: × 1.2
- Festival: × 1.35

**Outputs Displayed:**
- ✅ Scenario feasibility (✓ Manageable or ❌ Capacity Exceeded)
- ✅ Base vs scenario demand comparison
- ✅ Train configuration analysis
- ✅ Impact metrics:
  - Average waiting time
  - Crowding level %
  - Surge status
- ✅ Applied factors summary

**Function:** `runSimulation()` [Line ~950]

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   FRONTEND (index.html)                     │
│  ┌─────────────┬─────────────┬──────────────┬──────────────┐│
│  │ Operations  │   Demand    │    Train     │   Scenario   ││
│  │  Control    │  Analytics  │   Planning   │  Simulator   ││
│  └─────────────┴─────────────┴──────────────┴──────────────┘│
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │            JavaScript Functions                        │ │
│  │  • fetchDemandForecast()                              │ │
│  │  • fetchTrainRecommendation()                         │ │
│  │  • detectSurge()                                      │ │
│  │  • getWeatherData()                                   │ │
│  │  • weatherDemandMultiplier()                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                    ↓                    ↓                    │
└────────────────────────┼──────────────────┼─────────────────┘
                         │                  │
            ┌────────────▼──┐    ┌──────────▼──────────┐
            │ External APIs │    │ FastAPI Backend     │
            ├───────────────┤    │ (localhost:8000)    │
            │ WeatherAPI    │    ├─────────────────────┤
            │               │    │ demand_api.py       │
            │ (Optional)    │    │  → /api/demand/     │
            │               │    │     predict         │
            └───────────────┘    │                     │
                                 │ induction_api.py    │
                                 │  → /api/induction/  │
                                 │     recommend       │
                                 │                     │
                                 │ surge_detection.py  │
                                 │  (logic embedded)   │
                                 │                     │
                                 │ app.py              │
                                 │  (FastAPI server)   │
                                 └─────────────────────┘
                                          ↓
                                   ML Models / Q-Table
                                   (model/ directory)
```

---

## API Response Formats

### Demand Prediction Response
```json
{
  "predicted_demand": 5234,
  "weather": {
    "temp": 28.5,
    "rain_mm": 0.5,
    "condition": "Partly cloudy"
  },
  "explanation": "Passenger demand is influenced by peak-hour travel patterns and prevailing weather conditions."
}
```

### Train Induction Response
```json
{
  "recommended_trains": 6,
  "confidence": 92,
  "policy": "reinforcement-learning"
}
```

### Surge Detection Response (Local)
```json
{
  "surge": true,
  "severity": "High"
}
```

---

## Data Flow Examples

### Scenario 1: Morning Rush Hour (9 AM, Weekday)
```
User Input → Hour: 9, Weekday, 5 trains/hour, Direction: A→B
                ↓
Demand API → Predicts 5234 passengers
Weather API → 28°C, Clear (multiplier: 1.0x)
Adjusted: 5234 passengers
                ↓
Induction API → RL recommends 6 trains
Surge Detection → Normal (< 7000)
                ↓
Display: "Deploy 6 trains/hour (10-min headway)"
```

### Scenario 2: Festive Season Peak
```
User Input → Demand +20%, Festival ON, 2 trains unavailable
Base Demand: 5234 passengers (peak hour)
                ↓
Apply Multipliers:
  • +20% = 1.2x = 6281
  • Festival = × 1.35 = 8479
Available Trains: 10 - 2 = 8
                ↓
Induction API → RL recommends 9 trains (RL suggests more)
Available: 8 < Recommended: 9
                ↓
Display: "⚠️ Capacity Exceeded! Need 9 trains but only 8 available"
```

---

## Testing Checklist

- [ ] Backend running on port 8000
- [ ] Frontend accessible (file or web server)
- [ ] Operations Control: Click "Run AI" → Shows recommendation
- [ ] Demand Analytics: Click "Run" → Shows heatmap
- [ ] Train Planning: Click "Generate AI Plan" → Shows schedule table
- [ ] Scenario Simulator: Adjust values, click "Run Simulation" → Shows analysis
- [ ] All error messages appear (test with backend offline)

---

## Performance Metrics

| Operation | Expected Time | Bottleneck |
|-----------|--------------|-----------|
| Single demand prediction | < 500ms | ML model inference |
| Train recommendation | < 100ms | RL Q-table lookup |
| Weather fetch | 1-3s | External API |
| Heatmap (16 hours) | 8-10s | Multiple API calls |
| Full scenario | 2-3s | Sequential API calls |

---

## Configuration Variables

### Frontend Configuration
**File:** [Frontend/index.html](Frontend/index.html)

```javascript
const API_BASE = "http://localhost:8000/api";  // Backend URL
const MIN_TRAINS = 2;                          // Min trains constraint
const MAX_TRAINS = 20;                         // Max trains constraint
```

### Peak Hour Definition
```javascript
const isPeak = (hour >= 8 && hour <= 10) || (hour >= 17 && hour <= 20);
```

### Surge Detection Threshold
```javascript
const threshold = 7000;  // Passengers
```

---

## Known Issues & Workarounds

| Issue | Cause | Workaround |
|-------|-------|-----------|
| "Backend unavailable" | Server not running | Start: `python -m uvicorn backend.app:app --reload` |
| Weather API timeout | External service down | Dashboard uses fallback (28°C, clear) |
| Heatmap is slow | Multiple API calls | Reduce time range (6-12 instead of 6-22) |
| RL policy not used | Q-table not found | Falls back to rule-based (confidence: 78%) |
| CORS errors | Browser security | CORS already enabled in backend |
| Blank dashboard | JavaScript error | Check: F12 → Console tab |

---

## Next Steps

1. **Verify Setup:**
   ```bash
   # Terminal 1: Start backend
   python -m uvicorn backend.app:app --reload --port 8000
   
   # Terminal 2: Open frontend
   # Navigate to Frontend/index.html in browser
   ```

2. **Test Each Tab:**
   - Operations Control: Verify recommendations update with hour changes
   - Demand Analytics: Check heatmap shows varying demand across hours
   - Train Planning: Verify schedule respects available trains
   - Scenario Simulator: Test multipliers and impact calculations

3. **Monitor API Calls:**
   - Open browser Developer Tools: `F12 → Network tab`
   - Click "Run AI"
   - Verify requests to `localhost:8000/api/*`

4. **Customize as Needed:**
   - Change peak hour ranges
   - Adjust surge detection threshold
   - Modify station list
   - Add more features

---

## Support & Documentation

- **Quick Start:** [QUICK_START.md](QUICK_START.md)
- **Detailed Guide:** [FRONTEND_INTEGRATION_GUIDE.md](FRONTEND_INTEGRATION_GUIDE.md)
- **FastAPI Auto-Docs:** http://localhost:8000/docs (when backend running)
- **Model Training:** [notebook/demand-forecasting.ipynb](notebook/demand-forecasting.ipynb)
- **RL Training:** [notebook/rl_train_induction.ipynb](notebook/rl_train_induction.ipynb)

---

**Integration Completed:** January 23, 2026
**Status:** ✅ Ready for Testing & Deployment
**Frontend File:** [Frontend/index.html](Frontend/index.html)
**Backend Directory:** [backend/](backend/)
