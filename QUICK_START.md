# Quick Start: Running the Integrated Dashboard

## Prerequisites
- Python 3.8+
- FastAPI installed (`pip install fastapi`)
- Uvicorn installed (`pip install uvicorn`)
- Modern web browser (Chrome, Firefox, Safari, Edge)

---

## Step 1: Start the Backend FastAPI Server

```bash
cd c:\Users\kaviy\ai-train-induction-planning-and-scheduling
python -m uvicorn backend.app:app --reload --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

✅ Backend is now running at `http://localhost:8000/api`

---

## Step 2: Open the Frontend Dashboard

### Option A: Direct File Open (Easiest)
1. Navigate to: `Frontend/index.html`
2. Right-click → "Open with" → Your browser
3. Dashboard loads at `file:///C:/Users/kaviy/ai-train-induction-planning-and-scheduling/Frontend/index.html`

### Option B: Local Web Server
```bash
cd Frontend
python -m http.server 8080
```
Then visit: `http://localhost:8080/index.html`

---

## Step 3: Verify Connection

In the dashboard:
1. Go to **Operations Control** tab
2. Adjust hour to 9 (peak hour)
3. Click **Run AI** button
4. ✅ You should see recommendations with real backend data

**If you see error messages:**
- Backend server not running → Start it (Step 1)
- Wrong API_BASE URL → Check [Frontend/index.html](Frontend/index.html) line ~350

---

## What Each Tab Does

### 🚆 Operations Control (Tab 1)
- **Input:** Hour, Day Type, Current Trains, Direction
- **Output:** AI recommendations with weather impact
- **Example:** "At 9 AM on weekday with 5 trains/hour → Recommend 6 trains/hour (headway: 10 min)"

### 📊 Demand Analytics (Tab 2)
- **Input:** Date, Time range, Day type, Comparison toggle
- **Output:** Heatmap showing passenger demand across stations and hours
- **Example:** Peak hours (8-10 AM, 5-8 PM) show red (high demand), off-peak shows green (low demand)

### 🚆 Train Planning (Tab 3)
- **Input:** Available trains, Peak mode toggle
- **Output:** Hourly schedule with recommended train counts
- **Example:** "6 AM: 2 trains, 9 AM: 8 trains, 5 PM: 8 trains"

### 🔮 Scenario Simulator (Tab 4)
- **Input:** Demand increase %, unavailable trains, weather/festival events
- **Output:** Feasibility assessment and impact metrics
- **Example:** "Festival scenario: Need 10 trains but only 8 available → Capacity Exceeded"

---

## Data Flow Diagram

```
Frontend Dashboard (index.html)
    ↓
[User Input: hour, demand %, trains, etc.]
    ↓
JavaScript Functions
    ↓
Fetch API Calls (HTTP POST)
    ↓
FastAPI Backend (port 8000)
    ├─ /api/demand/forecast → ML demand prediction model
    ├─ /api/induction/recommend → RL-based train scheduling
    └─ /api/stations → Station metadata
    ↓
[Process with ML/RL Models]
    ↓
Return JSON Response
    ↓
Frontend Updates Charts/Tables/Alerts
    ↓
Display Results to User
```

---

## API Endpoints Summary

| Endpoint | Method | Purpose | Used By |
|----------|--------|---------|---------|
| `/api/demand/forecast` | POST | Predict passenger demand | All tabs |
| `/api/induction/recommend` | POST | Recommend trains via RL | Tabs 1, 3, 4 |
| `/api/stations` | GET | Get station metadata | Tab 2 (heatmap) |
| `/` | GET | Health check | Dashboard init |

---

## Troubleshooting

### Issue: "Backend service unavailable" error
**Solution:**
1. Check if backend is running: `http://localhost:8000/` in browser
2. If not, run: `python -m uvicorn backend.app:app --reload --port 8000`
3. Refresh dashboard

### Issue: CORS error in browser console
**Solution:**
- Backend CORS is already configured in [backend/app.py](backend/app.py)
- If still getting errors, allow all origins (development only):
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],  # ← Already set
  )
  ```

### Issue: Weather data not showing
**Solution:**
- External weather API is optional
- Dashboard uses fallback values (28°C, 0mm rain)
- To use real weather, add `WEATHER_API_KEY` to `.env` file

### Issue: Charts not rendering
**Solution:**
- Clear browser cache: `Ctrl+Shift+Delete`
- Check if Chart.js loaded: Search for "Chart.js" in [Frontend/index.html](Frontend/index.html)
- Verify JavaScript has no errors: Press `F12` → Console tab

---

## File Structure for Reference

```
ai-train-induction-planning-and-scheduling/
├── Frontend/
│   └── index.html                    ← Main dashboard (UPDATED ✅)
├── backend/
│   ├── app.py                        ← FastAPI server
│   ├── api/
│   │   ├── demand_api.py            ← Demand forecasting
│   │   ├── induction_api.py         ← Train scheduling (RL)
│   │   └── stations_api.py          ← Station metadata
│   └── utils/
│       └── surge_detection.py       ← Surge detection logic
├── model/
│   ├── demand_forecast_model.pkl    ← ML model (required)
│   ├── model_features.pkl           ← Model features (required)
│   └── rl_q_table.pkl              ← RL Q-table (optional)
├── notebook/
│   ├── data-preprocessing.ipynb
│   ├── demand-forecasting.ipynb
│   └── rl_train_induction.ipynb
├── data/
│   ├── raw/                         ← GTFS data
│   └── processed/                   ← Processed data
└── Control_Dashboard.py             ← Streamlit dashboard (alternative)
```

---

## Performance Tips

1. **Heatmap generation is slow:** Reduce time range (e.g., 6-12 instead of 6-22)
2. **Multiple requests:** Dashboard batches them; wait for "⏳ Processing..." to complete
3. **Large dataset queries:** Use Demand Analytics with filters (weekday/weekend, time range)

---

## Next Steps

1. ✅ **Verify integration:** Run dashboard and check all tabs
2. ✅ **Test scenario:** Try different demand increases and observe impact
3. ✅ **Review KPIs:** Check waiting time, load, and crowding metrics
4. 📝 **Customize thresholds:** Edit peak hour ranges, surge thresholds in code
5. 🚀 **Deploy:** Copy to web server for team access

---

## Support Resources

- **FastAPI Docs:** http://localhost:8000/docs (auto-generated when backend runs)
- **Backend Integration Guide:** [FRONTEND_INTEGRATION_GUIDE.md](FRONTEND_INTEGRATION_GUIDE.md)
- **Model Training:** See [notebook/demand-forecasting.ipynb](notebook/demand-forecasting.ipynb)
- **RL Training:** See [notebook/rl_train_induction.ipynb](notebook/rl_train_induction.ipynb)

---

**Status:** ✅ Ready to use
**Last Updated:** January 23, 2026
