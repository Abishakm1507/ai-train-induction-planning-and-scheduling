# 🚆 AI Train Induction Planning and Scheduling System

An end-to-end, AI-driven operations management platform for the **Kochi Metro Rail Limited (KMRL)**. This system dynamically optimizes metro operations by predicting passenger demand and recommending optimal train induction and withdrawal timings.

## 🌟 Key Features

- **Weather-Aware Demand Forecasting**: Predicts passenger inflow using historical data integrated with real-time weather conditions (Rain, Temperature).
- **RL-Based Train Induction**: Utilizes Reinforcement Learning (Q-Learning) to recommend the number of trains to deploy based on demand and peak hours.
- **Explainable AI (XAI)**: Integrated with **Google Gemini AI** to provide natural language "AI Insights," explaining the reasoning behind operational decisions directly on the dashboard.
- **Interactive Dashboards**:
  - **Operations Control**: Real-time monitoring and AI recommendations with dynamic explanations.
  - **Passenger Demand Analytics**: Visualizes historical and predicted trends.
  - **What-If Scenario Simulator**: Simulates the impact of demand surges or service changes on KPIs.
- **KPI Monitoring**: Tracks Load Factor, Waiting Time, Energy Efficiency, and Passenger Comfort.

## 🛠️ Tech Stack

- **Frontend**: HTML5, Vanilla CSS, JavaScript (Kochi Metro AI Dashboard)
- **Backend**: FastAPI
- **Data Science**: Pandas, NumPy, Scikit-learn, Joblib
- **Machine Learning**: Random Forest (Forecasting), Q-Learning (RL Induction)
- **Generative AI**: Google Gemini AI 
- **Visualization**: Matplotlib, Seaborn

## 📂 Project Structure

```text
├── frontend/                   # Web Frontend
│   └── index.html              # Main AI Operations Dashboard (Single Page App)
├── backend/                    # FastAPI Backend
│   ├── app.py                  # API hub
│   ├── api/                    # Routers (Demand, Induction, Stations)
│   └── start_server.py         # Startup script
├── Control_Dashboard.py        # Legacy Streamlit entrance (Optional)
├── pages/                      # Legacy Streamlit pages (Optional)
├── notebook/                   # Jupyter Notebooks for R&D
├── model/                      # Serialized ML & RL models (.pkl)
└── data/                       # Datasets
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- [Google Gemini API Key](https://aistudio.google.com/)
- [WeatherAPI Key](https://www.weatherapi.com/)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-train-induction-planning-and-scheduling
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have `streamlit`, `fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `google-generativeai`, `python-dotenv`, and `requests` installed.)*

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   WEATHER_API_KEY=your_weather_api_key
   GOOGLE_API_KEY=your_gemini_api_key
   ```

### Running the Application

1. **Start the Backend API**:
   ```bash
   cd backend
   python start_server.py
   ```
   The API will be available at `http://localhost:8001`.

2. **Open the Dashboard**:
   Simply open `frontend/index.html` in any modern web browser or use a live server extension.
   - The dashboard will connect to the backend API at `http://127.0.0.1:8001`.

## 🤖 AI Logic Overview

### Demand Forecasting
The system uses a Random Forest Regressor trained on historical ticketing data, enhanced by external weather features. Higher rainfall typically correlates with increased road traffic and higher metro demand.

### RL-Based Induction
A Q-Learning agent was trained to find the balance between:
- **Minimizing Waiting Time**: Deploying more trains.
- **Reducing Energy/Cost**: Avoiding empty runs.

### Explainable AI (XAI)
The system integrates **Google Gemini AI** to transform complex operational data into human-readable insights. It contextualizes demand forecasts and RL recommendations, providing staff with clear justifications for high-stakes operational changes.

---
*Developed for Metro Operations Optimization.*
