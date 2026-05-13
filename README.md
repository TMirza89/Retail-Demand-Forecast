# 🛒 Retail Demand Forecast

A production-grade demand forecasting system for retail FOODS sales, built with LightGBM, FastAPI, MLflow, and Streamlit.

---

## 📊 Project Overview

Forecasts daily food sales for Walmart Store CA_1 using the M5 Forecasting dataset (30,000+ SKUs, 5 years of data). Achieves **6.70% MAPE** — outperforming the retail industry benchmark of 10-15%.

### Key Business Insights
- **SNAP benefit days** drive a 12.1% sales lift — enabling proactive inventory planning
- **Day of month** is the strongest predictor — pay cycle drives consumer behavior
- **Weekend surge** peaks on Sunday — 3,733 units vs 1,573 on Wednesday
- Model uncertainty flagging — when LightGBM and XGBoost disagree by >10%, the system flags the day for manual review

---

## 🏗️ Architecture

```
M5 Dataset (3 CSVs)
      ↓
Feature Engineering (30 features)
      ↓
Model Training → LightGBM (MAPE: 6.70%) + XGBoost (MAPE: 6.73%)
      ↓
MLflow Experiment Tracking
      ↓
FastAPI (4 endpoints) → Docker Container
      ↓
Streamlit Dashboard (interactive UI)
```

---

## 🚀 Quick Start

### Option 1 — Local

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/retail-demand-forecast.git
cd retail-demand-forecast

# Install
pip install -r requirements.txt

# Download M5 data from Kaggle into /data folder
# https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

# Run pipeline
python src/features.py
python src/train.py

# Start API
cd api && uvicorn main:app --reload --port 8000

# Start Dashboard (new terminal)
cd app && streamlit run streamlit_app.py
```

### Option 2 — Docker

```bash
docker-compose up --build
```

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`
- API Docs: `http://localhost:8000/docs`

---

## 📁 Project Structure

```
retail-demand-forecast/
├── data/                        # processed data + plots
├── src/
│   ├── eda.py                   # exploratory data analysis
│   ├── features.py              # feature engineering (30 features)
│   └── train.py                 # model training + MLflow tracking
├── api/
│   └── main.py                  # FastAPI (4 endpoints)
├── app/
│   └── streamlit_app.py         # interactive dashboard
├── models/                      # saved models
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/forecast` | Single day forecast |
| GET | `/forecast/week` | 7-day rolling forecast |

### Example Request

```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "forecast_date": "2016-04-25",
    "avg_price": 2.50,
    "snap_CA": 1,
    "has_event": 0,
    "is_sporting_event": 0
  }'
```

### Example Response

```json
{
  "forecast_date": "2016-04-25",
  "predicted_sales": 2372,
  "lower_bound": 2213,
  "upper_bound": 2530,
  "snap_day": true,
  "business_insight": "SNAP benefit day detected — expect ~12% demand lift. Recommend pre-stocking FOODS inventory by EOD prior day."
}
```

---

## 🤖 Models

| Model | MAPE | MAE | RMSE |
|---|---|---|---|
| **LightGBM** ✅ | **6.70%** | **201 units** | **266 units** |
| XGBoost | 6.73% | 205 units | 276 units |

### Top Features by Importance
1. `day_of_month` — pay cycle effect
2. `lag_14` — 2-week demand memory
3. `lag_7` — same day last week
4. `rolling_mean_7` — recent weekly trend
5. `event_name_1_enc` — calendar event impact

---

## 📈 Results

- **MAPE: 6.70%** vs industry benchmark of 10-15%
- **5 years** of training data (2011-2016)
- **1,885 daily records** after feature engineering
- **30 features** including lag, rolling, cyclical, and event features
- **2 models** tracked and compared via MLflow

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data | M5 Walmart Forecasting Dataset |
| Feature Engineering | Pandas, NumPy |
| Modeling | LightGBM, XGBoost |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| Dashboard | Streamlit |
| Language | Python 3.11 |

---

## 💼 Business Impact

> *"By identifying that SNAP benefit days drive a 12.1% sales lift, store managers can pre-stock FOODS inventory the prior evening — reducing stockouts and improving customer satisfaction. At 2,800 avg daily units and $2.50 avg price, a 12.1% lift represents approximately $847 in additional daily revenue per SNAP day."*

---

## 📝 Data Source

M5 Forecasting Competition — Walmart Sales Data
- [Kaggle Competition Page](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- Store: CA_1 | Category: FOODS | Date range: 2011-2016 