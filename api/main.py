from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD MODEL AND FEATURES ────────────────────────────────────────────────
print("Loading model...")
model       = joblib.load('../models/best_model.pkl')
feature_cols = joblib.load('../models/feature_cols.pkl')
print(f"Model loaded ✅")
print(f"Features expected: {len(feature_cols)}")

# ── 2. LOAD HISTORICAL DATA FOR LAG COMPUTATION ───────────────────────────────
# We need recent sales history to compute lag features at prediction time
# This is a key production consideration — model needs context to predict
print("Loading historical data for lag computation...")
history = pd.read_csv('../data/features.csv', parse_dates=['date'])
history = history[['date', 'sales']].sort_values('date').reset_index(drop=True)
print(f"History loaded: {len(history)} days")

# ── 3. FASTAPI APP ────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Retail Demand Forecast API",
    description = "Predicts daily FOODS sales for Walmart CA_1 store using LightGBM",
    version     = "1.0.0"
)

# ── 4. REQUEST SCHEMA ─────────────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    forecast_date : str   = Field(..., example="2016-04-25",
                                  description="Date to forecast (YYYY-MM-DD)")
    avg_price     : float = Field(default=2.50, example=2.50,
                                  description="Average item price on that day")
    snap_CA       : int   = Field(default=0,    example=0,
                                  description="Is it a SNAP benefit day? (0 or 1)")
    has_event     : int   = Field(default=0,    example=0,
                                  description="Is there a calendar event? (0 or 1)")
    is_sporting_event: int = Field(default=0,   example=0,
                                  description="Is it a sporting event day? (0 or 1)")

# ── 5. RESPONSE SCHEMA ────────────────────────────────────────────────────────
class ForecastResponse(BaseModel):
    forecast_date     : str
    predicted_sales   : int
    lower_bound       : int
    upper_bound       : int
    snap_day          : bool
    business_insight  : str

# ── 6. FEATURE BUILDER ────────────────────────────────────────────────────────
def build_features(req: ForecastRequest) -> pd.DataFrame:
    """
    Build the same 30 features the model was trained on.
    At prediction time we compute lags from historical data.
    """
    target_date = pd.to_datetime(req.forecast_date)

    # ── Lag features from history ─────────────────────────────────────────────
    def get_lag(days):
        lag_date = target_date - timedelta(days=days)
        row = history[history['date'] == lag_date]
        if len(row) == 0:
            # fallback to nearest available date
            row = history[history['date'] <= lag_date].tail(1)
        return float(row['sales'].values[0]) if len(row) > 0 else 3000.0

    lag_7  = get_lag(7)
    lag_14 = get_lag(14)
    lag_28 = get_lag(28)

    # ── Rolling features from history ─────────────────────────────────────────
    def get_rolling(window):
        end_date   = target_date - timedelta(days=1)
        start_date = end_date - timedelta(days=window)
        window_data = history[
            (history['date'] > start_date) &
            (history['date'] <= end_date)
        ]['sales']
        if len(window_data) == 0:
            return 3000.0, 300.0
        return float(window_data.mean()), float(window_data.std())

    rm7,  rs7  = get_rolling(7)
    rm14, rs14 = get_rolling(14)
    rm28, rs28 = get_rolling(28)

    # ── Price features ────────────────────────────────────────────────────────
    price_lag_7    = req.avg_price * 0.98      # slight historical approximation
    price_change   = req.avg_price - price_lag_7
    price_pct_change = price_change / price_lag_7 if price_lag_7 != 0 else 0

    # ── Time features ─────────────────────────────────────────────────────────
    dow         = target_date.dayofweek
    dom         = target_date.day
    month       = target_date.month
    quarter     = target_date.quarter
    year        = target_date.year
    is_weekend  = int(dow >= 5)
    is_month_start = int(dom == 1)
    is_month_end   = int(dom == target_date.days_in_month)

    # Cyclical encoding
    wday_sin   = np.sin(2 * np.pi * dow   / 7)
    wday_cos   = np.cos(2 * np.pi * dow   / 7)
    month_sin  = np.sin(2 * np.pi * month / 12)
    month_cos  = np.cos(2 * np.pi * month / 12)

    # Trend — extrapolate beyond training data
    last_trend = len(history)
    days_ahead = (target_date - history['date'].max()).days
    trend      = last_trend + days_ahead

    # ── Assemble feature dict ─────────────────────────────────────────────────
    features = {
        'avg_price'         : req.avg_price,
        'snap_CA'           : req.snap_CA,
        'year'              : year,
        'has_event'         : req.has_event,
        'is_sporting_event' : req.is_sporting_event,
        'lag_7'             : lag_7,
        'lag_14'            : lag_14,
        'lag_28'            : lag_28,
        'rolling_mean_7'    : rm7,
        'rolling_std_7'     : rs7,
        'rolling_mean_14'   : rm14,
        'rolling_std_14'    : rs14,
        'rolling_mean_28'   : rm28,
        'rolling_std_28'    : rs28,
        'price_lag_7'       : price_lag_7,
        'price_change'      : price_change,
        'price_pct_change'  : price_pct_change,
        'day_of_week'       : dow,
        'day_of_month'      : dom,
        'quarter'           : quarter,
        'is_weekend'        : is_weekend,
        'is_month_start'    : is_month_start,
        'is_month_end'      : is_month_end,
        'wday_sin'          : wday_sin,
        'wday_cos'          : wday_cos,
        'month_sin'         : month_sin,
        'month_cos'         : month_cos,
        'trend'             : trend,
        'event_name_1_enc'  : int(req.has_event),
        'event_name_2_enc'  : 0,
    }

    return pd.DataFrame([features])[feature_cols]

# ── 7. BUSINESS INSIGHT GENERATOR ────────────────────────────────────────────
def generate_insight(predicted: int, snap: bool, has_event: bool,
                     lower: int, upper: int) -> str:
    insights = []

    if snap:
        insights.append(
            f"SNAP benefit day detected — expect ~12% demand lift. "
            f"Recommend pre-stocking FOODS inventory by EOD prior day."
        )
    if has_event:
        insights.append(
            "Calendar event detected — demand volatility likely. "
            "Monitor sell-through hourly."
        )
    if predicted > 3500:
        insights.append(
            f"High demand day forecast ({predicted:,} units). "
            f"Ensure full shelf capacity and extra staff."
        )
    elif predicted < 2500:
        insights.append(
            f"Low demand day forecast ({predicted:,} units). "
            f"Good day for restocking and inventory audits."
        )

    if not insights:
        insights.append(
            f"Normal demand day. Forecast range: {lower:,}–{upper:,} units. "
            f"Standard operations recommended."
        )

    return " | ".join(insights)

# ── 8. ENDPOINTS ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message"    : "Retail Demand Forecast API",
        "version"    : "1.0.0",
        "model"      : "LightGBM",
        "mape"       : "6.70%",
        "docs"       : "/docs",
        "health"     : "/health"
    }

@app.get("/health")
def health():
    return {
        "status"          : "healthy",
        "model_loaded"    : True,
        "history_days"    : len(history),
        "last_history_date": str(history['date'].max().date())
    }

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    try:
        # Validate date format
        target_date = pd.to_datetime(req.forecast_date)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    try:
        # Build features
        X = build_features(req)

        # Predict
        prediction = int(model.predict(X)[0])

        # Confidence interval — use training MAPE as uncertainty estimate
        mape_buffer = 0.067   # 6.70% MAPE
        lower = int(prediction * (1 - mape_buffer))
        upper = int(prediction * (1 + mape_buffer))

        # Business insight
        insight = generate_insight(
            predicted = prediction,
            snap      = bool(req.snap_CA),
            has_event = bool(req.has_event),
            lower     = lower,
            upper     = upper
        )

        return ForecastResponse(
            forecast_date   = req.forecast_date,
            predicted_sales = prediction,
            lower_bound     = lower,
            upper_bound     = upper,
            snap_day        = bool(req.snap_CA),
            business_insight= insight
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/week")
def forecast_week(start_date: str = "2016-04-25", avg_price: float = 2.50):
    """Forecast the next 7 days from a start date"""
    results = []
    for i in range(7):
        date = (pd.to_datetime(start_date) + timedelta(days=i)).strftime('%Y-%m-%d')
        req  = ForecastRequest(
            forecast_date = date,
            avg_price     = avg_price
        )
        X          = build_features(req)
        prediction = int(model.predict(X)[0])
        results.append({
            "date"           : date,
            "predicted_sales": prediction,
            "day_of_week"    : pd.to_datetime(date).strftime('%A')
        })
    return {"week_forecast": results, "total_predicted": sum(r['predicted_sales'] for r in results)}