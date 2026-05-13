import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Retail Demand Forecast",
    page_icon  = "🛒",
    layout     = "wide"
)

# ── LOAD MODELS DIRECTLY (no API call needed) ─────────────────────────────────
# We load both models so user can compare them side by side
@st.cache_resource
def load_models():
    import os
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    data_dir   = os.path.join(os.path.dirname(__file__), '..', 'data')

    lgbm    = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
    feat    = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))
    history = pd.read_csv(os.path.join(data_dir, 'features.csv'), parse_dates=['date'])
    return lgbm, feat, history

@st.cache_resource
def load_xgb():
    import os
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    try:
        xgb = joblib.load(os.path.join(models_dir, 'xgb_model.pkl'))
        return xgb
    except:
        return None

lgbm_model, feature_cols, history = load_models()
xgb_model = load_xgb()

# ── FEATURE BUILDER ───────────────────────────────────────────────────────────
def build_features(target_date, avg_price, snap_CA, has_event, is_sporting_event):
    from datetime import timedelta

    def get_lag(days):
        lag_date = target_date - timedelta(days=days)
        row = history[history['date'] == lag_date]
        if len(row) == 0:
            row = history[history['date'] <= lag_date].tail(1)
        return float(row['sales'].values[0]) if len(row) > 0 else 3000.0

    def get_rolling(window):
        end_date   = target_date - timedelta(days=1)
        start_date = end_date - timedelta(days=window)
        data = history[
            (history['date'] > start_date) &
            (history['date'] <= end_date)
        ]['sales']
        if len(data) == 0:
            return 3000.0, 300.0
        return float(data.mean()), float(data.std())

    lag_7,  lag_14, lag_28 = get_lag(7), get_lag(14), get_lag(28)
    rm7,  rs7  = get_rolling(7)
    rm14, rs14 = get_rolling(14)
    rm28, rs28 = get_rolling(28)

    price_lag_7      = avg_price * 0.98
    price_change     = avg_price - price_lag_7
    price_pct_change = price_change / price_lag_7

    dow            = target_date.dayofweek
    dom            = target_date.day
    month          = target_date.month
    last_trend     = len(history)
    days_ahead     = (target_date - history['date'].max()).days
    trend          = last_trend + days_ahead

    features = {
        'avg_price'          : avg_price,
        'snap_CA'            : snap_CA,
        'year'               : target_date.year,
        'has_event'          : has_event,
        'is_sporting_event'  : is_sporting_event,
        'lag_7'              : lag_7,
        'lag_14'             : lag_14,
        'lag_28'             : lag_28,
        'rolling_mean_7'     : rm7,
        'rolling_std_7'      : rs7,
        'rolling_mean_14'    : rm14,
        'rolling_std_14'     : rs14,
        'rolling_mean_28'    : rm28,
        'rolling_std_28'     : rs28,
        'price_lag_7'        : price_lag_7,
        'price_change'       : price_change,
        'price_pct_change'   : price_pct_change,
        'day_of_week'        : dow,
        'day_of_month'       : dom,
        'quarter'            : target_date.quarter,
        'is_weekend'         : int(dow >= 5),
        'is_month_start'     : int(dom == 1),
        'is_month_end'       : int(dom == pd.Timestamp(target_date).days_in_month),
        'wday_sin'           : np.sin(2 * np.pi * dow   / 7),
        'wday_cos'           : np.cos(2 * np.pi * dow   / 7),
        'month_sin'          : np.sin(2 * np.pi * month / 12),
        'month_cos'          : np.cos(2 * np.pi * month / 12),
        'trend'              : trend,
        'event_name_1_enc'   : int(has_event),
        'event_name_2_enc'   : 0,
    }
    return pd.DataFrame([features])[feature_cols]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🛒 Retail Demand Forecast Dashboard")
st.markdown("**Store:** Walmart CA_1 &nbsp;|&nbsp; **Category:** FOODS &nbsp;|&nbsp; **Model MAPE:** 6.70%")
st.divider()

# ── SIDEBAR CONTROLS ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Forecast Settings")

forecast_date = st.sidebar.date_input(
    "Forecast Date",
    value     = datetime(2016, 4, 25),
    min_value = datetime(2016, 1, 1),
    max_value = datetime(2017, 12, 31)
)

avg_price = st.sidebar.slider(
    "Average Item Price ($)",
    min_value = 1.0,
    max_value = 10.0,
    value     = 2.50,
    step      = 0.10
)

snap_day = st.sidebar.toggle("📋 SNAP Benefit Day", value=False)
has_event = st.sidebar.toggle("🎉 Calendar Event", value=False)
is_sporting = st.sidebar.toggle("🏈 Sporting Event", value=False)

model_choice = st.sidebar.radio(
    "🤖 Model",
    options = ["LightGBM", "XGBoost"],
    index   = 0
)

show_week = st.sidebar.toggle("📅 Show 7-Day Forecast", value=True)

st.sidebar.divider()
st.sidebar.markdown("**Model Performance**")
st.sidebar.markdown("| Model | MAPE | MAE |")
st.sidebar.markdown("|---|---|---|")
st.sidebar.markdown("| LightGBM | 6.70% | 201 |")
st.sidebar.markdown("| XGBoost  | 6.73% | 205 |")

# ── PREDICT ───────────────────────────────────────────────────────────────────
target_date = pd.Timestamp(forecast_date)
X = build_features(
    target_date      = target_date,
    avg_price        = avg_price,
    snap_CA          = int(snap_day),
    has_event        = int(has_event),
    is_sporting_event= int(is_sporting)
)

# Choose model
if model_choice == "XGBoost" and xgb_model is not None:
    model     = xgb_model
    mape_buf  = 0.0673
else:
    model     = lgbm_model
    mape_buf  = 0.0670
    model_choice = "LightGBM"

prediction = int(model.predict(X)[0])
lower      = int(prediction * (1 - mape_buf))
upper      = int(prediction * (1 + mape_buf))

# ── TOP METRICS ROW ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label = "📦 Predicted Sales",
        value = f"{prediction:,} units",
        delta = f"±{prediction - lower:,} units"
    )
with col2:
    st.metric(label="📉 Lower Bound", value=f"{lower:,} units")
with col3:
    st.metric(label="📈 Upper Bound", value=f"{upper:,} units")
with col4:
    snap_lift = int(prediction * 0.121) if snap_day else 0
    st.metric(
        label = "🎯 SNAP Lift",
        value = f"+{snap_lift:,} units" if snap_day else "No SNAP",
        delta = "12.1% boost" if snap_day else None
    )

st.divider()

# ── BUSINESS INSIGHT ──────────────────────────────────────────────────────────
insights = []
if snap_day:
    insights.append("📋 **SNAP Day:** Expect ~12% demand lift. Pre-stock FOODS inventory by EOD prior day.")
if has_event:
    insights.append("🎉 **Event Day:** Demand volatility likely. Monitor sell-through hourly.")
if is_sporting:
    insights.append("🏈 **Sporting Event:** Snack and beverage categories will spike.")
if prediction > 3500:
    insights.append(f"🔴 **High Demand Day:** {prediction:,} units forecast. Ensure full shelf capacity and extra staff.")
elif prediction < 2500:
    insights.append(f"🟡 **Low Demand Day:** {prediction:,} units forecast. Good day for restocking and inventory audits.")
else:
    insights.append(f"🟢 **Normal Demand Day:** {prediction:,} units forecast. Standard operations recommended.")

st.subheader("💡 Business Insight")
for insight in insights:
    st.info(insight)

# ── 7-DAY FORECAST CHART ──────────────────────────────────────────────────────
if show_week:
    st.subheader("📅 7-Day Forecast")

    week_dates  = [target_date + timedelta(days=i) for i in range(7)]
    week_preds  = []
    week_lowers = []
    week_uppers = []

    for d in week_dates:
        X_d  = build_features(d, avg_price, int(snap_day), int(has_event), int(is_sporting))
        pred = int(model.predict(X_d)[0])
        week_preds.append(pred)
        week_lowers.append(int(pred * (1 - mape_buf)))
        week_uppers.append(int(pred * (1 + mape_buf)))

    week_df = pd.DataFrame({
        'Date'      : [d.strftime('%a %b %d') for d in week_dates],
        'Forecast'  : week_preds,
        'Lower'     : week_lowers,
        'Upper'     : week_uppers,
    })

    # Chart
    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(len(week_dates))

    ax.fill_between(x, week_lowers, week_uppers, alpha=0.2, color='steelblue', label='Confidence Band')
    ax.plot(x, week_preds, marker='o', linewidth=2, color='steelblue', markersize=8, label='Forecast')
    ax.axhline(y=np.mean(week_preds), linestyle='--', color='coral', linewidth=1, alpha=0.7, label=f'Week Avg: {int(np.mean(week_preds)):,}')

    ax.set_xticks(list(x))
    ax.set_xticklabels(week_df['Date'], rotation=15)
    ax.set_ylabel('Units Sold')
    ax.set_title(f'7-Day Demand Forecast — {model_choice} | Starting {target_date.strftime("%b %d, %Y")}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Table
    st.dataframe(
        week_df.style.highlight_max(subset=['Forecast'], color='lightgreen')
                     .highlight_min(subset=['Forecast'], color='lightyellow'),
        use_container_width=True
    )

    total = sum(week_preds)
    st.success(f"📦 **Total 7-Day Forecast: {total:,} units** across all days")

# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
st.divider()
st.subheader("🤖 Model Comparison")

col1, col2 = st.columns(2)

lgbm_pred = int(lgbm_model.predict(X)[0])

with col1:
    st.markdown("### LightGBM")
    st.metric("Prediction", f"{lgbm_pred:,} units")
    st.metric("MAPE", "6.70%")
    st.metric("MAE", "201 units/day")
    st.progress(0.933, text="93.3% accurate")

with col2:
    if xgb_model:
        xgb_pred = int(xgb_model.predict(X)[0])
        st.markdown("### XGBoost")
        st.metric("Prediction", f"{xgb_pred:,} units", delta=f"{xgb_pred - lgbm_pred:+,} vs LightGBM")
        st.metric("MAPE", "6.73%")
        st.metric("MAE", "205 units/day")
        st.progress(0.927, text="92.7% accurate")
    else:
        st.markdown("### XGBoost")
        st.info("XGBoost model not found in MLflow. Retrain with train.py to enable comparison.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with LightGBM + FastAPI + Streamlit | Data: M5 Walmart Forecasting Dataset | Store: CA_1 FOODS")