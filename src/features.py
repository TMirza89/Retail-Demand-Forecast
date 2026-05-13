import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
sales    = pd.read_csv('../data/sales_train_validation.csv')
calendar = pd.read_csv('../data/calendar.csv')
prices   = pd.read_csv('../data/sell_prices.csv')

# ── 2. FILTER TO CA_1 + FOODS ONLY ───────────────────────────────────────────
# Reason: FOODS is 75% of volume, most predictable, most relevant to retailers
print("Filtering to CA_1 FOODS...")
sales_ca1 = sales[
    (sales['store_id'] == 'CA_1') &
    (sales['cat_id'] == 'FOODS')
].copy()
print(f"Items: {len(sales_ca1)}")

# ── 3. MELT TO LONG FORMAT ────────────────────────────────────────────────────
day_cols = [c for c in sales_ca1.columns if c.startswith('d_')]

df = sales_ca1.melt(
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    value_vars=day_cols,
    var_name='d',
    value_name='sales'
)

# ── 4. MERGE CALENDAR ─────────────────────────────────────────────────────────
cal_cols = ['d', 'date', 'wday', 'month', 'year',
            'event_name_1', 'event_type_1',
            'event_name_2', 'event_type_2',
            'snap_CA']

df = df.merge(calendar[cal_cols], on='d', how='left')
df['date'] = pd.to_datetime(df['date'])
print(f"After calendar merge: {df.shape}")

# ── 5. MERGE PRICES ───────────────────────────────────────────────────────────
# Add week number to match price data (prices change weekly)
df['wm_yr_wk'] = df['d'].map(
    calendar.set_index('d')['wm_yr_wk']
)

df = df.merge(
    prices[['store_id', 'item_id', 'wm_yr_wk', 'sell_price']],
    on=['store_id', 'item_id', 'wm_yr_wk'],
    how='left'
)
print(f"After price merge: {df.shape}")

# ── 6. SORT — CRITICAL FOR LAG FEATURES ───────────────────────────────────────
# Must sort by item then date before computing lags
# Otherwise lag(7) for item A will grab item B's data
df = df.sort_values(['item_id', 'date']).reset_index(drop=True)

# ── 7. AGGREGATE TO DAILY STORE LEVEL ─────────────────────────────────────────
# We forecast total daily FOODS sales for CA_1
# Item-level is too granular for a portfolio project
print("Aggregating to daily store level...")
daily = df.groupby('date').agg(
    sales        = ('sales', 'sum'),
    avg_price    = ('sell_price', 'mean'),
    snap_CA      = ('snap_CA', 'first'),
    wday         = ('wday', 'first'),
    month        = ('month', 'first'),
    year         = ('year', 'first'),
    event_name_1 = ('event_name_1', 'first'),
    event_name_2 = ('event_name_2', 'first'),
).reset_index()

print(f"Daily shape: {daily.shape}")

# ── 8. HOLIDAY / CLOSURE FLAG ─────────────────────────────────────────────────
# Days with near-zero sales are store closures (Christmas etc.)
# Flag them so model doesn't treat them as real demand drops
daily['is_closed'] = (daily['sales'] < 500).astype(int)
print(f"Closure days found: {daily['is_closed'].sum()}")

# ── 9. EVENT FLAGS ────────────────────────────────────────────────────────────
# Binary flags — is there any event today?
daily['has_event'] = daily['event_name_1'].notna().astype(int)

# Specific high-impact events
sporting_events = ['SuperBowl', 'NBAFinalsStart', 'NBAFinalsEnd']
daily['is_sporting_event'] = daily['event_name_1'].isin(sporting_events).astype(int)

# ── 10. LAG FEATURES ──────────────────────────────────────────────────────────
# Lag = sales from N days ago
# WHY: yesterday's sales is the strongest single predictor of today's sales
# This is the discrete version of autoregressive modeling (AR process)
print("Building lag features...")
for lag in [7, 14, 28]:
    daily[f'lag_{lag}'] = daily['sales'].shift(lag)
    print(f"  lag_{lag} done")

# ── 11. ROLLING WINDOW FEATURES ───────────────────────────────────────────────
# Rolling mean = local trend smoother
# WHY: captures recent demand level without overfitting to single days
# Mathematically: (1/w) * sum(x_{t-1}, ..., x_{t-w}) — uniform convolution kernel
print("Building rolling features...")
for window in [7, 14, 28]:
    daily[f'rolling_mean_{window}'] = (
        daily['sales']
        .shift(1)                          # shift 1 to avoid data leakage
        .rolling(window=window, min_periods=1)
        .mean()
    )
    daily[f'rolling_std_{window}'] = (
        daily['sales']
        .shift(1)
        .rolling(window=window, min_periods=1)
        .std()
    )
    print(f"  rolling_{window} done")

# ── 12. PRICE FEATURES ────────────────────────────────────────────────────────
# Price change vs last week — captures promotional effects
daily['price_lag_7']    = daily['avg_price'].shift(7)
daily['price_change']   = daily['avg_price'] - daily['price_lag_7']
daily['price_pct_change'] = daily['price_change'] / daily['price_lag_7']

# ── 13. TIME FEATURES ─────────────────────────────────────────────────────────
# Cyclical encoding of day of week and month
# WHY: wday=1 and wday=7 are adjacent in time but far apart numerically
# Sin/cos encoding preserves cyclical distance — your Math brain will like this
print("Building time features...")
daily['day_of_week']  = daily['date'].dt.dayofweek
daily['day_of_month'] = daily['date'].dt.day
daily['quarter']      = daily['date'].dt.quarter
daily['is_weekend']   = (daily['day_of_week'] >= 5).astype(int)
daily['is_month_start'] = daily['date'].dt.is_month_start.astype(int)
daily['is_month_end']   = daily['date'].dt.is_month_end.astype(int)

# Cyclical encoding
daily['wday_sin'] = np.sin(2 * np.pi * daily['day_of_week'] / 7)
daily['wday_cos'] = np.cos(2 * np.pi * daily['day_of_week'] / 7)
daily['month_sin'] = np.sin(2 * np.pi * daily['month'] / 12)
daily['month_cos'] = np.cos(2 * np.pi * daily['month'] / 12)

# ── 14. TREND FEATURE ─────────────────────────────────────────────────────────
# Simple integer index — captures long-term store growth
daily['trend'] = np.arange(len(daily))

# ── 15. DROP ROWS WITH NaN LAGS ───────────────────────────────────────────────
# First 28 rows will have NaN lags — drop them
before = len(daily)
daily = daily.dropna(subset=[f'lag_{l}' for l in [7, 14, 28]]).reset_index(drop=True)
after = len(daily)
print(f"Dropped {before - after} rows with NaN lags")

# ── 16. SAVE ──────────────────────────────────────────────────────────────────
output_path = '../data/features.csv'
daily.to_csv(output_path, index=False)
print(f"\nFeatures saved to {output_path}")

# ── 17. SUMMARY ───────────────────────────────────────────────────────────────
print("\n========== FEATURE SUMMARY ==========")
print(f"Shape             : {daily.shape}")
print(f"Date range        : {daily['date'].min()} → {daily['date'].max()}")
print(f"Features built    : {daily.shape[1]} columns")
print(f"Closure days      : {daily['is_closed'].sum()}")
print(f"Event days        : {daily['has_event'].sum()}")
print(f"Avg SNAP sales    : {daily[daily['snap_CA']==1]['sales'].mean():.0f}")
print(f"Avg non-SNAP sales: {daily[daily['snap_CA']==0]['sales'].mean():.0f}")
print("=====================================")
print("\nAll features ready. Proceed to train.py")
print("\nFeature list:")
for col in daily.columns:
    print(f"  {col}")