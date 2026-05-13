import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
sales    = pd.read_csv('../data/sales_train_validation.csv')
calendar = pd.read_csv('../data/calendar.csv')
prices   = pd.read_csv('../data/sell_prices.csv')

print(f"Sales shape    : {sales.shape}")
print(f"Calendar shape : {calendar.shape}")
print(f"Prices shape   : {prices.shape}")

# ── 2. UNDERSTAND THE DATA ────────────────────────────────────────────────────
print("\n--- Sales Sample ---")
print(sales.head())
print("\nStores    :", sales['store_id'].unique())
print("States    :", sales['state_id'].unique())
print("Categories:", sales['cat_id'].unique())

# ── 3. FOCUS ON ONE STORE ─────────────────────────────────────────────────────
# CA_1 is the largest California store — realistic and defensible in interviews
STORE = 'CA_1'
sales_ca1 = sales[sales['store_id'] == STORE].copy()
print(f"\nItems in {STORE}: {len(sales_ca1)}")

# ── 4. MELT TO LONG FORMAT ────────────────────────────────────────────────────
# M5 data is wide — one column per day (d_1, d_2 ... d_1913)
# We melt it to long format: one row per item per day
day_cols = [c for c in sales_ca1.columns if c.startswith('d_')]

sales_long = sales_ca1.melt(
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    value_vars=day_cols,
    var_name='d',
    value_name='sales'
)
print(f"\nLong format shape: {sales_long.shape}")

# ── 5. MERGE WITH CALENDAR ────────────────────────────────────────────────────
sales_long = sales_long.merge(
    calendar[['d', 'date', 'wday', 'month', 'year',
              'event_name_1', 'event_type_1', 'snap_CA']],
    on='d',
    how='left'
)
sales_long['date'] = pd.to_datetime(sales_long['date'])
print("\nAfter calendar merge:")
print(sales_long.head())

# ── 6. TOTAL DAILY SALES ──────────────────────────────────────────────────────
print("\nPlotting daily sales...")
daily_sales = sales_long.groupby('date')['sales'].sum().reset_index()

plt.figure(figsize=(16, 5))
plt.plot(daily_sales['date'], daily_sales['sales'], linewidth=0.8, color='steelblue')
plt.title(f'Total Daily Sales — Store {STORE}', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.tight_layout()
plt.savefig('../data/daily_sales.png', dpi=150)
plt.show()

# ── 7. SALES BY CATEGORY ──────────────────────────────────────────────────────
print("Plotting category sales...")
cat_sales = sales_long.groupby(['date', 'cat_id'])['sales'].sum().reset_index()

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
for ax, cat in zip(axes, ['FOODS', 'HOBBIES', 'HOUSEHOLD']):
    data = cat_sales[cat_sales['cat_id'] == cat]
    ax.plot(data['date'], data['sales'], linewidth=0.8)
    ax.set_title(f'{cat} Sales')
    ax.set_ylabel('Units')
plt.tight_layout()
plt.savefig('../data/category_sales.png', dpi=150)
plt.show()

# ── 8. DAY OF WEEK PATTERN ────────────────────────────────────────────────────
print("Plotting day of week pattern...")
dow_sales = sales_long.groupby('wday')['sales'].mean().reset_index()
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.figure(figsize=(8, 4))
sns.barplot(data=dow_sales, x='wday', y='sales', palette='Blues_d')
plt.xticks(ticks=range(7), labels=dow_labels)
plt.title('Average Sales by Day of Week')
plt.ylabel('Avg Units Sold')
plt.tight_layout()
plt.savefig('../data/dow_sales.png', dpi=150)
plt.show()

# ── 9. SNAP DAY IMPACT ────────────────────────────────────────────────────────
print("Plotting SNAP day impact...")
snap_sales = sales_long.groupby('snap_CA')['sales'].mean().reset_index()
snap_sales['snap_CA'] = snap_sales['snap_CA'].map({0: 'No SNAP', 1: 'SNAP Day'})

plt.figure(figsize=(6, 4))
sns.barplot(data=snap_sales, x='snap_CA', y='sales', palette='Set2')
plt.title('Sales on SNAP vs Non-SNAP Days\n(SNAP = Food Stamp benefit days)')
plt.ylabel('Avg Units Sold')
plt.tight_layout()
plt.savefig('../data/snap_sales.png', dpi=150)
plt.show()

snap_val   = snap_sales[snap_sales['snap_CA'] == 'SNAP Day']['sales'].values[0]
no_snap_val = snap_sales[snap_sales['snap_CA'] == 'No SNAP']['sales'].values[0]
snap_lift  = (snap_val - no_snap_val) / no_snap_val * 100
print(f"\nSNAP lift: {snap_lift:.1f}%")

# ── 10. MONTHLY TREND ─────────────────────────────────────────────────────────
print("Plotting monthly trend...")
sales_long['year_month'] = sales_long['date'].dt.to_period('M')
monthly = sales_long.groupby('year_month')['sales'].sum().reset_index()
monthly['year_month'] = monthly['year_month'].astype(str)

plt.figure(figsize=(16, 4))
plt.plot(monthly['year_month'], monthly['sales'], marker='o', markersize=3, linewidth=0.8, color='coral')
plt.xticks(rotation=90, fontsize=7)
plt.title('Monthly Sales Trend')
plt.ylabel('Total Units Sold')
plt.tight_layout()
plt.savefig('../data/monthly_trend.png', dpi=150)
plt.show()

# ── 11. SUMMARY STATS ─────────────────────────────────────────────────────────
print("\n========== EDA SUMMARY ==========")
print(f"Date range     : {sales_long['date'].min()} → {sales_long['date'].max()}")
print(f"Total records  : {len(sales_long):,}")
print(f"Unique items   : {sales_long['item_id'].nunique()}")
print(f"Avg daily sales: {daily_sales['sales'].mean():.0f} units")
print(f"Peak day sales : {daily_sales['sales'].max():.0f} units")
print(f"SNAP lift      : {snap_lift:.1f}%")
print("==================================")
print("\nAll plots saved to /data folder.")
print("EDA complete. Ready for feature engineering.")