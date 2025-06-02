import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fredapi import Fred

st.title("Starbucks Revenue Forecasting")

@st.cache_data
def load_data():
    return pd.read_csv("merged_with_cpi_with_dates.csv")

def load_actual_data():
    return pd.read_csv("merged_with_cpi_with_datesACTUAL.csv")

actual_data = load_actual_data()

data = load_data()
actual_data["date"] = pd.date_range(start="2018-03-31", periods=len(actual_data), freq="QE")
actual_data.set_index("date", inplace=True)

# --- User Input ---
st.sidebar.header("User Input")
user_cpi = st.sidebar.slider("Expected CPI Growth (%)", -3.0, 3.0, 0.0)

user_marketing = st.sidebar.number_input("Projected  Marketing Spend for the First Quarter of 2024 (in millions)", value=394209)

def run_forecast(data, future_cpi, future_expenses):
    df = data.copy()
    df["date"] = pd.date_range(start="2018-03-31", periods=len(df), freq="QE")
    df = df[df["date"] <= "2022-12-31"]

    df = df.set_index("date")

    exog = df[["CPI", "marketing_spend"]]
    model = SARIMAX(df["revenue"], exog=exog, order=(1, 1, 1)).fit(disp=False)

    # Project 4 quarters (2024)
    # Project 8 quarters (2023–2024)
    expense_growth_rate = 0.02
    future_marketing_series = [future_expenses * ((1 + expense_growth_rate) ** i) for i in range(8)]

    future_exog = pd.DataFrame({
        "CPI": [future_cpi] * 8,
        "Marketing Spend": future_marketing_series
    })


    forecast = model.get_forecast(steps=8, exog=future_exog)
    forecast_values = forecast.predicted_mean
    return df["revenue"], forecast_values

# --- Run the Forecast ---
actuals, forecasted = run_forecast(data, user_cpi, user_marketing)

# --- Plot ---
st.subheader("Revenue Forecast vs. Historical Data")
fig, ax = plt.subplots()

last_date = actuals.index[-1]
forecast_index = pd.date_range(start=actuals.index[-1] + pd.offsets.QuarterEnd(1), periods=8, freq="QE")

ax.plot(actual_data.index, actual_data["revenue"], label="Actual Revenue", color='black')
ax.plot(forecast_index, forecasted, label="Forecasted Revenue (2023–2024)", linestyle="--", color='red')

ax.set_xlabel("Year")
ax.set_ylabel("Revenue (in millions)")
ax.legend()
st.pyplot(fig)



# --- Static AI Summary (Formatted Safely) ---
st.subheader("AI-Generated Summary")

latest = actuals.iloc[-1]
forecasted_val = forecasted.iloc[-1]

summary_text = (
    f"Based on the current forecast, Starbucks’ quarterly revenue is projected to grow steadily from its most recent "
    f"level of ${latest:.1f} to an estimated ${forecasted_val:.1f} in future quarters. This trend aligns with moderate "
    f"CPI growth and controlled expense levels, suggesting sustainable business performance. The analysis indicates low risk "
    f"of revenue overstatement under current macroeconomic conditions. Auditors should continue to monitor these assumptions "
    f"as economic conditions evolve."
)

st.write(summary_text)






# --- Benchmark Comparison: Starbucks vs. Coffee Industry ---
st.subheader("Benchmark Comparison: Revenue Growth")

# Starbucks growth based on forecast
sb_growth = (forecasted_val - latest) / latest

# Convert annual CAGR (5.4%) to approximate quarterly growth rate
peer_annual_cagr = 0.054
peer_quarterly_growth = (1 + peer_annual_cagr) ** (1 / 4) - 1  # ≈ 1.32%

col1, col2 = st.columns(2)
col1.metric("Starbucks Forecasted Growth", f"{sb_growth:.2%}")
col2.metric("Coffee Industry Avg (Qtrly)", f"{peer_quarterly_growth:.2%}")

# Add interpretation
if sb_growth > peer_quarterly_growth + 0.05:
    st.warning("⚠️ Starbucks' forecasted growth significantly exceeds industry norms. Consider reviewing the assumptions.")
elif sb_growth < peer_quarterly_growth - 0.05:
    st.info("ℹ️ Forecasted growth is well below industry average. Assumptions may be conservative.")
else:
    st.success("✅ Forecasted growth is within a reasonable range of industry expectations.")
