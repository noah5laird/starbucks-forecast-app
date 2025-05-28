import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fredapi import Fred

st.title("Starbucks Revenue Forecasting App")

@st.cache_data
def load_data():
    return pd.read_csv("merged_with_cpi_with_dates.csv")

data = load_data()

# --- User Input ---
st.sidebar.header("User Input")
user_cpi = st.sidebar.slider("Expected CPI Growth (%)", 1.0, 5.0, 3.0)
user_expenses = st.sidebar.number_input("Projected Quarterly Expenses (in millions)", value=6000.0)

# --- Forecast Function ---
def run_forecast(data, future_cpi, future_expenses):
    df = data.copy()
    df["date"] = pd.date_range(start="2018-03-31", periods=len(df), freq="QE")
    df = df.set_index("date")

    exog = df[["CPI", "expenses"]]
    model = SARIMAX(df["revenue"], exog=exog, order=(1, 1, 1)).fit(disp=False)

    future_exog = pd.DataFrame({
        "CPI": [future_cpi] * 4,
        "expenses": [future_expenses] * 4
    })

    forecast = model.get_forecast(steps=4, exog=future_exog)
    forecast_values = forecast.predicted_mean
    return df["revenue"], forecast_values

# --- Run the Forecast ---
actuals, forecasted = run_forecast(data, user_cpi, user_expenses)

# --- Plot ---
st.subheader("Revenue Forecast vs. Historical Data")
fig, ax = plt.subplots()
last_date = actuals.index[-1]
forecast_index = pd.date_range(start=last_date + pd.offsets.QuarterEnd(1), periods=4, freq="QE")

ax.plot(actuals.index, actuals.values, label="Actual Revenue")
ax.plot(forecast_index, forecasted, label="Forecasted Revenue", linestyle="--")
ax.set_xlabel("Quarter")
ax.set_ylabel("Revenue (in millions)")
ax.legend()
st.pyplot(fig)

# --- Static AI Summary ---
st.subheader("AI-Generated Summary (Simulated)")

summary_text = (
    "Based on the current forecast, Starbucks’ quarterly revenue is projected to grow steadily from its most recent "
    f"level of ${actuals.iloc[-1]:,.1f}  to approximately  ${forecasted.iloc[-1]:,.1f} in future quarters. This trend aligns "
    "with moderate CPI growth and controlled expense levels, suggesting sustainable business performance. The analysis "
    "indicates low risk of revenue overstatement under current macroeconomic conditions. Auditors should continue to monitor "
    "these assumptions as economic conditions evolve."
)

st.write(summary_text)
