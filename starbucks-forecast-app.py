import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fredapi import Fred
import openai

# Configuration
FRED_API_KEY = '18b5149ec21c04e0b38290b1de865d0b'
OPENAI_API_KEY = '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'
fred = Fred(api_key=FRED_API_KEY)
openai.api_key = OPENAI_API_KEY

st.title("Starbucks Revenue Forecasting App")

# --- Load your data (replace this with actual data path) ---
@st.cache_data
def load_data():
    return pd.read_csv("merged_with_cpi_with_dates.csv")  # must match Jupyter output

data = load_data()

# --- User Input Section ---
st.sidebar.header("User Input")
user_cpi = st.sidebar.slider("Expected CPI Growth (%)", 1.0, 5.0, 3.0)
user_expenses = st.sidebar.number_input("Projected Quarterly Expenses (in millions)", value=6000.0)

# --- Create Exogenous Variable Dataframe ---
def prepare_exog(df):
    return df[["CPI", "StoreCount"]]  # or new variable

# --- Fit ARIMAX Model ---
def run_forecast(data, future_cpi, future_expenses):
    df = data.copy()
    df["date"] = pd.date_range(start="2018-03-31", periods=len(df), freq="Q")
    df = df.set_index("date")

    exog = df[["CPI", "expenses"]]  # Replace store_count with expenses
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

# Create future dates for the forecast based on the last actual date
last_date = actuals.index[-1]
forecast_index = pd.date_range(start=last_date + pd.offsets.QuarterEnd(1), periods=4, freq="Q")

# Plot actuals and forecast
ax.plot(actuals.index, actuals.values, label="Actual Revenue")
ax.plot(forecast_index, forecasted, label="Forecasted Revenue", linestyle="--")
ax.set_xlabel("Quarter")
ax.set_ylabel("Revenue (in millions)")
ax.legend()
st.pyplot(fig)


# --- AI Summary ---
def generate_summary(actuals, forecast):
    summary_text = f"Revenue grew from ${actuals.iloc[-1]:,.0f} to an estimated ${forecast.iloc[-1]:,.0f} under current assumptions. "
    prompt = f"Write a 75-word summary for an audit committee about this trend: {summary_text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=100
    )
    return response['choices'][0]['message']['content']

ai_summary = generate_summary(actuals, forecasted)
st.subheader("AI-Generated Summary")
st.write(ai_summary)
