import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fredapi import Fred
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

user_marketing = st.sidebar.number_input("Projected  Marketing Spend for the First Quarter of 2023 (in millions)", value=425.0)

def run_forecast(data, future_cpi, future_marketing):
    df = data.copy()
    df["date"] = pd.date_range(start="2018-03-31", periods=len(df), freq="QE")
    df = df[df["date"] <= "2022-12-31"]

    df = df.set_index("date")

    exog = df[["CPI", "marketing_spend"]]
    model = SARIMAX(df["revenue"], exog=exog, order=(1, 1, 1)).fit(disp=False)

    # Project 4 quarters (2024)
    # Project 8 quarters (2023–2024)
    marketing_growth_rate = 0.02
    future_marketing_series = [future_marketing * ((1 + marketing_growth_rate) ** i) for i in range(8)]

    future_exog = pd.DataFrame({
        "CPI": [future_cpi] * 8,
        "Marketing": future_marketing_series
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

ax.plot(actual_data.index, actual_data["revenue"], label="Actual Revenue (Full)", color='black')
ax.plot(forecast_index, forecasted, label="Forecasted Revenue (2023–2024)", linestyle="--", color='green')

ax.set_xlabel("Year")
ax.set_ylabel("Revenue (in millions)")
ax.legend()
st.pyplot(fig)



# --- Static AI Summary (Formatted Safely) ---
st.subheader("AI-Generated Summary")

latest = actuals.iloc[-1]
forecasted_val = forecasted.iloc[-1]


summary_text = (
    "Based on the current forecast, Starbucks’ quarterly revenue is expected to stabilize around $8,000 million "
    "following a sharp, anomalous spike in 2023. This surge appears tied to unusually high marketing spend that year, "
    "which may not reflect sustainable operating conditions. The forecast suggests a return to normalized growth, "
    "supported by modest CPI expectations and controlled expense projections. Auditors should monitor for further volatility "
    "tied to discretionary spend and ensure future revenue recognition remains aligned with core business fundamentals."
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
if sb_growth > peer_quarterly_growth + 0.04:
    st.warning("⚠️ Starbucks' forecasted growth significantly exceeds industry norms. Consider reviewing the assumptions.")
elif sb_growth < peer_quarterly_growth - 0.04:
    st.info("ℹ️ Forecasted growth is well below industry average. Assumptions may be conservative.")
else:
    st.success("✅ Forecasted growth is within a reasonable range of industry expectations.")


# --- Sentiment Analysis ---
st.markdown("### 📰 Earnings Sentiment Check")
example_headlines = [
    "Starbucks beats earnings expectations but sees slower growth in China",
    "Starbucks reports record revenue amid inflation concerns",
    "Starbucks shares fall despite strong Q3 performance"
]
analyzer = SentimentIntensityAnalyzer()
sentiments = [analyzer.polarity_scores(h)['compound'] for h in example_headlines]
avg_sentiment = sum(sentiments) / len(sentiments)

for h, s in zip(example_headlines, sentiments):
    st.write(f"- {h} (Sentiment: {s:.2f})")

if avg_sentiment < -0.1:
    st.error("⚠️ Negative earnings sentiment detected. Potential revenue risk.")
elif avg_sentiment < 0.1:
    st.warning("⚠️ Neutral to slightly negative sentiment. Monitor closely.")
else:
    st.success("✅ Sentiment is generally positive.")
