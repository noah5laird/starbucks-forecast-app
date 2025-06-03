import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import statsmodels.api as sm
import plotly.graph_objects as go

st.title("Starbucks Revenue Forecasting")
st.markdown(
    """
    ### 🎯 Thesis Statement  
    There is evidence to suggest that Starbucks' reported revenue may be **overstated**, particularly in fiscal year 2023. Forecasts derived from ARIMAX modeling — which incorporate marketing spend and inflation — indicate that revenue trends do not align with operational inputs or macroeconomic indicators, raising audit concerns over the validity of recent figures. Furthermore, consumer sentiment suggest slightly negative market views of starbucks, directly conflicting with Starbucks' alarming spike in revenue.
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_csv("merged_with_cpi_with_dates.csv")

def load_actual_data():
    return pd.read_csv("merged_with_cpi_with_datesACTUAL.csv")

actual_data = load_actual_data()
data = load_data()

actual_data["date"] = pd.date_range(start="2018-03-31", periods=len(actual_data), freq="QE")
actual_data.set_index("date", inplace=True)

# --- Live CPI from FRED ---
def get_latest_cpi():
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "CPALTT01USQ657N",
        "api_key": "464890becc3d3b822c960913019c586d",
        "file_type": "json",
        "observation_start": "2023-01-01"
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        observations = r.json()["observations"]
        latest = float(observations[-1]['value'])
        return latest
    return None

# --- User Input ---
st.sidebar.header("User Input")

# Fetch latest CPI
live_cpi = get_latest_cpi()
if live_cpi:
    st.sidebar.markdown(f"📊 **Live CPI from FRED:** {live_cpi:.2f}%")
else:
    st.sidebar.warning("⚠️ Could not fetch live CPI data.")

# Input sliders
user_cpi = st.sidebar.slider("Expected CPI Growth (%)", -3.0, 3.0, 0.0)
user_marketing_pct_change = st.sidebar.slider("Projected Marketing Spend Change (%) from Q4 2022", -10.0, 10.0, 0.0)

def run_forecast(data, future_cpi, pct_change):
    df = data.copy()
    df["date"] = pd.date_range(start="2018-03-31", periods=len(df), freq="QE")
    df = df[df["date"] <= "2022-12-31"]
    df = df.set_index("date")

    exog = df[["CPI", "marketing_spend"]]
    model = SARIMAX(df["revenue"], exog=exog, order=(1, 1, 1)).fit(disp=False)

    last_marketing_spend = actual_data["marketing_spend"].iloc[-1]
    projected_q1_2023_marketing_spend = last_marketing_spend * (1 + pct_change / 100)

    marketing_growth_rate = 0.02
    future_marketings_series = [
        projected_q1_2023_marketing_spend * ((1 + marketing_growth_rate) ** i) for i in range(8)
    ]

    future_exog = pd.DataFrame({
        "CPI": [future_cpi] * 8,
        "marketing_spend": future_marketings_series
    })

    forecast = model.get_forecast(steps=8, exog=future_exog)
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return df["revenue"], forecast_values, conf_int

# --- Run the Forecast ---
actuals, forecasted, conf_int = run_forecast(data, user_cpi, user_marketing_pct_change)

# --- Plot ---
st.subheader("Revenue Forecast vs. Historical Data (ARIMAX)")
fig, ax = plt.subplots()

forecast_index = pd.date_range(start=actuals.index[-1] + pd.offsets.QuarterEnd(1), periods=8, freq="QE")

ax.plot(actual_data.index, actual_data["revenue"], label="Actual Revenue", color='black')
ax.plot(forecast_index, forecasted, label="Forecasted Revenue (2023–2024)", linestyle="--", color='green')

# Add confidence interval shading
ax.fill_between(forecast_index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1],
                color='gray',
                alpha=0.3,
                label='95% Confidence Interval')

ax.set_xlabel("Year")
ax.set_ylabel("Revenue (in millions)")
ax.legend()
st.pyplot(fig)

# --- Static AI Summary ---
st.subheader("AI-Generated Summary")

latest = actuals.iloc[-1]
forecasted_val = forecasted.iloc[-1]

summary_text = (
    "Based on the ARIMAX model, which incorporates CPI and marketing spend as predictors, Starbucks’ revenue forecast shows "
    "a highly moderate projected growth in marketing spend through 2023 and 2024. This trend may indicate a potential revenue "
    "overstatement in the most recent periods—particularly given the anomalous spike observed in 2023. While CPI expectations "
    "are modest, the model's reaction to increased marketing costs suggests diminishing returns or inflated revenue recognition. "
    "Auditors should scrutinize the assumptions underlying revenue growth and assess whether reported figures reflect sustainable "
    "business activity."
)

st.write(summary_text)
# --- Expenses Insight with Simple Regression ---

df1 = data.copy()
df1["date"] = pd.date_range(start="2018-03-31", periods=len(df1), freq="QE")
df1.set_index("date", inplace=True)

# --- Expenses Insight with Simple Regression ---
st.markdown("### 💸 Expenses vs Revenue (Regression Model)")

X = sm.add_constant(df1['expenses'])
y = df1['revenue']
model_exp = sm.OLS(y, X).fit()
predicted_revenue = model_exp.predict(X)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df1.index, y=df1['expenses'], name="Expenses", mode="lines+markers"))
fig2.add_trace(go.Scatter(x=df1.index, y=df1['revenue'], name="Actual Revenue", mode="lines+markers"))
fig2.add_trace(go.Scatter(x=df1.index, y=predicted_revenue, name="Predicted Revenue", mode="lines"))

fig2.update_layout(
    title="Revenue vs Expenses",
    xaxis=dict(title="Date", tickformat="%Y"),  # <-- Formats ticks as Year
    yaxis=dict(title="Value"),
    legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(fig2, use_container_width=True)


st.info("This regression chart shows how well Starbucks' expenses align with revenue. If actual revenue significantly deviates from predicted revenue based on expenses, it may indicate a misstatement or unusual revenue recognition. Strong alignment supports revenue and expense correlation.")


# --- Benchmark Comparison ---
st.subheader("Benchmark Comparison: Revenue Growth")

sb_growth = (forecasted_val - latest) / latest
peer_annual_cagr = 0.054
peer_quarterly_growth = (1 + peer_annual_cagr) ** (1 / 4) - 1  # ≈ 1.32%

col1, col2 = st.columns(2)
col1.metric("Starbucks Forecasted Growth", f"{sb_growth:.2%}")
col2.metric("Coffee Industry Avg (Qtrly)", f"{peer_quarterly_growth:.2%}")

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
    "Further Downside Brewing for Starbucks (SBUX)?",
    "Starbucks shares fall despite strong Q3 performance",
    "Starbucks' China Sales Soar, but Revenue and Same-marketing Sales Miss Forecasts"
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
