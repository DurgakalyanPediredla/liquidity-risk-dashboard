# Liquidity Risk Reporting Dashboard – Streamlit Version (Production Enhanced)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import datetime
import warnings
warnings.filterwarnings("ignore")

# ---------------------- Simulated Data Generation ----------------------
np.random.seed(42)
currency_list = ['USD', 'EUR', 'GBP']
maturity_buckets = ['0-7D', '8-30D', '31-90D']
n = 60

df = pd.DataFrame({
    'Date': pd.date_range(start='2024-12-01', periods=n, freq='D'),
    'Currency': np.random.choice(currency_list, n),
    'HQLA_Level1A': np.random.normal(350, 20, n),
    'HQLA_Level1B': np.random.normal(200, 15, n),
    'HQLA_Level2A': np.random.normal(150, 10, n),
    'HQLA_Level2B': np.random.normal(70, 8, n),
    'Expected_Inflows': np.random.normal(360, 25, n),
    'Expected_Outflows': np.random.normal(500, 30, n),
    'Stable_Funding': np.random.normal(1100, 50, n),
    'Required_Funding': np.random.normal(900, 40, n),
    'Contingent_Buffer': np.random.normal(150, 20, n),
    'Top5_Counterparty_Exposure': np.random.uniform(200, 400, n),
    'Unencumbered_Assets': np.random.normal(300, 30, n),
    'Undrawn_Credit_Lines': np.random.normal(250, 20, n),
    'Maturity_Bucket': np.random.choice(maturity_buckets, n)
})

# KPI Calculations
df['Total_HQLA'] = df[['HQLA_Level1A', 'HQLA_Level1B', 'HQLA_Level2A', 'HQLA_Level2B']].sum(axis=1)
df['Net_Cash_Outflows'] = df['Expected_Outflows'] - df['Expected_Inflows']
df['LCR'] = (df['Total_HQLA'] / df['Net_Cash_Outflows']) * 100
df['NSFR'] = (df['Stable_Funding'] / df['Required_Funding']) * 100

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Liquidity Risk Dashboard", layout="wide")
st.title("Liquidity Risk Dashboard – Basel III / CRR Production Enhanced")

scenario = st.sidebar.selectbox("Select Stress Scenario", ['Base Case', 'Adverse', 'Severely Adverse'])
currency = st.sidebar.multiselect("Currency Filter", df['Currency'].unique(), default=df['Currency'].unique())
maturity = st.sidebar.multiselect("Maturity Buckets", df['Maturity_Bucket'].unique(), default=df['Maturity_Bucket'].unique())

filtered_df = df[(df['Currency'].isin(currency)) & (df['Maturity_Bucket'].isin(maturity))]

if scenario == 'Adverse':
    filtered_df['Expected_Outflows'] *= 1.15
    filtered_df['Total_HQLA'] *= 0.90
elif scenario == 'Severely Adverse':
    filtered_df['Expected_Outflows'] *= 1.30
    filtered_df['Total_HQLA'] *= 0.80

filtered_df['Net_Cash_Outflows'] = filtered_df['Expected_Outflows'] - filtered_df['Expected_Inflows']
filtered_df['LCR'] = (filtered_df['Total_HQLA'] / filtered_df['Net_Cash_Outflows']) * 100

# KPI Cards
col1, col2, col3 = st.columns(3)
col1.metric("Average LCR", f"{filtered_df['LCR'].mean():.2f}%")
col2.metric("Average NSFR", f"{filtered_df['NSFR'].mean():.2f}%")
col3.metric("Avg Counterparty Exposure", f"{filtered_df['Top5_Counterparty_Exposure'].mean():.2f}M")

# Unencumbered & Contingency Buffer
st.markdown("### Contingency Liquidity & Buffer Assets")
buf1, buf2 = st.columns(2)
buf1.metric("Available Unencumbered Assets", f"{filtered_df['Unencumbered_Assets'].mean():.2f}M")
buf2.metric("Undrawn Committed Credit Lines", f"{filtered_df['Undrawn_Credit_Lines'].mean():.2f}M")

# Charts
st.plotly_chart(px.line(filtered_df, x='Date', y='LCR', title='LCR Over Time'))
st.plotly_chart(px.line(filtered_df, x='Date', y='NSFR', title='NSFR Over Time'))

# Maturity Ladder
st.subheader("Maturity Ladder Cash Flow")
st.dataframe(filtered_df.groupby('Maturity_Bucket')[['Expected_Inflows', 'Expected_Outflows']].mean().round(2))

# Forecasting
st.subheader("LCR Forecast (ARIMA Model)")
try:
    lcr_series = filtered_df[['Date', 'LCR']].copy()
    lcr_series.set_index('Date', inplace=True)
    model = ARIMA(lcr_series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)
    forecast_dates = [lcr_series.index[-1] + datetime.timedelta(days=i+1) for i in range(10)]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted_LCR': forecast})
    st.plotly_chart(px.line(forecast_df, x='Date', y='Forecasted_LCR', title='LCR Forecast - Next 10 Days'))
except Exception as e:
    st.warning(f"Forecast not available: {e}")

# Alerts
st.subheader("Liquidity Alerts & Commentary")
avg_lcr = filtered_df['LCR'].mean()
if avg_lcr >= 120:
    st.success("✅ Excellent liquidity resilience.")
elif avg_lcr >= 100:
    st.info("ℹ️ Compliant – Stable liquidity conditions.")
elif avg_lcr >= 90:
    st.warning("⚠️ Close to stress threshold – review needed.")
else:
    st.error("❌ Critical – Activate contingency funding plans!")

# Export
st.download_button("📥 Download Report as CSV", data=filtered_df.to_csv(index=False), file_name="Liquidity_Report.csv")
