
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import seaborn as sns
import matplotlib.pyplot as plt

# Auto-refresh every 5 minutes
st_autorefresh(interval=300000, key="datarefresh")

fno_stocks_sector = {
    "RELIANCE.NS":"OIL","INFY.NS":"IT","TCS.NS":"IT","HDFCBANK.NS":"BANKING",
    "ICICIBANK.NS":"BANKING","SBIN.NS":"BANKING","LT.NS":"AUTO","ITC.NS":"CONSUMER",
    "AXISBANK.NS":"BANKING","KOTAKBANK.NS":"BANKING","HCLTECH.NS":"IT",
    "BHARTIARTL.NS":"TELECOM","BAJAJ-AUTO.NS":"AUTO","MARUTI.NS":"AUTO","ONGC.NS":"OIL"
}

PRICE_BREAKOUT_DAYS = 2
VOL_MULTIPLIER = 1.5
STOP_LOSS_PCT = 0.02
TARGET_PCT = 0.04

@st.cache
def get_historical(stock):
    df = yf.download(stock, period="20d", interval="1d")
    df["Avg_Volume"] = df["Volume"].rolling(20).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Demand_Zone"] = df["Low"].rolling(3).min()
    df["Supply_Zone"] = df["High"].rolling(3).max()
    df["Open"] = df["Open"]
    return df

def scan_stocks(fno_stocks_sector):
    long_candidates = []
    short_candidates = []
    heatmap_data = []

    for stock, sector in fno_stocks_sector.items():
        df = get_historical(stock)
        latest = df.iloc[-1]
        prev_high_2 = df["High"].iloc[-PRICE_BREAKOUT_DAYS]
        prev_low_2 = df["Low"].iloc[-PRICE_BREAKOUT_DAYS]
        avg_vol = latest["Avg_Volume"]
        price = latest["Close"]
        volume = latest["Volume"]
        ema20 = latest["EMA20"]
        dz = latest["Demand_Zone"]
        sz = latest["Supply_Zone"]
        open_price = latest["Open"]

        volume_surge = volume / avg_vol
        percent_change = (price - open_price) / open_price * 100
        heatmap_data.append({"Stock": stock, "Sector": sector, "Percent_Change": percent_change})

        if price > prev_high_2 and volume_surge > VOL_MULTIPLIER and price > ema20:
            score = 0.4*((price-prev_high_2)/prev_high_2) + 0.3*(volume_surge-1)
            long_candidates.append((stock, price, dz, score, percent_change, sector))

        if price < prev_low_2 and volume_surge > VOL_MULTIPLIER and price < ema20:
            score = 0.4*((prev_low_2-price)/prev_low_2) + 0.3*(volume_surge-1)
            short_candidates.append((stock, price, sz, score, percent_change, sector))

    long_candidates = sorted(long_candidates, key=lambda x: x[3], reverse=True)[:10]
    short_candidates = sorted(short_candidates, key=lambda x: x[3], reverse=True)[:10]
    heatmap_df = pd.DataFrame(heatmap_data).pivot("Sector", "Stock", "Percent_Change").fillna(0)
    return long_candidates, short_candidates, heatmap_df

st.title("🔥 F&O Momentum Scanner with Sector Heatmap 🔥")
st.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

longs, shorts, heatmap_df = scan_stocks(fno_stocks_sector)

st.subheader("Top 10 Long Candidates")
for s, price, dz, score, pc, sector in longs:
    sl = round(price * (1 - STOP_LOSS_PCT),2)
    tgt = round(price * (1 + TARGET_PCT),2)
    st.write(f"{s} | Sector: {sector} | Price: {price:.2f} | Demand Zone: {dz:.2f} | Stop: {sl} | Target: {tgt} | Score: {score:.2f} | %Change: {pc:.2f}%")

st.subheader("Top 10 Short Candidates")
for s, price, sz, score, pc, sector in shorts:
    sl = round(price * (1 + STOP_LOSS_PCT),2)
    tgt = round(price * (1 - TARGET_PCT),2)
    st.write(f"{s} | Sector: {sector} | Price: {price:.2f} | Supply Zone: {sz:.2f} | Stop: {sl} | Target: {tgt} | Score: {score:.2f} | %Change: {pc:.2f}%")

st.subheader("Sector-wise Heatmap of % Price Change from Open")
fig, ax = plt.subplots(figsize=(16,8))
sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.5, linecolor="gray")
st.pyplot(fig)
