import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import os
import time

def fetch_data(ticker, period="1d", interval="1m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError("No price data found. Ticker may be invalid or delisted.")
        return df
    except Exception as e:
        st.error(f"[Error] {e}")
        return pd.DataFrame()

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(prices, window=20, num_std=2):
    ma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, lower

def detect_pattern(prices):
    if len(prices) < 6:
        return None
    p = prices.iloc
    try:
        if (
            p[-5] > p[-4] and
            p[-3] < p[-2] and
            p[-2] > p[-1]
        ):
            return "W"
        elif (
            p[-5] < p[-4] and
            p[-3] > p[-2] and
            p[-2] < p[-1]
        ):
            return "M"
        elif list(p[-5:]) == sorted(p[-5:]):
            return "UPTREND"
        elif list(p[-5:]) == sorted(p[-5:], reverse=True):
            return "DOWNTREND"
    except:
        return None
    return None

def detect_volume_breakout(volumes):
    if len(volumes) < 20:
        return False
    return volumes.iloc[-1] > 2 * volumes.iloc[-20:-1].mean()

def get_signal(pattern, vol_breakout):
    pattern = str(pattern) if pattern is not None else ""
    if pattern in ["W", "UPTREND"] and vol_breakout:
        return "STRONG BUY"
    elif pattern in ["W", "UPTREND"]:
        return "BUY"
    elif pattern in ["M", "DOWNTREND"] and vol_breakout:
        return "STRONG SELL"
    elif pattern in ["M", "DOWNTREND"]:
        return "SELL"
    return "HOLD"

def plot_chart(ticker, df, show_rsi, show_boll):
    closes = df["Close"]
    volumes = df["Volume"]
    times = df.index

    x = np.arange(len(closes))
    y = closes.values
    m, b = np.polyfit(x, y, 1)
    trend = m * x + b

    pattern = detect_pattern(closes)
    vol_break = detect_volume_breakout(volumes)
    signal = get_signal(pattern, vol_break)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(times, closes, label="Price", color="blue", linewidth=2)
    ax1.plot(times, trend, linestyle="--", color="orange", label="Trend")

    if "BUY" in signal:
        ax1.axhline(closes.iloc[-1], color="green", linestyle=":", label=signal)
    elif "SELL" in signal:
        ax1.axhline(closes.iloc[-1], color="red", linestyle=":", label=signal)

    if show_boll:
        upper, lower = compute_bollinger_bands(closes)
        ax1.plot(times, upper, linestyle="--", color="gray", alpha=0.5, label="Bollinger Upper")
        ax1.plot(times, lower, linestyle="--", color="gray", alpha=0.5, label="Bollinger Lower")

    ax1.set_title(f"{ticker} • Pattern: {pattern or 'None'} • Signal: {signal}", fontsize=14)
    ax1.set_ylabel("Price (USD)")
    ax1.set_xlabel("Time")
    ax1.legend()
    ax1.grid(True)

    if show_rsi:
        rsi = compute_rsi(closes)
        ax2 = ax1.twinx()
        ax2.plot(times, rsi, color="purple", alpha=0.5, label="RSI")
        ax2.axhline(70, color="red", linestyle="--", alpha=0.3)
        ax2.axhline(30, color="green", linestyle="--", alpha=0.3)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI")
        ax2.legend(loc="upper right")

    st.pyplot(fig)

# Streamlit UI
st.title("📊 Live Stock Chart Viewer")

symbol = st.text_input("Enter stock ticker symbol:", value="AAPL").upper()
show_rsi = st.checkbox("Show RSI", value=True)
show_boll = st.checkbox("Show Bollinger Bands", value=True)
refresh_rate = st.slider("Refresh every N minutes", min_value=1, max_value=5, value=1)

if st.button("Start Chart"):
    while True:
        df = fetch_data(symbol)
        if not df.empty:
            plot_chart(symbol, df, show_rsi, show_boll)
        time.sleep(refresh_rate * 60)
        st.experimental_rerun()
