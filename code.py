import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
import time

# --- Data Functions ---
@st.cache_data
def fetch_data(ticker, period="1d", interval="1m"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError("No price data found. Ticker may be invalid or delisted.")
    return df

@st.cache_data
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@st.cache_data
def compute_bollinger(prices, window=20, num_std=2):
    ma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return ma + num_std * std, ma - num_std * std

# --- Session State ---
if 'started' not in st.session_state:
    st.session_state.started = False
if 'symbol' not in st.session_state:
    st.session_state.symbol = ""

# --- Input Screen ---
if not st.session_state.started:
    st.title("Investment Trend App")
    symbol_input = st.text_input("Enter stock ticker:", value="AAPL").upper()
    show_rsi_input = st.checkbox("Show RSI", value=True)
    show_boll_input = st.checkbox("Show Bollinger Bands", value=True)
    refresh_input = st.slider("Refresh every N minutes", 1, 5, 1)
    if st.button("Start Chart") and symbol_input:
        st.session_state.started   = True
        st.session_state.symbol    = symbol_input
        st.session_state.show_rsi  = show_rsi_input
        st.session_state.show_boll = show_boll_input
        st.session_state.refresh   = refresh_input
    st.stop()

# --- Chart Screen ---
if st.button("Stop Chart"):
    st.session_state.started = False
    st.experimental_rerun()

# Auto-refresh
st_autorefresh(interval=st.session_state.refresh * 60 * 1000, key="auto")

symbol       = st.session_state.symbol
show_rsi     = st.session_state.show_rsi
show_boll    = st.session_state.show_boll
refresh_rate = st.session_state.refresh

# Fetch data
try:
    df = fetch_data(symbol)
except Exception as e:
    st.error(e)
    st.stop()

closes = df['Close']
times  = df.index
first  = closes.iloc[0].item()
last   = closes.iloc[-1].item()

# Trend line
x = np.arange(len(closes))
m, b = np.polyfit(x, closes.values, 1)
trend = m * x + b

# Bollinger Bands
if show_boll:
    upper, lower = compute_bollinger(closes)
else:
    upper = lower = None

# RSI
if show_rsi:
    rsi = compute_rsi(closes)
else:
    rsi = None

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Price & Trend
ax1.plot(times, closes, label='Price')
ax1.plot(times, trend, '--', label='Trend')
if show_boll:
    ax1.plot(times, upper, '--', alpha=0.5, label='Bollinger Upper')
    ax1.plot(times, lower, '--', alpha=0.5, label='Bollinger Lower')
ax1.set_title(f"{symbol} – Change: {last-first:+.2f}")
ax1.set_ylabel("Price (USD)")
ax1.legend()
ax1.grid(True)

# RSI subplot
if show_rsi and rsi is not None:
    ax2.plot(times, rsi, label='RSI')
    ax2.axhline(70, linestyle='--', color='red', alpha=0.3)
