import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide", page_title="Investment Trend App")

#--- Caching data functions ---
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
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, lower

#--- Initialize session state ---
if 'started' not in st.session_state:
    st.session_state.started = False

#--- Initial input form ---
if not st.session_state.started:
    st.title("📈 Investment Trend App")
    symbol = st.text_input("Enter stock ticker symbol:", value="AAPL").upper()
    show_rsi = st.checkbox("Show RSI", value=True)
    show_boll = st.checkbox("Show Bollinger Bands", value=True)
    refresh_rate = st.slider("Refresh every N minutes", 1, 5, 1)

    if st.button("Start Chart") and symbol:
        st.session_state.started = True
        st.session_state.symbol = symbol
        st.session_state.show_rsi = show_rsi
        st.session_state.show_boll = show_boll
        st.session_state.refresh = refresh_rate
        st.experimental_rerun()

#--- Live chart view ---
else:
    # Sidebar controls
    stop = st.sidebar.button("Stop Chart")
    show_rsi = st.sidebar.checkbox("Show RSI", value=st.session_state.show_rsi)
    show_boll = st.sidebar.checkbox("Show Bollinger Bands", value=st.session_state.show_boll)
    refresh_rate = st.sidebar.slider("Refresh every N minutes", 1, 5, st.session_state.refresh)
    symbol = st.session_state.symbol

    placeholder = st.empty()
    cycle = 1

    # Update loop
    while not stop:
        try:
            df = fetch_data(symbol)
        except Exception as e:
            st.error(e)
            break

        closes = df['Close']
        times = df.index

        first_price = closes.iloc[0]
        last_price = closes.iloc[-1]
        price_color = 'green' if last_price >= first_price else 'red'

        # Trend line
        x = np.arange(len(closes))
        y = closes.values
        m, b = np.polyfit(x, y, 1)
        trend = m * x + b

        # Bollinger Bands
        if show_boll:
            upper, lower = compute_bollinger(closes)

        # RSI
        if show_rsi:
            rsi = compute_rsi(closes)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Price & Trend
        ax1.plot(times, closes, label='Price', color=price_color, linewidth=2)
        ax1.plot(times, trend, linestyle='--', label='Trend', color='orange')
        if show_boll:
            ax1.plot(times, upper, linestyle='--', color='gray', alpha=0.5, label='Bollinger Upper')
            ax1.plot(times, lower, linestyle='--', color='gray', alpha=0.5, label='Bollinger Lower')
        ax1.set_title(f"{symbol} • Daily Change: {last_price-first_price:+.2f}")
        ax1.set_ylabel("Price (USD)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        # RSI
        if show_rsi:
            ax2.plot(times, rsi, color='purple', label='RSI')
            ax2.axhline(70, color='red', linestyle='--', alpha=0.3)
            ax2.axhline(30, color='green', linestyle='--', alpha=0.3)
            ax2.set_ylabel('RSI')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)
        else:
            ax2.axis('off')

        plt.tight_layout()
        placeholder.pyplot(fig)
        st.markdown(f"**Refresh #{cycle}** — Next update in {refresh_rate} minute(s)")
        cycle += 1

        time.sleep(refresh_rate * 60)
        stop = st.sidebar.button("Stop Chart")

    # Reset session for new input
    st.session_state.started = False
