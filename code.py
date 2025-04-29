```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# -------- Pattern Detection Helpers --------
def detect_pattern(closes: pd.Series) -> str:
    """
    Very simple pattern detection over the last few data points:
    - Head & Shoulders
    - Double Top
    - Triple Top
    - Rising Wedge
    - Inverse Head & Shoulders
    - Double Bottom
    - Unique Three River
    - Falling Wedge
    """
    p = closes.values
n    if len(p) < 7:
        return "None"

    # Head & Shoulders: peak, higher peak, peak with shoulders
    if p[-7] < p[-6] > p[-5] and p[-5] < p[-4] and p[-4] < p[-3] > p[-2] and p[-2] < p[-1]:
        return "Head & Shoulders"
    # Double Top: two similar peaks
    if abs(p[-6] - p[-3]) / p[-6] < 0.01 and p[-6] > p[-5] and p[-3] > p[-2]:
        return "Double Top"
    # Triple Top
    if abs(p[-8] - p[-5]) / p[-8] < 0.01 and abs(p[-5] - p[-2]) / p[-5] < 0.01:
        return "Triple Top"
    # Rising Wedge: narrowing higher lows/higher highs
    x = np.arange(6)
    highs = p[-6:]
    lows = closes[-6:].rolling(2).min().dropna().values
    if np.polyfit(x, highs, 1)[0] > 0 and np.polyfit(x, lows, 1)[0] > 0 and abs(np.polyfit(x, highs,1)[0] - np.polyfit(x, lows,1)[0]) < 0.01:
        return "Rising Wedge"
    # Inverse Head & Shoulders
    if p[-7] > p[-6] < p[-5] and p[-5] > p[-4] and p[-4] > p[-3] < p[-2] and p[-2] > p[-1]:
        return "Inverse Head & Shoulders"
    # Double Bottom
    if abs(p[-6] - p[-3]) / p[-6] < 0.01 and p[-6] < p[-5] and p[-3] < p[-2]:
        return "Double Bottom"
    # Unique Three River (three lows)
    lows_idx = np.argsort(p[-7:])[:3]
    if len(set(lows_idx)) == 3:
        return "Unique Three River"
    # Falling Wedge
    if np.polyfit(x, highs, 1)[0] < 0 and np.polyfit(x, lows, 1)[0] < 0 and abs(np.polyfit(x, highs,1)[0] - np.polyfit(x, lows,1)[0]) < 0.01:
        return "Falling Wedge"

    return "None"

# -------- Market Status --------
def get_market_status(tz=None) -> str:
    now = datetime.now()
    weekday = now.weekday()
    h = now.hour + now.minute/60
    if weekday < 5 and 9.5 <= h < 16:
        return "Market Open"
    if weekday < 5 and 16 <= h < 20:
        return "After Hours Trading"
    return "Market Closed"

def get_24h_status() -> str:
    # Stocks typically trade Sun 20:00 ET -> Fri 20:00 ET
    now = datetime.now()
    weekday = now.weekday()
    if weekday == 6 and now.hour < 20:
        return "24h Closed"
    if weekday == 4 and now.hour >= 20:
        return "24h Closed"
    return "24h Markets Open"

# -------- RSI Calculation --------
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -------- Streamlit App --------
st.set_page_config(layout="wide")

st.title("📈 Investment Trend App")

# Sidebar Options
st.sidebar.header("Chart Settings")
ticker = st.sidebar.text_input("Ticker symbol:", "AAPL").upper()
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_boll = st.sidebar.checkbox("Show Bollinger Bands", value=True)
refresh = st.sidebar.slider("Refresh every (min)", 1, 5, 1)
if st.sidebar.button("Start Chart"):
    st.session_state.running = True
if 'running' not in st.session_state:
    st.session_state.running = False

if st.session_state.running:
    # Fetch data
    df = yf.download(ticker, period='1d', interval='1m', auto_adjust=True)
    if df.empty:
        st.error(f"[Error] No data for {ticker}")
    else:
        closes = df['Close']
        pattern = detect_pattern(closes)
        today_change = closes[-1] - closes[0]
        trend = 'Uptrend' if today_change >= 0 else 'Downtrend'
        signal = 'BUY' if trend=='Uptrend' else ('SELL' if trend=='Downtrend' else 'HOLD')
        rsi = compute_rsi(closes)
        # Bollinger
        if show_boll:
            mid = closes.rolling(20).mean()
            std = closes.rolling(20).std()
            upper = mid + 2*std
            lower = mid - 2*std
        
        # Plot
        fig, axs = plt.subplots(2,1, figsize=(10,6), sharex=True)
        ax1, ax2 = axs
        color = 'green' if today_change>=0 else 'red'
        ax1.plot(df.index, closes, color=color, label='Price')
        ax1.plot(df.index, np.poly1d(np.polyfit(np.arange(len(closes)), closes,1))(np.arange(len(closes))), '--', color='orange', label='Trend')
        if show_boll:
            ax1.plot(df.index, upper, '--', color='grey', label='Bollinger Upper')
            ax1.plot(df.index, lower, '--', color='grey', label='Bollinger Lower')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.set_title(f"{ticker} - Daily Change: {today_change:.2f}")

        if show_rsi:
            ax2.plot(df.index, rsi, color='orange')
            ax2.axhline(70, linestyle='--', alpha=0.5)
            ax2.axhline(30, linestyle='--', alpha=0.5)
            ax2.set_ylabel('RSI')
        ax2.set_xlabel('Time')
        st.pyplot(fig)

        # Info Panels
        col1, col2 = st.columns([1,2])
        with col1:
            st.header("Signal & Market")
            if signal=='BUY': st.success(signal)
            elif signal=='SELL': st.error(signal)
            else: st.warning(signal)
            status_str = f"{get_market_status()} -------- {get_24h_status()}"
            st.info(status_str)
            st.write(f"Last refresh: {datetime.now().strftime('%H:%M:%S')} — next in {refresh} min")
        with col2:
            st.header("Info Panels")
            st.info(f"{trend}\nDetected trend: price is {'rising' if trend=='Uptrend' else 'falling'}.")
            st.warning(f"Pattern: {pattern}\nRecognizable pattern: {pattern}.")

    # Auto refresh
    st.experimental_rerun()
```
