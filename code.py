import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import pytz

# ─── Helpers ───────────────────────────────────────────────────────────────────

def get_intraday(ticker):
    now = datetime.now(pytz.UTC)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    df = yf.download(ticker, start=start, end=now, interval="1m", progress=False)
    return df

def compute_indicators(df):
    # linear trendline
    df["Trend"] = pd.Series(
        np.poly1d(np.polyfit(np.arange(len(df)), df["Close"], 1))(np.arange(len(df)))
    )
    # Bollinger Bands
    df["BB_mid"] = df["Close"].rolling(20).mean()
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]
    # RSI
    delta = df["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up = up.ewm(span=14, adjust=False).mean()
    ma_down = down.ewm(span=14, adjust=False).mean()
    rs = ma_up / ma_down
    df["RSI"] = 100 - (100 / (1 + rs))
    return df.dropna()

def detect_pattern(df):
    closes = df["Close"].values
    found = []
    # scan 5-bar windows
    for i in range(len(closes) - 4):
        w = closes[i : i + 5]
        # Head & Shoulders
        if w[2] > w[1] and w[2] > w[3] and w[0] < w[1] and w[4] < w[3]:
            found.append(("Head & Shoulders", i + 2))
        # Inverse Head & Shoulders
        elif w[2] < w[1] and w[2] < w[3] and w[0] > w[1] and w[4] > w[3]:
            found.append(("Inverse H&S", i + 2))
        # Double Top
        elif w[1] > w[2] < w[3] and abs(w[1] - w[3]) / w[1] < 0.01:
            found.append(("Double Top", i + 1))
        # Double Bottom
        elif w[1] < w[2] > w[3] and abs(w[1] - w[3]) / w[1] < 0.01:
            found.append(("Double Bottom", i + 1))
        # Triple Top
        elif w[0] < w[1] > w[2] < w[3] and w[3] > w[4]:
            found.append(("Triple Top", i + 2))
        # Unique Three River
        elif w[0] > w[1] < w[2] > w[3] < w[4] and abs(w[1] - w[3]) / w[1] < 0.01:
            found.append(("Unique Three River", i + 2))
        # Rising Wedge (gentle up slope)
        elif np.polyfit(np.arange(5), w, 1)[0] > 0 and (w.max() - w.min()) < np.std(w) * 1.5:
            found.append(("Rising Wedge", i + 2))
        # Falling Wedge
        elif np.polyfit(np.arange(5), w, 1)[0] < 0 and (w.max() - w.min()) < np.std(w) * 1.5:
            found.append(("Falling Wedge", i + 2))
    return found[-1] if found else ("None", None)

def get_market_status():
    et = datetime.now(pytz.timezone("US/Eastern")).time()
    if time(9,30) <= et <= time(16,0):
        return "Market Open"
    elif time(16,0) < et <= time(20,0):
        return "After-Hours"
    else:
        return "Market Closed"

def get_24h_status():
    now = datetime.now(pytz.timezone("US/Eastern"))
    dow, tod = now.weekday(), now.time()
    # 24h open Sun 20:00 ET → Fri 20:00 ET
    if dow == 6 and tod < time(20,0):
        return "24h Closed"
    if dow < 5 or (dow == 5 and tod < time(20,0)):
        return "24h Open"
    return "24h Closed"

# ─── Streamlit App ────────────────────────────────────────────────────────────

st.title("Live Intraday Trend & Pattern")

ticker    = st.text_input("Ticker", "AAPL").upper()
show_rsi  = st.checkbox("Show RSI", True)
show_bb   = st.checkbox("Show Bollinger Bands", True)
refresh   = st.slider("Refresh every N minutes", 1, 5, 1)

if st.button("Start Chart"):
    st.session_state.started = True

if st.session_state.get("started", False):
    df      = get_intraday(ticker)
    df      = compute_indicators(df)
    pattern, idx = detect_pattern(df)

    first, last = df["Close"].iloc[0], df["Close"].iloc[-1]
    color       = "green" if last >= first else "red"
    # buy/sell/hold by trendline
    sig = "BUY"  if last > df["Trend"].iloc[-1] \
        else "SELL" if last < df["Trend"].iloc[-1] \
        else "HOLD"

    # ─── Plot ─────────────────────────
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6), sharex=True)
    ts = df.index
    ax1.plot(ts, df["Close"], color=color, label="Price")
    ax1.plot(ts, df["Trend"], "--", label="Trend")
    if show_bb:
        ax1.plot(ts, df["BB_upper"], ":", label="BB Upper", alpha=0.6)
        ax1.plot(ts, df["BB_lower"], ":", label="BB Lower", alpha=0.6)
    ax1.set_title(f"{ticker} – Daily Change: {last-first:+.2f}")
    ax1.legend(loc="upper left")

    if show_rsi:
        ax2.plot(ts, df["RSI"], color="orange", label="RSI", linewidth=1.5)
        ax2.axhline(70, linestyle="--", alpha=0.6)
        ax2.axhline(30, linestyle="--", alpha=0.6)
        ax2.set_ylabel("RSI")
        ax2.legend(loc="upper left")
    else:
        ax2.axis("off")

    st.pyplot(fig, clear_figure=True)

    # ─── Info Panels ────────────────────
    col1, col2, col3 = st.columns([1,5,2])
    with col1:
        st.header("Signal")
        st.success(sig) if sig=="BUY" else st.error(sig) if sig=="SELL" else st.warning(sig)
        status_line = f"{get_market_status()} -------- {get_24h_status()}"
        st.info(status_line)
        st.write(f"Last refresh: {datetime.now().strftime('%H:%M:%S')} — next in {refresh} min")

    with col3:
        st.subheader("Trend Info")
        st.info(f"{'No' if pattern=='None' else pattern} trend detected.")
        st.subheader("Pattern Info")
        st.info(f"{pattern} pattern detected." if pattern!="None" else "No recognizable pattern.")

    # rerun on interval
    st.experimental_rerun()
