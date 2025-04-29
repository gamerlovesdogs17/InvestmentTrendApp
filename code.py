import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time
import pytz
from streamlit_autorefresh import st_autorefresh

# ----- Data Functions -----
@st.cache_data
def fetch_data(ticker, period="1d", interval="1m"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError("No price data found. Ticker may be invalid or delisted.")
    return df

@st.cache_data
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@st.cache_data
def compute_bollinger(prices, window=20, num_std=2):
    ma  = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return ma + num_std*std, ma - num_std*std

# ----- Pattern Detection -----
def detect_pattern(prices):
    s = prices.values
    idx = np.arange(len(s))
    peaks    = [i for i in range(1,len(s)-1) if s[i]>s[i-1] and s[i]>s[i+1]]
    troughs  = [i for i in range(1,len(s)-1) if s[i]<s[i-1] and s[i]<s[i+1]]
    detections = []
    # (detection logic as before...)
    # choose most recent
    if detections:
        return max(detections, key=lambda x: x[1])[0]
    return None

# ----- Market Status -----
def get_market_status():
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern)
    wd = now_et.weekday()
    ct = now_et.time()
    if wd<5 and time(9,30)<=ct<time(16,0):
        return 'Market Open'
    elif wd<5 and time(16,0)<=ct<time(20,0):
        return 'After Hours Trading'
    return 'Market Closed'

# ----- 24h Market Status -----
def get_24h_status():
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern)
    wd = now_et.weekday()
    ct = now_et.time()
    # closed Fri 20:00 to Sun 20:00 ET
    if (wd==4 and ct>=time(20,0)) or wd==5 or (wd==6 and ct<time(20,0)):
        return '24h Markets Closed'
    return '24h Markets Open'

# ----- Session State -----
if 'started' not in st.session_state:
    st.session_state.started = False
if 'symbol' not in st.session_state:
    st.session_state.symbol = ""

# ----- Input Screen -----
if not st.session_state.started:
    st.title("Investment Trend App")
    sy = st.text_input("Enter ticker:", "AAPL").upper()
    cb_rsi  = st.checkbox("Show RSI", True)
    cb_boll = st.checkbox("Show Bollinger Bands", True)
    sl_ref  = st.slider("Refresh every N minutes", 1,5,1)
    if st.button("Start Chart"):
        st.session_state.update({
            'started':True,'symbol':sy,'show_rsi':cb_rsi,'show_boll':cb_boll,'refresh':sl_ref
        })
    st.stop()

# ----- Chart Screen -----
if st.button("Stop Chart"):
    st.session_state.started=False
    st.stop()

st_autorefresh(interval=st.session_state.refresh*60*1000,key='auto')

# Fetch Data
try:
    df = fetch_data(st.session_state.symbol)
except Exception as e:
    st.error(e)
    st.stop()

cl = df['Close']
tm = df.index
first = float(cl.iloc[0])
last = float(cl.iloc[-1])

i = np.arange(len(cl))
mt, bt = np.polyfit(i,cl.values,1)
trend = 'Uptrend' if mt>0 else 'Downtrend'
trend_msg = f"Detected trend: price is {'rising' if mt>0 else 'falling'}."

pat = detect_pattern(cl)
pat_name = pat or 'None'
pat_msg = pat and f"Detected pattern: {pat}." or "No recognizable chart pattern detected."
bull = ['Inverse head and shoulders','Double bottom','Unique three river','Falling wedge']
bear = ['Head and shoulders','Double top','Triple top','Rising wedge']
if pat in bull:
    sig = 'BUY'
elif pat in bear:
    sig = 'SELL'
else:
    sig = 'BUY' if last>first else 'SELL' if last<first else 'HOLD'

ub, lb = (compute_bollinger(cl) if st.session_state.show_boll else (None,None))
rsi = compute_rsi(cl) if st.session_state.show_rsi else None

# ----- Layout -----
col_sig, col_chart, col_info = st.columns([1.5,6,2])

with col_sig:
    st.markdown("### Signal & Market")
    if sig=='BUY': st.success(sig)
    elif sig=='SELL': st.error(sig)
    else: st.warning(sig)
    st.info(f"{get_market_status()} -------- {get_24h_status()}")()}")

with col_chart:
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(14,8), sharex=True)
    color = 'green' if last>=first else 'red'
    ax1.plot(tm, cl, color=color, label='Price')
    ax1.plot(tm, mt*i+bt, '--', color='orange', label='Trend')
    if st.session_state.show_boll:
        ax1.plot(tm, ub, '--', alpha=0.5, label='Boll Upper')
        ax1.plot(tm, lb, '--', alpha=0.5, label='Boll Lower')
    ax1.set_title(f"{st.session_state.symbol} – Daily Change: {last-first:+.2f}")
    ax1.set_ylabel('Price (USD)')
    ax1.legend(); ax1.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.tick_params(axis='x', rotation=45)
    if rsi is not None:
        ax2.plot(tm, rsi, label='RSI')
        ax2.axhline(70, linestyle='--', alpha=0.3)
        ax2.axhline(30, linestyle='--', alpha=0.3)
        ax2.set_ylabel('RSI'); ax2.legend(); ax2.grid(True)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.axis('off')
    plt.tight_layout(); st.pyplot(fig)
    st.markdown(f"Last refresh: {pd.Timestamp.now().strftime('%H:%M:%S')} — next in {st.session_state.refresh} min")

with col_info:
    st.markdown('### Info Panels')
    st.info(f"**{trend}**\n{trend_msg}")
    st.warning(f"**{pat_name}**\n{pat_msg}")
