import random, json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

# ─── Configuration Menu ───
turns_slider      = widgets.IntSlider(value=25, min=5, max=50, step=1,   description='Turns:')
volatility_slider = widgets.FloatSlider(value=0.02, min=0.0, max=0.10, step=0.005, description='Volatility:')
event_freq_slider = widgets.FloatSlider(value=0.15, min=0.0, max=0.50, step=0.01,  description='Event Freq:')
commission_slider = widgets.FloatSlider(value=0.001, min=0.0, max=0.01, step=0.0005, description='Commission:')
slippage_slider   = widgets.FloatSlider(value=0.002, min=0.0, max=0.01, step=0.0005, description='Slippage:')
start_button      = widgets.Button(description='Start Game', button_style='success')

display(widgets.VBox([
    widgets.Label("🏁 Configure Game Settings"),
    turns_slider,
    volatility_slider,
    event_freq_slider,
    commission_slider,
    slippage_slider,
    start_button
]))

# ─── Globals to be set on start ───
MAX_TURNS       = None
volatility      = None
event_freq      = None
commission_rate = None
slippage_rate   = None

# ─── Placeholders for game state ───
cash = 0.0
stocks = {}
sector_map = {}
portfolio = {}
turn = 0
history = []
price_hist = {}
scheduled_event = None
event_log = []
trade_count = 0
limit_orders = []
stop_orders = []
achievements = set()

# ─── Core Functions ───
def port_value():
    return cash + sum(stocks[t]*q for t,q in portfolio.items())

def apply_fees_slippage(ticker, qty, is_buy):
    global cash
    price = stocks[ticker]
    cost  = price*qty
    fee   = cost*commission_rate
    slip  = cost*slippage_rate
    if is_buy and cash >= cost+fee+slip:
        cash -= cost+fee+slip
        portfolio[ticker] += qty
        return True
    if (not is_buy) and portfolio[ticker] >= qty:
        cash += cost-fee-slip
        portfolio[ticker] -= qty
        return True
    return False

def process_orders():
    for order in limit_orders[:]:
        t,qty,pr,is_buy = order
        if (is_buy and stocks[t] <= pr) or (not is_buy and stocks[t] >= pr):
            if apply_fees_slippage(t, qty, is_buy):
                event_log.append(f"{'Buy' if is_buy else 'Sell'} limit {qty}×{t}@{pr}")
                limit_orders.remove(order)
    for order in stop_orders[:]:
        t,qty,sp,is_buy = order
        cond = (is_buy and stocks[t] >= sp) or (not is_buy and stocks[t] <= sp)
        if cond and apply_fees_slippage(t, qty, not is_buy):
            event_log.append(f"{'Sell' if not is_buy else 'Buy'} stop {qty}×{t}@{sp}")
            stop_orders.remove(order)

def simulate():
    global turn, scheduled_event
    turn += 1
    # half-event
    if scheduled_event:
        ev = scheduled_event
        half = ev['change']/2
        for t in price_hist:
            if sector_map[t]==ev['sector']:
                stocks[t] = round(stocks[t]*(1+half),2)
        ev['turns_left'] -= 1
        event_log.append(f"{ev['name']} half @turn{turn}")
    # random + momentum
    for t,h in price_hist.items():
        last = h[-1]
        if len(h)>1:
            delta = last - h[-2]
            change = random.uniform(0,volatility) if delta>0 else random.uniform(-volatility,0)
        else:
            change = random.uniform(-volatility,volatility)
        new = round(stocks[t]*(1+change),2)
        stocks[t], h[:] = new, h+[new]
    # second half
    if scheduled_event and scheduled_event['turns_left']==0:
        event_log.append(f"{scheduled_event['name']} 2nd half @turn{turn}")
        scheduled_event = None
    process_orders()
    history.append(port_value())

def maybe_schedule():
    global scheduled_event
    if scheduled_event is None and random.random()<event_freq:
        ev = random.choice([
            {'name':'Tech Rally','sector':'Tech','change':0.10,'blurb':'Tech earnings beat.'},
            {'name':'Consumer Slump','sector':'Consumer','change':-0.10,'blurb':'Consumer dip.'}
        ]).copy()
        ev['turns_left']=2
        scheduled_event=ev
        event_log.append(f"Scheduled {ev['name']} next turn")

def check_achievements():
    roi=(port_value()-10000)/10000
    if port_value()>=2*10000 and 'DoubleUp' not in achievements:
        achievements.add('DoubleUp'); event_log.append("Achievement: Doubled money")
    if trade_count>=10 and 'TenTrades' not in achievements:
        achievements.add('TenTrades'); event_log.append("Achievement: 10 trades")

def end_game():
    for w in all_widgets: w.disabled=True
    final=port_value(); roi=(final-10000)/10000
    stars=1+(roi>=0.10)+(roi>=0.20); bonus=' ⭐' if roi>0.20 else ''
    clear_output()
    print("🎉 GAME OVER 🎉")
    print(f"Turns: {turn}/{MAX_TURNS}  Trades: {trade_count}")
    print(f"Value: ${final:.2f}  ROI: {roi*100:+.2f}%  Rating: {'★'*stars}{bonus}\n")
    if event_log:
        print("Log:")
        for e in event_log: print("•",e)
    df=pd.DataFrame({'Price':stocks,'Shares':portfolio})
    print("\nFinal Portfolio:"); display(df)
    sec={}
    for t,q in portfolio.items():
        sec.setdefault(sector_map[t],0)
        sec[sector_map[t]]+=stocks[t]*q
    if sec:
        fig,ax=plt.subplots(figsize=(4,4))
        ax.pie(sec.values(), labels=sec.keys(), autopct='%1.1f%%', wedgeprops={'width':0.4})
        ax.set_title("Sector Allocation"); plt.show()

# ─── UI Refresh & Controls ───
output = widgets.Output(layout={'border':'1px solid gray'})

action_dd   = widgets.Dropdown(options=['Buy','Sell'], description='Action:')
stock_dd    = widgets.Dropdown(options=[],            description='Stock:')
qty         = widgets.BoundedIntText(value=1, min=1, max=100, description='Qty:')
limit_price = widgets.FloatText(description='Limit $')
stop_price  = widgets.FloatText(description='Stop $')
go_btn      = widgets.Button(description='Go!', button_style='primary')
limit_btn   = widgets.Button(description='Set Limit')
stop_btn    = widgets.Button(description='Set Stop')
hold_btn    = widgets.Button(description='Holdings')
sect_btn    = widgets.Button(description='Sectors')
save_btn    = widgets.Button(description='Save')
load_btn    = widgets.Button(description='Load')

def refresh():
    output.clear_output()
    if scheduled_event and scheduled_event['turns_left']==2:
        print("🔔", scheduled_event['name'], "-", scheduled_event['blurb'])
    pv=port_value(); prev=history[-2] if len(history)>1 else pv
    pct=(pv-prev)/prev*100 if prev else 0; icon='🟢' if pct>=0 else '🔴'
    with output:
        print(f"Turn {turn}/{MAX_TURNS} | Cash ${cash:.2f} | Value ${pv:.2f} ({pct:+.2f}%){icon}")
        df=pd.DataFrame({'Price':stocks,'Shares':portfolio})
        display(df)
        color='green' if pv>=10000 else 'red'
        plt.figure(figsize=(5,3)); plt.plot(history,marker='o',color=color); plt.grid(True); plt.show()

def on_go(_):
    global trade_count
    act, t, q = action_dd.value, stock_dd.value, qty.value
    if apply_fees_slippage(t,q,act=='Buy'):
        trade_count+=1; event_log.append(f"{act} {q}×{t}@turn{turn}")
    simulate(); maybe_schedule(); check_achievements()
    if turn>=MAX_TURNS: end_game()
    else: refresh()

def on_limit(_):
    limit_orders.append((stock_dd.value, qty.value, limit_price.value, action_dd.value=='Buy'))
    event_log.append("Limit set")

def on_stop(_):
    stop_orders.append((stock_dd.value, qty.value, stop_price.value, action_dd.value!='Buy'))
    event_log.append("Stop set")

def show_holdings(_):
    refresh()
    labels=[t for t,q in portfolio.items() if q>0]
    sizes =[stocks[t]*q for t,q in portfolio.items() if q>0]
    if sizes:
        fig,ax=plt.subplots(figsize=(4,4))
        ax.pie(sizes,labels=labels,autopct='%1.1f%%',wedgeprops={'width':0.4}); plt.show()

def show_sectors(_):
    refresh()
    sec={}
    for t,q in portfolio.items():
        sec.setdefault(sector_map[t],0)
        sec[sector_map[t]]+=stocks[t]*q
    labels,sizes=list(sec.keys()),list(sec.values())
    if sizes:
        fig,ax=plt.subplots(figsize=(4,4))
        ax.pie(sizes,labels=labels,autopct='%1.1f%%',wedgeprops={'width':0.4}); plt.show()

def on_save(_):
    state = dict(cash=cash, portfolio=portfolio, stocks=stocks,
                 turn=turn, history=history, price_hist=price_hist,
                 event_log=event_log, trade_count=trade_count)
    with open('game_save.json','w') as f: json.dump(state,f)
    event_log.append("Game saved.")

def on_load(_):
    global cash,portfolio,stocks,turn,history,price_hist,event_log,trade_count
    s=json.load(open('game_save.json'))
    cash,portfolio,stocks = s['cash'],s['portfolio'],s['stocks']
    turn,history,price_hist = s['turn'],s['history'],s['price_hist']
    event_log,trade_count  = s['event_log'],s['trade_count']
    refresh()

go_btn   .on_click(on_go)
limit_btn.on_click(on_limit)
stop_btn .on_click(on_stop)
hold_btn .on_click(show_holdings)
sect_btn .on_click(show_sectors)
save_btn .on_click(on_save)
load_btn .on_click(on_load)

all_widgets = [action_dd,stock_dd,qty,go_btn,limit_price,limit_btn,
               stop_price,stop_btn,hold_btn,sect_btn,save_btn,load_btn]

controls = widgets.VBox([output, widgets.HBox(all_widgets)])

def start_game(_):
    global MAX_TURNS, volatility, event_freq, commission_rate, slippage_rate
    MAX_TURNS       = turns_slider.value
    volatility      = volatility_slider.value
    event_freq      = event_freq_slider.value
    commission_rate = commission_slider.value
    slippage_rate   = slippage_slider.value

    # initialize game state
    global cash, stocks, sector_map, portfolio, turn, history, price_hist
    global scheduled_event, event_log, trade_count, limit_orders, stop_orders, achievements
    cash            = 10000.00
    stocks          = {'AAPL':150,'MSFT':300,'AMZN':3300}
    sector_map      = {'AAPL':'Tech','MSFT':'Tech','AMZN':'Consumer'}
    portfolio       = {t:0 for t in stocks}
    turn            = 0
    history         = []
    price_hist      = {t:[p] for t,p in stocks.items()}
    scheduled_event = None
    event_log       = []
    trade_count     = 0
    limit_orders    = []
    stop_orders     = []
    achievements    = set()

    clear_output(wait=True)
    display(controls)

    # populate stock dropdown
    stock_dd.options = list(stocks.keys())

    # start
    history.append(port_value())
    simulate()
    refresh()

start_button.on_click(start_game)
