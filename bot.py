import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame
from dotenv import load_dotenv
import os
import csv

pair1 = "TSLA"
pair2 = "RIVN"

# ✅ Load API Keys from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
BASE_URL = 'https://paper-api.alpaca.markets'
api = REST(API_KEY, SECRET_KEY, BASE_URL)

# ✅ Logging Trades
def log_trade(symbol, side, qty):
    with open('trade_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([pd.Timestamp.now(), symbol, side, qty])
        print(f"Trade Logged: {symbol}, {side}, {qty}")

# ✅ Fetch Data from Alpaca
def get_stock_data(symbol, start_date, end_date):
    try:
        data = api.get_bars(symbol, TimeFrame.Day, start=start_date, end=end_date).df
        data = data[['close']]
        data.columns = [symbol]
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# ✅ Get Data for pair1 and pair2
start_date = '2024-01-01'
end_date = '2024-04-24'
pair1_data = get_stock_data('pair1', start_date, end_date)
pair2_data = get_stock_data('pair2', start_date, end_date)

# ✅ Merge Data
df = pd.merge(pair1_data, pair2_data, left_index=True, right_index=True)
#print(df.head())

# ✅ Calculate Correlation
correlation = df.corr().iloc[0, 1]
print(f"Correlation between pair1 and pair2: {correlation:.2f}")

if abs(correlation) < 0.7:
    print("Correlation is too weak for pair trading. Exiting.")
    exit()

# ✅ Perform Linear Regression to Find Hedge Ratio (Beta)
X = sm.add_constant(df['pair2'])
model = sm.OLS(df['pair1'], X).fit()
hedge_ratio = model.params['pair2']
print(f"Hedge Ratio (Beta): {hedge_ratio:.2f}")

# ✅ Calculate Spread
df['Spread'] = df['pair1'] - hedge_ratio * df['pair2']

# ✅ Calculate Rolling Mean and Standard Deviation
df['Spread_Mean'] = df['Spread'].rolling(window=30).mean()
df['Spread_Std'] = df['Spread'].rolling(window=30).std()

# ✅ Calculate Z-Score
df['Z-Score'] = (df['Spread'] - df['Spread_Mean']) / df['Spread_Std']
#print(df.tail())

# ✅ Generate Signals Based on Z-Score
def generate_signals(row):
    if row['Z-Score'] > 2:
        return "Short"
    elif row['Z-Score'] < -2:
        return "Long"
    elif -0.5 <= row['Z-Score'] <= 0.5:
        return "Exit"
    else:
        return "Hold"

df['Signal'] = df.apply(generate_signals, axis=1)
print(df[['Z-Score', 'Signal']].tail())

# ✅ Validate Trade Quantity
def validate_quantity(symbol, qty):
    if qty <= 0:
        print(f"Invalid quantity for {symbol}. Qty should be greater than zero.")
        return False
    return True

# ✅ Place Trades with TP and SL
def place_trade(symbol, qty, side, tp_percent=2, sl_percent=1):
    if not validate_quantity(symbol, qty):
        return

    try:
        latest_trade = api.get_latest_trade(symbol)
        current_price = float(latest_trade.price)

        # Calculate TP and SL
        if side == 'buy':
            limit_price = round(current_price * 1.001, 2) # 0.1% above current price
        else:
            limit_price = round(current_price * 0.999, 2) # 0.1% below current price

        print(f"{symbol} Current Price: ${current_price}")
        print(f"Placing Limit Order at: ${limit_price}")

        # Place Limit Order to enter the trade
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            time_in_force='gtc',
            limit_price=limit_price
        )
        print(f"Limit Order Placed: {side} {qty} of {symbol}")

    except Exception as e:
        print(f"Error placing trade for {symbol}: {e}")


# ✅ Check Current Positions
def get_current_position(symbol):
    try:
        position = api.get_position(symbol)
        return int(position.qty)
    except Exception:
        return 0

# ✅ Execute Trades Based on Signals
def execute_trades(df):
    position_open = False
    lot_size = 10

    for index, row in df.iterrows():
        signal = row['Signal']
        pair1_qty = lot_size
        pair2_qty = int(lot_size * hedge_ratio)

        if signal == "Long" and not position_open:
            print(f"{index}: Opening Long Position (Long pair1, Short pair2)")
            place_trade('pair1', pair1_qty, 'buy')
            place_trade('pair2', pair2_qty, 'sell')
            position_open = True

        elif signal == "Short" and not position_open:
            print(f"{index}: Opening Short Position (Short pair1, Long pair2)")
            place_trade('pair1', pair1_qty, 'sell')
            place_trade('pair2', pair2_qty, 'buy')
            position_open = True

        elif signal == "Exit" and position_open:
            print(f"{index}: Exiting Position")
            if get_current_position('pair1') != 0:
                place_trade('pair1', pair1_qty, 'sell' if get_current_position('pair1') > 0 else 'buy')
            if get_current_position('pair2') != 0:
                place_trade('pair2', pair2_qty, 'buy' if get_current_position('pair2') < 0 else 'sell')
            position_open = False


def get_latest_price(symbol):
    try:
        bars = api.get_bars(symbol, TimeFrame.Minute, limit=1).df
        latest_price = bars['close'].iloc[0]
        print(f"{symbol} Latest Price (from historical data): ${latest_price}")
        return latest_price
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
import time
timee = 0
while(1):
    if(timee == 0):
        execute_trades(df)
        get_latest_price('pair1')
        get_latest_price('pair2')
        timee = 3600
    timee -= 1
    time.sleep(1)
# ✅ Run the Trade Execution
