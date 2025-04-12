import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame
from dotenv import load_dotenv
import os
import csv
import time

pair1 = "GBPUSD"
pair2 = "EURUSD"

# ✅ Load API Keys from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
BASE_URL = 'https://paper-api.alpaca.markets'
api = REST(API_KEY, SECRET_KEY, BASE_URL)

# ✅ Logging Trades
def log_trade(symbol, side, qty, price, tp_price=None, sl_price=None):
    with open('trade_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([pd.Timestamp.now(), symbol, side, qty, price, tp_price, sl_price])
        print(f"Trade Logged: {symbol}, {side}, {qty}, price: ${price}, TP: ${tp_price}, SL: ${sl_price}")

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
pair1_data = get_stock_data(pair1, start_date, end_date)
pair2_data = get_stock_data(pair2, start_date, end_date)

# ✅ Merge Data
df = pd.merge(pair1_data, pair2_data, left_index=True, right_index=True)
print(df.head())

# ✅ Calculate Correlation
correlation = df.corr().iloc[0, 1]
print(f"Correlation between {pair1} and {pair2}: {correlation:.2f}")

if abs(correlation) < 0.7:
    print("Correlation is too weak for pair trading. Exiting.")
    exit()

# ✅ Perform Linear Regression to Find Hedge Ratio (Beta)
X = sm.add_constant(df[pair2])
model = sm.OLS(df[pair1], X).fit()
hedge_ratio = model.params[pair2]
print(f"Hedge Ratio (Beta): {hedge_ratio:.2f}")

# ✅ Calculate Spread
df['Spread'] = df[pair1] - hedge_ratio * df[pair2]

# ✅ Calculate Rolling Mean and Standard Deviation
df['Spread_Mean'] = df['Spread'].rolling(window=30).mean()
df['Spread_Std'] = df['Spread'].rolling(window=30).std()

# ✅ Calculate Z-Score
df['Z-Score'] = (df['Spread'] - df['Spread_Mean']) / df['Spread_Std']
print(df.tail())

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

# ✅ Plot Spread with Entry/Exit Points and TP/SL levels
def plot_strategy(df):
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Spread and Z-score
    ax1 = plt.subplot(211)
    ax1.plot(df.index, df['Spread'], label='Spread')
    ax1.plot(df.index, df['Spread_Mean'], label='Mean', linestyle='--', color='orange')
    ax1.fill_between(df.index, 
                     df['Spread_Mean'] + 2*df['Spread_Std'], 
                     df['Spread_Mean'] - 2*df['Spread_Std'], 
                     alpha=0.2, color='gray', label='±2 Std Dev')
    
    # Plot entry and exit points
    buy_signals = df[df['Signal'] == 'Long']
    sell_signals = df[df['Signal'] == 'Short']
    exit_signals = df[df['Signal'] == 'Exit']
    
    ax1.scatter(buy_signals.index, buy_signals['Spread'], color='green', marker='^', s=100, label='Long')
    ax1.scatter(sell_signals.index, sell_signals['Spread'], color='red', marker='v', s=100, label='Short')
    ax1.scatter(exit_signals.index, exit_signals['Spread'], color='black', marker='x', s=100, label='Exit')
    
    ax1.set_title(f'Spread Between {pair1} and {pair2} with Signals')
    ax1.legend()
    
    # Plot 2: Z-Score
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(df.index, df['Z-Score'], label='Z-Score', color='blue')
    ax2.axhline(y=2, linestyle='--', color='red', label='Upper Threshold (2)')
    ax2.axhline(y=-2, linestyle='--', color='green', label='Lower Threshold (-2)')
    ax2.axhline(y=0.5, linestyle=':', color='gray', label='Exit Threshold (0.5)')
    ax2.axhline(y=-0.5, linestyle=':', color='gray', label='Exit Threshold (-0.5)')
    ax2.axhline(y=0, linestyle='-', color='black', alpha=0.3)
    
    # Add TP and SL levels based on standard deviation
    tp_level = 3.0  # Take profit at 3 standard deviations
    sl_level = 4.0  # Stop loss at 4 standard deviations (reversal of trade)
    
    ax2.axhline(y=tp_level, linestyle='--', color='green', alpha=0.5, label=f'TP Level (+{tp_level})')
    ax2.axhline(y=-tp_level, linestyle='--', color='green', alpha=0.5, label=f'TP Level (-{tp_level})')
    ax2.axhline(y=sl_level, linestyle='--', color='red', alpha=0.5, label=f'SL Level (+{sl_level})')
    ax2.axhline(y=-sl_level, linestyle='--', color='red', alpha=0.5, label=f'SL Level (-{sl_level})')
    
    ax2.set_title('Z-Score with TP/SL Levels')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('pairs_trading_strategy.png')
    plt.show()

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

        # Calculate TP and SL based on direction
        if side == 'buy':
            entry_price = round(current_price * 1.001, 2)  # 0.1% above current price
            tp_price = round(entry_price * (1 + tp_percent/100), 2)
            sl_price = round(entry_price * (1 - sl_percent/100), 2)
        else:  # sell
            entry_price = round(current_price * 0.999, 2)  # 0.1% below current price
            tp_price = round(entry_price * (1 - tp_percent/100), 2)
            sl_price = round(entry_price * (1 + sl_percent/100), 2)

        print(f"{symbol} Current Price: ${current_price}")
        print(f"Placing Limit Order at: ${entry_price}")
        print(f"Take Profit: ${tp_price} ({tp_percent}%)")
        print(f"Stop Loss: ${sl_price} ({sl_percent}%)")

        # Place Limit Order to enter the trade
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            time_in_force='gtc',
            limit_price=entry_price
        )
        print(f"Limit Order Placed: {side} {qty} of {symbol} (Order ID: {order.id})")
        
        # Submit bracket orders (One-Cancels-Other) for TP and SL
        # Wait for the primary order to fill first
        time.sleep(1)  # Give time for order to be processed
        orders = api.list_orders(status='filled')
        for filled_order in orders:
            if filled_order.symbol == symbol and filled_order.side == side:
                # Place take profit order
                tp_side = 'sell' if side == 'buy' else 'buy'
                tp_order = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=tp_side,
                    type='limit',
                    time_in_force='gtc',
                    limit_price=tp_price,
                    order_class='oco',  # One-Cancels-Other
                    stop_loss={'stop_price': sl_price}
                )
                print(f"TP/SL Orders Placed for {symbol} (Order ID: {tp_order.id})")
                
                # Log the trade
                log_trade(symbol, side, qty, entry_price, tp_price, sl_price)
                break

    except Exception as e:
        print(f"Error placing trade for {symbol}: {e}")


# ✅ Check Current Positions
def get_current_position(symbol):
    try:
        position = api.get_position(symbol)
        return int(position.qty)
    except Exception:
        return 0

# ✅ NEW: List all open orders
def list_open_orders(symbol=None):
    try:
        if symbol:
            orders = api.list_orders(status='open', symbols=symbol)
            print(f"\n--- Open Orders for {symbol} ---")
        else:
            orders = api.list_orders(status='open')
            print("\n--- All Open Orders ---")
            
        if not orders:
            print("No open orders found.")
            return []
            
        for order in orders:
            print(f"Order ID: {order.id}")
            print(f"Symbol: {order.symbol}")
            print(f"Side: {order.side}")
            print(f"Qty: {order.qty}")
            print(f"Type: {order.type}")
            if hasattr(order, 'limit_price') and order.limit_price:
                print(f"Limit Price: ${order.limit_price}")
            if hasattr(order, 'stop_price') and order.stop_price:
                print(f"Stop Price: ${order.stop_price}")
            print(f"Status: {order.status}")
            print(f"Created: {order.created_at}")
            print("---")
        
        return orders
    except Exception as e:
        print(f"Error listing orders: {e}")
        return []

# ✅ NEW: Cancel specific order by ID
def cancel_order(order_id):
    try:
        api.cancel_order(order_id)
        print(f"Order {order_id} cancelled successfully.")
        return True
    except Exception as e:
        print(f"Error cancelling order {order_id}: {e}")
        return False

# ✅ NEW: Cancel all open orders for a symbol or all symbols
def cancel_all_orders(symbol=None):
    try:
        if symbol:
            api.cancel_all_orders(symbols=symbol)
            print(f"All open orders for {symbol} cancelled.")
        else:
            api.cancel_all_orders()
            print("All open orders cancelled.")
        return True
    except Exception as e:
        print(f"Error cancelling orders: {e}")
        return False

# ✅ NEW: Close position for a symbol
def close_position(symbol):
    try:
        position = api.get_position(symbol)
        qty = abs(float(position.qty))
        side = 'sell' if float(position.qty) > 0 else 'buy'
        
        # Cancel any open orders for this symbol first
        cancel_all_orders(symbol)
        
        # Submit market order to close the position
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        print(f"Position for {symbol} closed with market order ({side} {qty} shares).")
        return True
    except Exception as e:
        print(f"Error closing position for {symbol}: {e}")
        return False

# ✅ NEW: Close all positions
def close_all_positions():
    try:
        api.close_all_positions()
        print("All positions closed.")
        return True
    except Exception as e:
        print(f"Error closing all positions: {e}")
        return False

# ✅ Execute Trades Based on Signals
def execute_trades(df):
    position_open = False
    lot_size = 10

    for index, row in df.iterrows():
        signal = row['Signal']
        pair1_qty = lot_size
        pair2_qty = int(lot_size * hedge_ratio)

        if signal == "Long" and not position_open:
            print(f"{index}: Opening Long Position (Long {pair1}, Short {pair2})")
            place_trade(pair1, pair1_qty, 'buy', tp_percent=2, sl_percent=1)
            place_trade(pair2, pair2_qty, 'sell', tp_percent=2, sl_percent=1)
            position_open = True

        elif signal == "Short" and not position_open:
            print(f"{index}: Opening Short Position (Short {pair1}, Long {pair2})")
            place_trade(pair1, pair1_qty, 'sell', tp_percent=2, sl_percent=1)
            place_trade(pair2, pair2_qty, 'buy', tp_percent=2, sl_percent=1)
            position_open = True

        elif signal == "Exit" and position_open:
            print(f"{index}: Exiting Position")
            # Close existing positions
            pair1_pos = get_current_position(pair1)
            pair2_pos = get_current_position(pair2)
            
            if pair1_pos != 0:
                place_trade(pair1, abs(pair1_pos), 'sell' if pair1_pos > 0 else 'buy')
            if pair2_pos != 0:
                place_trade(pair2, abs(pair2_pos), 'buy' if pair2_pos < 0 else 'sell')
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

# ✅ NEW: Interactive command menu
def show_command_menu():
    print("\n===== Pairs Trading Command Menu =====")
    print("1. Show open orders")
    print("2. Show positions")
    print("3. Cancel specific order")
    print("4. Cancel all orders")
    print("5. Close specific position")
    print("6. Close all positions")
    print("7. Run trading algorithm once")
    print("8. Start automated trading")
    print("9. Plot strategy")
    print("0. Exit program")
    print("=====================================")
    
    choice = input("Enter your choice (0-9): ")
    return choice

# ✅ NEW: Show all current positions
def show_positions():
    try:
        positions = api.list_positions()
        print("\n--- Current Positions ---")
        if not positions:
            print("No open positions.")
            return
            
        for position in positions:
            print(f"Symbol: {position.symbol}")
            print(f"Quantity: {position.qty}")
            print(f"Side: {'Long' if float(position.qty) > 0 else 'Short'}")
            print(f"Entry Price: ${position.avg_entry_price}")
            print(f"Current Price: ${position.current_price}")
            print(f"Market Value: ${position.market_value}")
            profit_loss = float(position.unrealized_pl)
            print(f"Unrealized P&L: ${profit_loss:.2f}")
            profit_loss_percent = float(position.unrealized_plpc) * 100
            print(f"Unrealized P&L %: {profit_loss_percent:.2f}%")
            print("---")
    except Exception as e:
        print(f"Error retrieving positions: {e}")

# ✅ Main function with interactive menu
def main():
    # Initial setup and analysis
    #plot_strategy(df)  # Plot the strategy before execution
    
    automated_mode = False
    check_interval = 3600  # Check every hour
    next_check_time = 0
    
    while True:
        if automated_mode:
            current_time = int(time.time())
            
            if current_time >= next_check_time:
                print(f"\n--- Trading Check at {pd.Timestamp.now()} ---")
                execute_trades(df)
                
                # Get latest prices for monitoring
                get_latest_price(pair1)
                get_latest_price(pair2)
                
                # Set next check time
                next_check_time = current_time + check_interval
                print(f"Next check in {check_interval} seconds")
                print("\nPress Ctrl+C to return to menu...")
            
            # Sleep to avoid excessive CPU usage
            time.sleep(5)
            
            # Check if user wants to go back to menu (non-blocking input)
            # This is simplified here, actual non-blocking input requires more complex code
            try:
                # This is a simplification - in a real application, you'd use a more 
                # sophisticated approach for non-blocking input
                pass
            except KeyboardInterrupt:
                automated_mode = False
                print("\nReturning to menu...")
        else:
            choice = show_command_menu()
            
            if choice == '1':
                list_open_orders()
            elif choice == '2':
                show_positions()
            elif choice == '3':
                order_id = input("Enter Order ID to cancel: ")
                cancel_order(order_id)
            elif choice == '4':
                symbol = input("Enter symbol to cancel all orders (or press Enter for all symbols): ")
                cancel_all_orders(symbol if symbol else None)
            elif choice == '5':
                symbol = input("Enter symbol to close position: ")
                close_position(symbol)
            elif choice == '6':
                close_all_positions()
            elif choice == '7':
                execute_trades(df)
            elif choice == '8':
                automated_mode = True
                next_check_time = int(time.time())  # Start immediately
                print("Starting automated trading. Press Ctrl+C to return to menu.")
            elif choice == '9':
                plot_strategy(df)
            elif choice == '0':
                print("Exiting program...")
                break
            else:
                print("Invalid choice. Please try again.")

# Run the main function if script is executed directly
if __name__ == "__main__":
    main()