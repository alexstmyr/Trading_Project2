import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(data, signals_df, initial_capital, n_shares, commission, margin_cap=250000):
    """
    Runs a backtest of the pair trading strategy.
    
    Strategy rules:
      - Open a trade when the signal transitions from 0 to ±1.
      - For a long trade (signal = 1): Buy dependent asset, short independent asset.
      - For a short trade (signal = -1): Short dependent asset, buy independent asset.
      - Close all open trades when the signal returns to 0 (i.e. when the normalized spread crosses 0).
      - Multiple trades may be open concurrently.
    
    Commission is applied on both entry and exit.
    A margin cap ensures that the remaining capital is above a specified threshold.
    
    Returns:
      - portfolio_df: A DataFrame of portfolio value over time.
      - initial_capital: The starting capital.
      - final_capital: The ending portfolio value.
      - total_trades: Total number of trades executed.
      - win_rate: Percentage of trades with net profit > 0.
    
    Also plots the portfolio value over time.
    """
    capital = initial_capital
    portfolio_values = []
    trade_log = []      # Records each closed trade.
    active_trades = []  # List of open trades.
    
    sig_series = signals_df['Signal']
    prev_signal = 0
    
    for timestamp, row in data.iterrows():
        current_signal = sig_series.loc[timestamp] if timestamp in sig_series.index else 0
        price1 = row[data.columns[0]]  # Dependent asset (e.g., MSFT)
        price2 = row[data.columns[1]]  # Independent asset (e.g., AMD)
        
        # --- Trade Entry ---
        # If signal transitions from 0 to nonzero, open a new trade.
        if prev_signal == 0 and current_signal != 0:
            # Determine trade cost:
            if current_signal == 1:
                # Long trade: Buy asset1 (price1) and short asset2 (price2).
                cost = price1 * n_shares * (1 + commission) + price2 * n_shares * commission
            elif current_signal == -1:
                # Short trade: Short asset1 and buy asset2.
                cost = price1 * n_shares * commission + price2 * n_shares * (1 + commission)
            if capital > cost and capital > margin_cap:
                # Record trade details.
                trade = {
                    'entry_time': timestamp,
                    'entry_price1': price1,
                    'entry_price2': price2,
                    'direction': current_signal  # 1 for long, -1 for short.
                }
                active_trades.append(trade)
                capital -= cost  # Deduct entry cost.
        
        # --- Trade Exit ---
        # If signal transitions from nonzero to 0, close all open trades.
        if prev_signal != 0 and current_signal == 0 and active_trades:
            for trade in active_trades:
                if trade['direction'] == 1:
                    # Long trade: profit = (current price1 - entry_price1) * n_shares 
                    #              + (entry_price2 - current price2) * n_shares.
                    profit = (price1 - trade['entry_price1']) * n_shares + (trade['entry_price2'] - price2) * n_shares
                    exit_cost = (price1 * n_shares + price2 * n_shares) * commission
                else:
                    # Short trade: profit = (entry_price1 - current price1) * n_shares 
                    #              + (current price2 - entry_price2) * n_shares.
                    profit = (trade['entry_price1'] - price1) * n_shares + (price2 - trade['entry_price2']) * n_shares
                    exit_cost = (price1 * n_shares + price2 * n_shares) * commission
                net_profit = profit - exit_cost
                capital += net_profit
                trade['exit_time'] = timestamp
                trade['profit'] = net_profit
                trade_log.append(trade)
            active_trades = []
        
        # --- Update Portfolio Value ---
        # For any open trades, calculate unrealized P&L.
        unrealized = 0
        for trade in active_trades:
            if trade['direction'] == 1:
                pnl = (price1 - trade['entry_price1']) * n_shares + (trade['entry_price2'] - price2) * n_shares
            else:
                pnl = (trade['entry_price1'] - price1) * n_shares + (price2 - trade['entry_price2']) * n_shares
            unrealized += pnl
        portfolio_values.append(capital + unrealized)
        prev_signal = current_signal
    
    portfolio_df = pd.DataFrame({'Portfolio Value': portfolio_values}, index=data.index)
    final_capital = portfolio_values[-1]
    total_trades = len(trade_log)
    win_rate = (sum(1 for t in trade_log if t['profit'] > 0) / total_trades * 100) if total_trades > 0 else 0
    
    # Plot portfolio value over time.
    portfolio_df.plot(title="Portfolio Value Over Time")
    plt.show()
    
    return portfolio_df, initial_capital, final_capital, total_trades, win_rate
