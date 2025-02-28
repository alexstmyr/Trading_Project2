import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(data, signals_df, initial_capital, n_shares, commission, margin_cap=250000):
    """
    Runs a backtest of the pair trading strategy.
    
    Trading rules:
      - When the signal transitions from 0 to +1: Open a LONG trade (Buy dependent asset, short independent asset).
      - When the signal transitions from 0 to -1: Open a SHORT trade (Short dependent asset, buy independent asset).
      - When the signal transitions from ±1 to 0, close all open trades.
      - Multiple trades may be open concurrently.
    
    For LONG trades:
      - Entry cost = (dep_price * n_shares * (1+commission)) + (ind_price * n_shares * commission).
      - Exit proceeds = (dep_exit * n_shares * (1-commission)) + (ind_exit * n_shares * (1-commission)).
    For SHORT trades:
      - Entry cost = (dep_price * n_shares * commission) + (ind_price * n_shares * (1+commission)).
      - Exit proceeds = (dep_exit * n_shares * (1-commission)) + (ind_exit * n_shares * (1-commission)).
    
    Capital (cash) is reduced by the full entry cost at trade entry and increased by the exit proceeds when closing.
    The portfolio value is computed at each time as:
        Portfolio Value = Capital + Unrealized P&L of open trades.
    
    Returns:
      - portfolio_df: DataFrame of portfolio value over time.
      - initial_capital: Starting capital.
      - final_capital: Ending portfolio value.
      - total_trades: Total number of trades executed.
      - win_rate: Percentage of trades with net profit > 0.
      - trades_df: DataFrame with each trade's details, with columns ordered as:
            [entry_time, exit_time, <dep>_entry, <dep>_exit, <ind>_entry, <ind>_exit, direction, profit].
    
    Also plots the portfolio value over time.
    """
    capital = initial_capital
    portfolio_values = []
    trade_log = []  # To record details of each closed trade.
    active_trades = []  # List of currently open trades.
    
    # Assume data.columns[0] is the dependent asset (e.g., MSFT) and data.columns[1] is the independent asset (e.g., AMD).
    dep_asset = data.columns[0]
    ind_asset = data.columns[1]
    
    sig_series = signals_df['Signal']
    prev_signal = 0
    
    for timestamp, row in data.iterrows():
        current_signal = sig_series.loc[timestamp] if timestamp in sig_series.index else 0
        dep_price = row[dep_asset]
        ind_price = row[ind_asset]
        
        # --- Trade Exit ---
        # When signal transitions from nonzero to 0, close all open trades.
        if prev_signal != 0 and current_signal == 0 and active_trades:
            for trade in active_trades:
                dep_exit = dep_price
                ind_exit = ind_price
                # For LONG trades (signal +1).
                if trade['direction'] == 1:
                    exit_proceeds = (dep_exit * n_shares * (1 - commission)) + (ind_exit * n_shares * (1 - commission))
                else:
                    # For SHORT trades (signal -1).
                    exit_proceeds = (dep_exit * n_shares * (1 - commission)) + (ind_exit * n_shares * (1 - commission))
                profit = exit_proceeds - trade['entry_cost']
                capital += exit_proceeds
                trade['exit_time'] = timestamp
                trade[f"{dep_asset}_exit"] = dep_exit
                trade[f"{ind_asset}_exit"] = ind_exit
                trade['profit'] = profit
                trade_log.append(trade)
            active_trades = []
        
        # --- Trade Entry ---
        if prev_signal == 0 and current_signal != 0:
            if current_signal == 1:
                # LONG trade: Buy dep, short ind.
                entry_cost = (dep_price * n_shares * (1 + commission)) + (ind_price * n_shares * commission)
                direction = 1
            elif current_signal == -1:
                # SHORT trade: Short dep, buy ind.
                entry_cost = (dep_price * n_shares * commission) + (ind_price * n_shares * (1 + commission))
                direction = -1
            if capital >= entry_cost and capital > margin_cap:
                capital -= entry_cost
                active_trades.append({
                    'entry_time': timestamp,
                    f"{dep_asset}_entry": dep_price,
                    f"{ind_asset}_entry": ind_price,
                    'entry_cost': entry_cost,
                    'direction': direction
                })
        
        # --- Update Portfolio Value ---
        unrealized_pnl = 0
        for trade in active_trades:
            if trade['direction'] == 1:
                pnl = (dep_price - trade[f"{dep_asset}_entry"]) * n_shares + (trade[f"{ind_asset}_entry"] - ind_price) * n_shares
            else:
                pnl = (trade[f"{dep_asset}_entry"] - dep_price) * n_shares + (ind_price - trade[f"{ind_asset}_entry"]) * n_shares
            unrealized_pnl += pnl
        portfolio_values.append(capital + unrealized_pnl)
        prev_signal = current_signal
    
    portfolio_df = pd.DataFrame({'Portfolio Value': portfolio_values}, index=data.index)
    final_capital = portfolio_values[-1] if portfolio_values else capital
    
    total_trades = len(trade_log)
    win_trades = sum(1 for t in trade_log if t['profit'] > 0)
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    
    portfolio_df.plot(title="Portfolio Value Over Time")
    plt.show()
    
    trades_df = pd.DataFrame(trade_log, columns=[
        'entry_time', 'exit_time', f"{dep_asset}_entry", f"{dep_asset}_exit",
        f"{ind_asset}_entry", f"{ind_asset}_exit", 'direction', 'profit'
    ])
    
    return portfolio_df, initial_capital, final_capital, total_trades, win_rate, trades_df
