import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(data, signals_df, initial_capital, n_shares, commission, margin_cap=250000):
    """
    Runs a backtest of the pair trading strategy with separate handling for LONG and SHORT trades.
    
    For LONG trades (signal = +1: Buy dependent asset, Short independent asset):
      - Entry:
          * Buy dependent asset at entry price with commission: 
              cost_buy = MSFT_entry × n_shares × (1 + commission)
          * Short independent asset at entry price, receiving proceeds net of commission: 
              proceeds_short = AMD_entry × n_shares × (1 - commission)
          * Net entry cash outflow = cost_buy - proceeds_short.
      - Exit:
          * Sell dependent asset at exit price, receiving proceeds net of commission: 
              proceeds_sale = MSFT_exit × n_shares × (1 - commission)
          * Cover short for independent asset at exit price, paying cost net of commission: 
              cost_cover = AMD_exit × n_shares × (1 + commission)
          * Net exit cash inflow = proceeds_sale - cost_cover.
          
    For SHORT trades (signal = -1: Short dependent asset, Buy independent asset):
      - Entry:
          * Short dependent asset at entry price, receiving proceeds net of commission:
              proceeds_short = MSFT_entry × n_shares × (1 - commission)
          * Buy independent asset at entry price with commission:
              cost_buy = AMD_entry × n_shares × (1 + commission)
          * Net entry cash outflow = cost_buy - proceeds_short.
      - Exit:
          * Cover short for dependent asset at exit price, paying cost net of commission:
              cost_cover = MSFT_exit × n_shares × (1 + commission)
          * Sell independent asset at exit price, receiving proceeds net of commission:
              proceeds_sale = AMD_exit × n_shares × (1 - commission)
          * Net exit cash inflow = proceeds_sale - cost_cover.
    
    At trade entry, the net cash outflow is deducted from capital.
    At trade exit, the net cash inflow is added to capital.
    The portfolio value at each time is:
         Portfolio Value = Capital + Unrealized P&L of open trades.
    
    The function returns:
      - portfolio_df: DataFrame of portfolio value over time.
      - initial_capital: The starting capital.
      - final_capital: The ending portfolio value.
      - total_trades: Total number of trades executed.
      - win_rate: Percentage of trades with net profit > 0.
      - trades_df: DataFrame with trade details, with columns ordered as:
            [entry_time, exit_time, MSFT_entry, MSFT_exit, AMD_entry, AMD_exit, direction, profit].
    It also plots the portfolio value over time.
    """
    capital = initial_capital
    portfolio_values = []
    trade_log = []       # To record details of each closed trade.
    active_trades = []   # To store open trades.
    
    # Assume data.columns[0] is the dependent asset (e.g., MSFT) and data.columns[1] is the independent asset (e.g., AMD).
    dep_asset = data.columns[0]  # e.g., MSFT
    ind_asset = data.columns[1]  # e.g., AMD
    
    sig_series = signals_df['Signal']
    prev_signal = 0
    
    for timestamp, row in data.iterrows():
        current_signal = sig_series.loc[timestamp] if timestamp in sig_series.index else 0
        dep_price = row[dep_asset]
        ind_price = row[ind_asset]
        
        # --- Trade Exit ---
        # When the signal goes from nonzero to 0, close all open trades.
        if prev_signal != 0 and current_signal == 0 and active_trades:
            for trade in active_trades:
                # Record exit prices.
                dep_exit = dep_price
                ind_exit = ind_price
                # Compute net exit cash flow based on trade direction.
                if trade['direction'] == 1:  # LONG trade.
                    # At entry:
                    #   cost_buy = dep_entry × n_shares × (1 + commission)
                    #   proceeds_short = ind_entry × n_shares × (1 - commission)
                    #   entry_net = cost_buy - proceeds_short.
                    # At exit:
                    #   proceeds_sale = dep_exit × n_shares × (1 - commission)
                    #   cost_cover = ind_exit × n_shares × (1 + commission)
                    #   exit_net = proceeds_sale - cost_cover.
                    proceeds_sale = dep_exit * n_shares * (1 - commission)
                    cost_cover = ind_exit * n_shares * (1 + commission)
                    exit_net = proceeds_sale - cost_cover
                else:  # SHORT trade.
                    # At entry:
                    #   proceeds_short = dep_entry × n_shares × (1 - commission)
                    #   cost_buy = ind_entry × n_shares × (1 + commission)
                    #   entry_net = cost_buy - proceeds_short.
                    # At exit:
                    #   cost_cover = dep_exit × n_shares × (1 + commission)
                    #   proceeds_sale = ind_exit × n_shares × (1 - commission)
                    #   exit_net = proceeds_sale - cost_cover.
                    cost_cover = dep_exit * n_shares * (1 + commission)
                    proceeds_sale = ind_exit * n_shares * (1 - commission)
                    exit_net = proceeds_sale - cost_cover
                
                # Trade profit = exit_net - entry_net (which was deducted at entry).
                profit = exit_net - trade['entry_net']
                capital += exit_net  # Add exit cash inflow.
                
                trade['exit_time'] = timestamp
                trade[f"{dep_asset}_exit"] = dep_exit
                trade[f"{ind_asset}_exit"] = ind_exit
                trade['profit'] = profit
                trade_log.append(trade)
            active_trades = []
        
        # --- Trade Entry ---
        if prev_signal == 0 and current_signal != 0:
            if current_signal == 1:
                # LONG trade: Buy dep_asset, short ind_asset.
                cost_buy = dep_price * n_shares * (1 + commission)
                proceeds_short = ind_price * n_shares * (1 - commission)
                entry_net = cost_buy - proceeds_short  # Net cash outflow.
                direction = 1
            elif current_signal == -1:
                # SHORT trade: Short dep_asset, buy ind_asset.
                proceeds_short = dep_price * n_shares * (1 - commission)
                cost_buy = ind_price * n_shares * (1 + commission)
                entry_net = cost_buy - proceeds_short  # Net cash outflow.
                direction = -1
            
            if capital >= entry_net and capital > margin_cap:
                capital -= entry_net
                active_trades.append({
                    'entry_time': timestamp,
                    f"{dep_asset}_entry": dep_price,
                    f"{ind_asset}_entry": ind_price,
                    'entry_net': entry_net,
                    'direction': direction
                })
        
        # --- Update Portfolio Value ---
        # Compute unrealized P&L for open trades.
        unrealized_pnl = 0
        for trade in active_trades:
            if trade['direction'] == 1:  # LONG trade.
                # Unrealized P&L = (current dep_price - entry dep_price)*n_shares + (entry ind_price - current ind_price)*n_shares.
                pnl = (dep_price - trade[f"{dep_asset}_entry"]) * n_shares + (trade[f"{ind_asset}_entry"] - ind_price) * n_shares
            else:  # SHORT trade.
                pnl = (trade[f"{dep_asset}_entry"] - dep_price) * n_shares + (ind_price - trade[f"{ind_asset}_entry"]) * n_shares
            unrealized_pnl += pnl
        portfolio_values.append(capital + unrealized_pnl)
        prev_signal = current_signal
    
    portfolio_df = pd.DataFrame({'Portfolio Value': portfolio_values}, index=data.index)
    final_capital = portfolio_values[-1] if portfolio_values else capital
    
    total_trades = len(trade_log)
    win_rate = (sum(1 for t in trade_log if t['profit'] > 0) / total_trades * 100) if total_trades > 0 else 0
    
    portfolio_df.plot(title="Portfolio Value Over Time")
    plt.show()
    
    # Create trades DataFrame with desired column order.
    trades_df = pd.DataFrame(trade_log, columns=[
        'entry_time', 'exit_time', f"{dep_asset}_entry", f"{dep_asset}_exit",
        f"{ind_asset}_entry", f"{ind_asset}_exit", 'direction', 'profit'
    ])
    
    return portfolio_df, initial_capital, final_capital, total_trades, win_rate, trades_df
