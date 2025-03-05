import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(data, signals_df, initial_capital, n_shares, commission, margin_cap=250000):
    """
    Runs a backtest of the pair trading strategy with a dynamic hedge ratio.

    Trading rules:
      - When the signal transitions from 0 to +1: Open a LONG trade (Buy dependent asset, short independent asset).
      - When the signal transitions from 0 to -1: Open a SHORT trade (Short dependent asset, buy independent asset).
      - When the signal transitions from ±1 to 0 (either a general exit or a close signal), close all open trades.

    Capital (cash) is reduced by the full entry cost at trade entry and increased by the exit proceeds when closing.
    The portfolio value is computed at each time as:
        Portfolio Value = Capital + Unrealized P&L of open trades.

    Returns:
      - portfolio_df: DataFrame of portfolio value over time.
      - initial_capital: Starting capital.
      - final_capital: Ending portfolio value.
      - total_trades: Total number of trades executed.
      - win_rate: Percentage of trades with net profit > 0.
      - trades_df: DataFrame with each trade's details.
    """
    capital = initial_capital
    portfolio_values = []
    trade_log = []  # To record details of each closed trade.
    active_trades = []  # List of currently open trades.

    dep_asset = data.columns[0]
    ind_asset = data.columns[1]

    sig_series = signals_df['Signal']
    hedge_ratio_series = signals_df["Dynamic Hedge Ratio"]
    prev_signal = 0

    for timestamp, row in data.iterrows():
        current_signal = sig_series.loc[timestamp] if timestamp in sig_series.index else 0
        current_hr = hedge_ratio_series.loc[timestamp] if timestamp in hedge_ratio_series.index else 1

        dep_price = row[dep_asset]
        ind_price = row[ind_asset]

        # --- Trade Exit (Handling Close Signal) ---
        if prev_signal != 0 and (current_signal == 0 or current_signal != prev_signal) and active_trades:
            for trade in active_trades:
                dep_exit = dep_price
                ind_exit = ind_price
                trade_hr = trade["hedge_ratio"]  # Hedge ratio at entry

                qty_dep = n_shares
                qty_ind = n_shares * trade_hr  # Apply hedge ratio to the independent asset

                # Separate handling of LONG and SHORT exits
                if trade['direction'] == 1:  # Closing a LONG trade
                    dep_exit_proceeds = dep_exit * qty_dep * (1 - commission)  # Sell dependent asset
                    ind_exit_proceeds = ind_exit * qty_ind * (1 - commission)  # Buy back independent asset (Short Close)
                else:  # Closing a SHORT trade
                    dep_exit_proceeds = dep_exit * qty_dep * (1 - commission)  # Buy back dependent asset (Short Close)
                    ind_exit_proceeds = ind_exit * qty_ind * (1 - commission)  # Sell independent asset

                exit_proceeds = dep_exit_proceeds + ind_exit_proceeds
                profit = exit_proceeds - trade['entry_cost']
                capital += exit_proceeds

                trade['exit_time'] = timestamp
                trade[f"{dep_asset}_exit"] = dep_exit
                trade[f"{ind_asset}_exit"] = ind_exit
                trade['profit'] = profit
                trade_log.append(trade)

            active_trades = []  # Close all trades

        # --- Trade Entry ---
        if prev_signal == 0 and current_signal != 0:
            qty_dep = n_shares
            qty_ind = n_shares * current_hr  # Apply hedge ratio to the independent asset

            if current_signal == 1:
                # LONG trade: Buy dependent asset, short independent asset.
                dep_entry_cost = dep_price * qty_dep * (1 + commission)  # Buying dependent
                ind_entry_cost = ind_price * qty_ind * commission  # Shorting independent (no full cost)
            elif current_signal == -1:
                # SHORT trade: Short dependent asset, buy independent asset.
                dep_entry_cost = dep_price * qty_dep * commission  # Shorting dependent (no full cost)
                ind_entry_cost = ind_price * qty_ind * (1 + commission)  # Buying independent

            entry_cost = dep_entry_cost + ind_entry_cost
            direction = current_signal

            if capital >= entry_cost and capital > margin_cap:
                capital -= entry_cost
                active_trades.append({
                    'entry_time': timestamp,
                    f"{dep_asset}_entry": dep_price,
                    f"{ind_asset}_entry": ind_price,
                    'entry_cost': entry_cost,
                    'direction': direction,
                    'hedge_ratio': current_hr
                })

        # --- Update Portfolio Value ---
        unrealized_pnl = 0
        for trade in active_trades:
            qty_dep = n_shares
            qty_ind = n_shares * trade["hedge_ratio"]  # Apply hedge ratio to the independent asset

            if trade['direction'] == 1:
                pnl = (dep_price - trade[f"{dep_asset}_entry"]) * qty_dep + (trade[f"{ind_asset}_entry"] - ind_price) * qty_ind
            else:
                pnl = (trade[f"{dep_asset}_entry"] - dep_price) * qty_dep + (ind_price - trade[f"{ind_asset}_entry"]) * qty_ind

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