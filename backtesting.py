import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(data, signals_df, initial_capital, n_shares, commission, margin_cap=250000):
    """
    Runs a backtest of the pair trading strategy with a dynamic hedge ratio.

    Trading rules:
      - When the signal transitions from 0 to +1: Open a LONG trade (Buy dependent asset, short independent asset).
      - When the signal transitions from 0 to -1: Open a SHORT trade (Short dependent asset, buy independent asset).
      - When the signal transitions from ±1 to 0, close all open trades.
      - Multiple trades may be open concurrently.

    For LONG trades:
      - Entry cost = (dep_price * n_shares * (1+commission)) + (ind_price * (n_shares * hedge_ratio) * commission).
      - Exit proceeds = (dep_exit * n_shares * (1-commission)) + (ind_exit * (n_shares * hedge_ratio) * (1-commission)).
    For SHORT trades:
      - Entry cost = (dep_price * n_shares * commission) + (ind_price * (n_shares * hedge_ratio) * (1+commission)).
      - Exit proceeds = (dep_exit * n_shares * (1-commission)) + (ind_exit * (n_shares * hedge_ratio) * (1-commission)).

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

    # Assume data.columns[0] is the dependent asset  and data.columns[1] is the independent asset.
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

        # --- Trade Exit ---
        if prev_signal != 0 and current_signal == 0 and active_trades:
            for trade in active_trades:
                dep_exit = dep_price
                ind_exit = ind_price
                trade_hr = trade["hedge_ratio"]  # Hedge ratio at entry

                qty_dep = n_shares
                qty_ind = n_shares * trade_hr  # Apply hedge ratio to the independent asset

                # For LONG trades (signal +1).
                if trade['direction'] == 1:
                    exit_proceeds = (dep_exit * qty_dep * (1 - commission)) + (ind_exit * qty_ind * (1 - commission))
                else:
                    # For SHORT trades (signal -1).
                    exit_proceeds = (dep_exit * qty_dep * (1 - commission)) + (ind_exit * qty_ind * (1 - commission))

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
            qty_dep = n_shares
            qty_ind = n_shares * current_hr  # Apply hedge ratio to the independent asset

            if current_signal == 1:
                # LONG trade: Buy dependent asset, short independent asset.
                entry_cost = (dep_price * qty_dep * (1 + commission)) + (ind_price * qty_ind * commission)
                direction = 1
            elif current_signal == -1:
                # SHORT trade: Short dependent asset, buy independent asset.
                entry_cost = (dep_price * qty_dep * commission) + (ind_price * qty_ind * (1 + commission))
                direction = -1

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
