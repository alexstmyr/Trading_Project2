import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(data, signals_df, initial_capital, n_shares, commission, close_threshold=0.2):

    capital = initial_capital
    active_long_positions = []   # For long trades (originally: long Asset1, short Asset2)
    active_short_positions = []  # For short trades (originally: short Asset1, long Asset2)
    portfolio_value = []
    trades_count = 0
    trades_profit = []

    for timestamp, row in data.iterrows():
        # Retrieve and invert signal.
        if timestamp in signals_df.index:
            # Invert the signal: +1 becomes -1 and -1 becomes +1.
            signal = -signals_df.loc[timestamp, "Signal_Asset1"]
            norm_spread = signals_df.loc[timestamp, "Normalized Spread"]
            current_hr = signals_df.loc[timestamp, "Dynamic Hedge Ratio"] if "Dynamic Hedge Ratio" in signals_df.columns else 1
        else:
            signal = 0
            norm_spread = None
            current_hr = 1

        # If normalized spread is near zero, force signal=0 (close positions).
        if norm_spread is not None and abs(norm_spread) < close_threshold:
            signal = 0

        price1 = row[data.columns[0]]   # Dependent asset price
        price2 = row[data.columns[1]]   # Independent asset price 

        # Close long positions if signal is no longer long.
        if active_long_positions and signal != 1:
            for pos in active_long_positions:
                qty1 = n_shares
                qty2 = n_shares * pos['hedge_ratio']
                profit_asset1 = (price1 - pos['price1']) * qty1
                profit_asset2 = (pos['price2'] - price2) * qty2
                trade_profit = profit_asset1 + profit_asset2
                closing_cost = ((price1 * qty1) + (price2 * qty2)) * commission
                capital += trade_profit - closing_cost
                trades_profit.append(trade_profit - closing_cost)
                trades_count += 1
            active_long_positions = []

        # Close short positions if signal is no longer short.
        if active_short_positions and signal != -1:
            for pos in active_short_positions:
                qty1 = n_shares
                qty2 = n_shares * pos['hedge_ratio']
                profit_asset1 = (pos['price1'] - price1) * qty1
                profit_asset2 = (price2 - pos['price2']) * qty2
                trade_profit = profit_asset1 + profit_asset2
                closing_cost = ((price1 * qty1) + (price2 * qty2)) * commission
                capital += trade_profit - closing_cost
                trades_profit.append(trade_profit - closing_cost)
                trades_count += 1
            active_short_positions = []

        # Open new long position if signal is long and no long position exists.
        # Note: with inverted signals, a signal of 1 now means we want to go long.
        if signal == 1 and not active_long_positions:
            qty1 = n_shares
            qty2 = n_shares * current_hr
            cost_asset1 = price1 * qty1 * (1 + commission)
            cost_asset2 = price2 * qty2 * commission
            total_cost = cost_asset1 + cost_asset2
            if capital > total_cost and capital > 250_000:
                capital -= total_cost
                active_long_positions.append({
                    'date': timestamp,
                    'price1': price1,
                    'price2': price2,
                    'hedge_ratio': current_hr
                })

        # Open new short position if signal is short and no short position exists.
        if signal == -1 and not active_short_positions:
            qty1 = n_shares
            qty2 = n_shares * current_hr
            cost_asset1 = price1 * qty1 * commission
            cost_asset2 = price2 * qty2 * (1 + commission)
            total_cost = cost_asset1 + cost_asset2
            if capital > total_cost and capital > 250_000:
                capital -= total_cost
                active_short_positions.append({
                    'date': timestamp,
                    'price1': price1,
                    'price2': price2,
                    'hedge_ratio': current_hr
                })

        # Calculate unrealized P&L.
        long_pnl = sum([(price1 - pos['price1'])*n_shares + (pos['price2'] - price2)*n_shares * pos['hedge_ratio'] for pos in active_long_positions])
        short_pnl = sum([(pos['price1'] - price1)*n_shares + (price2 - pos['price2'])*n_shares * pos['hedge_ratio'] for pos in active_short_positions])
        total_value = capital + long_pnl + short_pnl
        portfolio_value.append(total_value)

    portfolio_df = pd.DataFrame({
         'Portfolio Value': portfolio_value
    }, index=data.index)

    final_value = portfolio_value[-1]
    win_rate = (sum([1 for p in trades_profit if p > 0]) / trades_count * 100) if trades_count > 0 else 0

    portfolio_df.plot(title="Portfolio Value Over Time")
    plt.show()
    
    return portfolio_df, initial_capital, final_value, trades_count, win_rate
