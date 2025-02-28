import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(data, signals_df, initial_capital, n_shares, commission):
    """
    Runs a backtest on the pairs trading strategy.
    
    For every timestamp in data:
      - If a signal is present in signals_df, use it; else, assume no signal.
      - Open/close positions based on signals.
      - Update capital and compute portfolio value.
      
    Returns:
      - portfolio_df: DataFrame with portfolio value over time.
      - initial_capital: the initial capital.
      - final_value: the final portfolio value.
      - trades_count: the total number of trades executed.
      - win_rate: percentage of profitable trades.
    """
    capital = initial_capital
    active_long_positions = []   # long Asset1, short Asset2
    active_short_positions = []  # short Asset1, long Asset2
    portfolio_value = []
    trades_count = 0
    trades_profit = []

    for timestamp, row in data.iterrows():
        # Use signal if available; otherwise, default to 0.
        if timestamp in signals_df.index:
            signal = signals_df.loc[timestamp, "Signal_Asset1"]
        else:
            signal = 0

        price1 = row[data.columns[0]]
        price2 = row[data.columns[1]]

        # Close long positions if signal is no longer long.
        if active_long_positions and signal != 1:
            for position in active_long_positions:
                profit_asset1 = (price1 - position['price1']) * n_shares
                profit_asset2 = (position['price2'] - price2) * n_shares
                trade_profit = profit_asset1 + profit_asset2
                closing_cost = (price1 * n_shares + price2 * n_shares) * commission
                capital += trade_profit - closing_cost
                trades_profit.append(trade_profit - closing_cost)
                trades_count += 1
            active_long_positions = []

        # Close short positions if signal is no longer short.
        if active_short_positions and signal != -1:
            for position in active_short_positions:
                profit_asset1 = (position['price1'] - price1) * n_shares
                profit_asset2 = (price2 - position['price2']) * n_shares
                trade_profit = profit_asset1 + profit_asset2
                closing_cost = (price1 * n_shares + price2 * n_shares) * commission
                capital += trade_profit - closing_cost
                trades_profit.append(trade_profit - closing_cost)
                trades_count += 1
            active_short_positions = []

        # Open new long position if signal is long and no long position exists.
        if signal == 1 and not active_long_positions:
            cost_asset1 = price1 * n_shares * (1 + commission)
            cost_asset2 = price2 * n_shares * commission
            total_cost = cost_asset1 + cost_asset2
            if capital > total_cost and capital > 250_000:
                capital -= total_cost
                active_long_positions.append({'date': timestamp, 'price1': price1, 'price2': price2})

        # Open new short position if signal is short and no short position exists.
        if signal == -1 and not active_short_positions:
            cost_asset1 = price1 * n_shares * commission
            cost_asset2 = price2 * n_shares * (1 + commission)
            total_cost = cost_asset1 + cost_asset2
            if capital > total_cost and capital > 250_000:
                capital -= total_cost
                active_short_positions.append({'date': timestamp, 'price1': price1, 'price2': price2})

        # Calculate unrealized P&L for active positions.
        long_pnl = sum([(price1 - pos['price1'])*n_shares + (pos['price2'] - price2)*n_shares for pos in active_long_positions])
        short_pnl = sum([(pos['price1'] - price1)*n_shares + (price2 - pos['price2'])*n_shares for pos in active_short_positions])
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
