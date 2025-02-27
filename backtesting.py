import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(data, signals_df, initial_capital, n_shares, commission):
    """
    Runs a backtest on the pair trading strategy using the provided price data and trading signals.
    
    Strategy:
      - When Signal_Asset1 == 1: go long Asset1 and short Asset2.
      - When Signal_Asset1 == -1: go short Asset1 and long Asset2.
      - Positions are closed when the signal reverts (i.e., signal != current).
    
    Implements a margin account with the given capital.
    
    Returns:
      - portfolio_df: DataFrame with portfolio value over time.
      - initial_capital: the initial capital.
      - final_value: the final portfolio value.
      - trades_count: the total number of trades executed.
      - win_rate: percentage of trades that were profitable.
      
    Also plots the portfolio value over time.
    """
    capital = initial_capital
    active_long_positions = []   # For long pair trades: long Asset1, short Asset2.
    active_short_positions = []  # For short pair trades: short Asset1, long Asset2.
    portfolio_value = [capital]
    trades_count = 0
    trades_profit = []  # Store profit/loss of each closed trade.

    # Iterate over the price data (assumed aligned with signals_df index).
    for timestamp, row in data.iterrows():
        # Get trading signal.
        try:
            signal = signals_df.loc[timestamp, "Signal_Asset1"]
        except KeyError:
            continue

        # Prices for Asset1 and Asset2.
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

        # Open new long pair position if signal is long and no long position exists.
        if signal == 1 and not active_long_positions:
            cost_asset1 = price1 * n_shares * (1 + commission)
            cost_asset2 = price2 * n_shares * commission  # Commission on shorting Asset2.
            total_cost = cost_asset1 + cost_asset2
            if capital > total_cost and capital > 250_000:
                capital -= total_cost
                active_long_positions.append({
                    'date': timestamp,
                    'price1': price1,  # Asset1 buy price.
                    'price2': price2   # Asset2 short price.
                })

        # Open new short pair position if signal is short and no short position exists.
        if signal == -1 and not active_short_positions:
            cost_asset1 = price1 * n_shares * commission  # Commission on shorting Asset1.
            cost_asset2 = price2 * n_shares * (1 + commission)  # Cost to buy Asset2.
            total_cost = cost_asset1 + cost_asset2
            if capital > total_cost and capital > 250_000:
                capital -= total_cost
                active_short_positions.append({
                    'date': timestamp,
                    'price1': price1,  # Asset1 short price.
                    'price2': price2   # Asset2 buy price.
                })

        # Calculate unrealized P&L.
        long_pnl = sum([(price1 - pos['price1'])*n_shares + (pos['price2'] - price2)*n_shares for pos in active_long_positions])
        short_pnl = sum([(pos['price1'] - price1)*n_shares + (price2 - pos['price2'])*n_shares for pos in active_short_positions])
        total_value = capital + long_pnl + short_pnl
        portfolio_value.append(total_value)

    portfolio_df = pd.DataFrame({
         'Date': data.index,
         'Portfolio Value': portfolio_value[1:]  # Align with data dates.
    }).set_index('Date')

    final_value = portfolio_value[-1]
    win_rate = (sum([1 for p in trades_profit if p > 0]) / trades_count * 100) if trades_count > 0 else 0

    # Plot portfolio value over time.
    portfolio_df.plot(title="Portfolio Value Over Time")
    plt.show()
    
    return portfolio_df, initial_capital, final_value, trades_count, win_rate
