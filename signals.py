import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_signals_with_indices(spread_vec):
    spread_mean = spread_vec.mean()
    spread_std = spread_vec.std()
    normalized_spread = (spread_vec - spread_mean) / spread_std

    n = len(normalized_spread)
    signals = np.zeros(n, dtype=int)
    long_signals = []
    short_signals = []
    close_signals = []
    position = None

    # Use .iloc for positional indexing.
    for i in range(n):
        value = normalized_spread.iloc[i]
        if position is None:
            if value < -1.5:
                signals[i] = 1
                long_signals.append(i)
                position = 'long'
            elif value > 1.5:
                signals[i] = -1
                short_signals.append(i)
                position = 'short'
        else:
            if position == 'long' and value >= 0:
                signals[i] = 0
                close_signals.append(i)
                position = None
            elif position == 'short' and value <= 0:
                signals[i] = 0
                close_signals.append(i)
                position = None

    signals_opp = -signals
    signals_df = pd.DataFrame({
        'Normalized Spread': normalized_spread,
        'Signal_Asset1': signals,
        'Signal_Asset2': signals_opp
    }, index=spread_vec.index)

    return signals_df, long_signals, short_signals, close_signals, spread_mean, spread_std

def plot_combined_chart(data, tickers, spread_vec, signals_df):
    """
    Plots asset prices with buy/sell signals and the normalized spread.
    
    - Upper panel: plots the price series for tickers[0] (left axis) and tickers[1] (right axis).
    - Lower panel: plots the normalized spread along with threshold lines.
    """
    fig, ax1 = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1]})
    
    # Upper panel: plot asset prices.
    ax1[0].plot(data.index, data[tickers[0]], label=tickers[0], color='blue')
    ax1[0].set_ylabel(f'{tickers[0]} Price', color='blue')
    ax2 = ax1[0].twinx()
    ax2.plot(data.index, data[tickers[1]], label=tickers[1], color='orange')
    ax2.set_ylabel(f'{tickers[1]} Price', color='orange')
    
    # Extract signals.
    buy_asset1 = signals_df[signals_df["Signal_Asset1"] == 1]
    sell_asset1 = signals_df[signals_df["Signal_Asset1"] == -1]
    buy_asset2 = signals_df[signals_df["Signal_Asset2"] == 1]
    sell_asset2 = signals_df[signals_df["Signal_Asset2"] == -1]
    
    ax1[0].scatter(buy_asset1.index, data.loc[buy_asset1.index, tickers[0]], 
                   marker='^', color='green', s=100, label='Buy ' + tickers[0])
    ax1[0].scatter(sell_asset1.index, data.loc[sell_asset1.index, tickers[0]], 
                   marker='v', color='red', s=100, label='Sell ' + tickers[0])
    ax2.scatter(buy_asset2.index, data.loc[buy_asset2.index, tickers[1]], 
                   marker='^', color='purple', s=100, label='Buy ' + tickers[1])
    ax2.scatter(sell_asset2.index, data.loc[sell_asset2.index, tickers[1]], 
                   marker='v', color='orange', s=100, label='Sell ' + tickers[1])
    
    # Lower panel: plot normalized spread.
    # Here, we assume spread_vec is the raw spread; we normalize it for plotting.
    norm_mean = spread_vec.mean()
    norm_std = spread_vec.std()
    normalized_spread = (spread_vec - norm_mean) / norm_std
    ax1[1].plot(signals_df.index, normalized_spread, color='blue', label='Normalized Spread')
    ax1[1].axhline(1.5, color='orange', linestyle='--', label='+1.5σ')
    ax1[1].axhline(-1.5, color='red', linestyle='--', label='-1.5σ')
    ax1[1].set_ylabel('Normalized Spread')
    
    plt.legend()
    plt.show()


