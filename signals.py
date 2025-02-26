import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_signals(spread_vec):
    """Generates trading signals based on the spread thresholds."""
    spread_mean = spread_vec.mean()
    spread_std = spread_vec.std()

    signal_1 = np.where(spread_vec > spread_mean + 1.5 * spread_std, -1, 
                        np.where(spread_vec < spread_mean - 1.5 * spread_std, 1, 0))  

    signal_2 = -signal_1  # Opposite trade for the second asset

    signals_df = pd.DataFrame({
        'Spread': spread_vec,
        'Signal_Asset1': signal_1,
        'Signal_Asset2': signal_2
    }, index=spread_vec.index)

    return signals_df, spread_mean, spread_std

def plot_combined_chart(data, tickers, spread_vec, signals_df):
    """Plots asset prices with buy/sell signals and normalized spread."""
    fig, ax1 = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1[0].plot(data.index, data[tickers[0]], label=tickers[0], color='blue')
    ax1[0].set_ylabel(f'{tickers[0]} Price', color='blue')
    ax2 = ax1[0].twinx()
    ax2.plot(data.index, data[tickers[1]], label=tickers[1], color='orange')
    ax2.set_ylabel(f'{tickers[1]} Price', color='orange')

    buy_asset1 = signals_df[signals_df["Signal_Asset1"] == 1]
    sell_asset1 = signals_df[signals_df["Signal_Asset1"] == -1]
    buy_asset2 = signals_df[signals_df["Signal_Asset2"] == 1]
    sell_asset2 = signals_df[signals_df["Signal_Asset2"] == -1]

    ax1[0].scatter(buy_asset1.index, data.loc[buy_asset1.index, tickers[0]], marker='^', color='green', s=100)
    ax1[0].scatter(sell_asset1.index, data.loc[sell_asset1.index, tickers[0]], marker='v', color='red', s=100)
    ax2.scatter(buy_asset2.index, data.loc[buy_asset2.index, tickers[1]], marker='^', color='purple', s=100)
    ax2.scatter(sell_asset2.index, data.loc[sell_asset2.index, tickers[1]], marker='v', color='orange', s=100)

    spread_mean = spread_vec.mean()
    spread_std = spread_vec.std()
    normalized_spread = (spread_vec - spread_mean) / spread_std

    ax1[1].plot(spread_vec.index, normalized_spread, color='blue')
    ax1[1].axhline(1.5, color='orange', linestyle='--')
    ax1[1].axhline(-1.5, color='orange', linestyle='--')

    plt.show()
