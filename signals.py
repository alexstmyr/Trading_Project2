import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_signals(spread_vec):
    """
    Generates trading signals based on the spread thresholds.
    A signal is generated only at the first crossing of the threshold.
    """
    spread_mean = spread_vec.mean()
    spread_std = spread_vec.std()
    
    # Compute raw signals continuously
    raw_signal = np.where(spread_vec > spread_mean + 1.5 * spread_std, -1, 
                   np.where(spread_vec < spread_mean - 1.5 * spread_std, 1, 0))
    
    # Filter signals: mark only the first instance when the spread crosses the threshold.
    filtered_signal = np.zeros_like(raw_signal)
    prev = 0
    for i in range(len(raw_signal)):
        if raw_signal[i] != 0 and prev == 0:
            filtered_signal[i] = raw_signal[i]
        # Reset prev to zero only when signal goes back to 0.
        if raw_signal[i] == 0:
            prev = 0
        else:
            prev = raw_signal[i]
            
    # For the second asset, use the opposite signal.
    filtered_signal_opp = -filtered_signal

    signals_df = pd.DataFrame({
        'Spread': spread_vec,
        'Signal_Asset1': filtered_signal,      # signal for dependent asset (tickers[0])
        'Signal_Asset2': filtered_signal_opp     # opposite signal for independent asset (tickers[1])
    }, index=spread_vec.index)

    return signals_df, spread_mean, spread_std

def plot_combined_chart(data, tickers, spread_vec, signals_df):
    """Plots asset prices with buy/sell signals and normalized spread."""
    fig, ax1 = plt.subplots(2, 1, figsize=(14, 8), sharex=True, 
                              gridspec_kw={'height_ratios': [2, 1]})
    
    # Upper panel: asset prices
    ax1[0].plot(data.index, data[tickers[0]], label=tickers[0], color='blue')
    ax1[0].set_ylabel(f'{tickers[0]} Price', color='blue')
    ax2 = ax1[0].twinx()
    ax2.plot(data.index, data[tickers[1]], label=tickers[1], color='orange')
    ax2.set_ylabel(f'{tickers[1]} Price', color='orange')
    
    # Extract filtered signals for plotting
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
    
    # Lower panel: normalized spread
    spread_mean = spread_vec.mean()
    spread_std = spread_vec.std()
    normalized_spread = (spread_vec - spread_mean) / spread_std
    ax1[1].plot(spread_vec.index, normalized_spread, color='blue', label='Normalized Spread')
    ax1[1].axhline(1.5, color='orange', linestyle='--', label='+1.5 STD')
    ax1[1].axhline(-1.5, color='orange', linestyle='--', label='-1.5 STD')
    ax1[1].set_ylabel('Normalized Spread')
    
    plt.legend()
    plt.show()