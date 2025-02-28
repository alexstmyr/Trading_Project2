import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_signals(spread_vec):
    """
    Generates trading signals based on the normalized dynamic spread.
    Mean-reversion assumption:
      - If normalized spread < -1.5 => signal = +1 (Buy MSFT, Sell AMD)
      - If normalized spread > +1.5  => signal = -1 (Sell MSFT, Buy AMD)
      - Else => signal = 0
    """
    mean_ = spread_vec.mean()
    std_ = spread_vec.std()
    normalized_spread = (spread_vec - mean_) / std_
    
    signals = np.zeros(len(normalized_spread), dtype=int)
    signals[normalized_spread < -1.5] = 1
    signals[normalized_spread >  1.5] = -1
    
    signals_df = pd.DataFrame({
        'Normalized Spread': normalized_spread,
        'Signal': signals
    }, index=spread_vec.index)
    
    return signals_df, mean_, std_

def plot_strategy(data, tickers, spread_vec, signals_df):
    """
    Plots:
      1) Upper panel: MSFT & AMD prices with signals
         - Buy MSFT / Sell AMD markers on each asset
         - Sell MSFT / Buy AMD markers on each asset
      2) Lower panel: Normalized spread with lines at ±1.5σ and 0

    Ensures only ONE legend label for each buy/sell action.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    
    ax_msft = axes[0]
    ax_amd = ax_msft.twinx()
    
    # Plot the two assets
    ax_msft.plot(data.index, data[tickers[0]], color='blue', label=tickers[0])
    ax_msft.set_ylabel(f'{tickers[0]} Price', color='blue')
    
    ax_amd.plot(data.index, data[tickers[1]], color='orange', label=tickers[1])
    ax_amd.set_ylabel(f'{tickers[1]} Price', color='orange')
    
    # Identify buy/sell signals
    sig = signals_df['Signal']
    buy_signals_index = sig[sig == 1].index
    sell_signals_index = sig[sig == -1].index

    # We'll do each label once:
    buy_msft_label_used = False
    sell_amd_label_used = False
    sell_msft_label_used = False
    buy_amd_label_used = False

    # Signal = +1 => Buy MSFT, Sell AMD
    for idx in buy_signals_index:
        msft_price = data.loc[idx, tickers[0]]
        amd_price  = data.loc[idx, tickers[1]]
        
        if not buy_msft_label_used:
            ax_msft.scatter(idx, msft_price, marker='^', color='green', s=100, label='Buy ' + tickers[0])
            buy_msft_label_used = True
        else:
            ax_msft.scatter(idx, msft_price, marker='^', color='green', s=100)
        
        if not sell_amd_label_used:
            ax_amd.scatter(idx, amd_price, marker='v', color='red', s=100, label='Sell ' + tickers[1])
            sell_amd_label_used = True
        else:
            ax_amd.scatter(idx, amd_price, marker='v', color='red', s=100)
    
    # Signal = -1 => Sell MSFT, Buy AMD
    for idx in sell_signals_index:
        msft_price = data.loc[idx, tickers[0]]
        amd_price  = data.loc[idx, tickers[1]]
        
        if not sell_msft_label_used:
            ax_msft.scatter(idx, msft_price, marker='v', color='red', s=100, label='Sell ' + tickers[0])
            sell_msft_label_used = True
        else:
            ax_msft.scatter(idx, msft_price, marker='v', color='red', s=100)
        
        if not buy_amd_label_used:
            ax_amd.scatter(idx, amd_price, marker='^', color='green', s=100, label='Buy ' + tickers[1])
            buy_amd_label_used = True
        else:
            ax_amd.scatter(idx, amd_price, marker='^', color='green', s=100)
    
    # Combine legends
    lines1, labels1 = ax_msft.get_legend_handles_labels()
    lines2, labels2 = ax_amd.get_legend_handles_labels()
    ax_msft.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Lower panel: Normalized spread
    ax_spread = axes[1]
    mean_ = spread_vec.mean()
    std_  = spread_vec.std()
    norm_spread = (spread_vec - mean_) / std_
    
    ax_spread.plot(norm_spread.index, norm_spread, color='blue', label='Normalized Spread')
    ax_spread.axhline(1.5, color='orange', linestyle='--', label='+1.5σ')
    ax_spread.axhline(-1.5, color='orange', linestyle='--')
    ax_spread.axhline(0, color='black', linestyle='-', label='0')
    ax_spread.legend(loc='upper left')
    
    plt.show()
