import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_signals(spread_vec):
    """
    Generates trading signals based on the normalized dynamic spread.
    
    Trading rules:
      - If normalized spread < -1.5, set Signal = +1 (i.e., Buy MSFT, Sell AMD).
      - If normalized spread >  1.5, set Signal = -1 (i.e., Sell MSFT, Buy AMD).
      - If normalized spread is near 0, Close the signal = 0.
    
    Returns:
      - signals_df: DataFrame with 'Normalized Spread' and 'Signal' columns.
      - spread_mean and spread_std.
    """
    mean_ = spread_vec.mean()
    std_ = spread_vec.std()
    normalized_spread = (spread_vec - mean_) / std_
    
    signals = np.zeros(len(normalized_spread), dtype=int)
    signals[normalized_spread < -1.5] = 1
    signals[normalized_spread > 1.5] = -1
    signals[np.absolute(normalized_spread) < 0.05] = 0
    
    signals_df = pd.DataFrame({
        'Normalized Spread': normalized_spread,
        'Signal': signals
    }, index=spread_vec.index)
    
    return signals_df, mean_, std_

def plot_strategy(data, tickers, spread_vec, signals_df):
    """
    Plots the trading strategy chart.

    Upper panel: Price series for both assets with trade markers.
      - Buy signal (+1): Green upward triangle (^) for dependent asset, red downward triangle (v) for independent asset.
      - Sell signal (-1): Red downward triangle (v) for dependent asset, green upward triangle (^) for independent asset.
      - Close signal (0, when spread ~ 0): **Black 'X'** for both assets.
    
    Lower panel: Plot the normalized dynamic spread with horizontal lines at +1.5, -1.5, and 0.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    
    # Create axes for prices:
    ax_dep = axes[0]
    ax_ind = ax_dep.twinx()
    dep_asset = tickers[0]
    ind_asset = tickers[1]
    
    ax_dep.plot(data.index, data[dep_asset], color='blue', label=dep_asset)
    ax_dep.set_ylabel(f"{dep_asset} Price", color='blue')
    ax_ind.plot(data.index, data[ind_asset], color='orange', label=ind_asset)
    ax_ind.set_ylabel(f"{ind_asset} Price", color='orange')
    
    # Get signals.
    sig = signals_df['Signal']
    buy_signals = sig[sig == 1].index
    sell_signals = sig[sig == -1].index
    close_signals = sig[(sig == 0) & (abs(signals_df['Normalized Spread']) < 0.05)].index  # Close only when spread ~ 0

    # Plot markers with one legend entry per type.
    dep_buy_plotted = False
    ind_sell_plotted = False
    dep_sell_plotted = False
    ind_buy_plotted = False
    close_plotted = False

    for ts in buy_signals:
        if not dep_buy_plotted:
            ax_dep.scatter(ts, data.loc[ts, dep_asset], marker='^', color='green', s=100, label=f"Buy {dep_asset}")
            dep_buy_plotted = True
        else:
            ax_dep.scatter(ts, data.loc[ts, dep_asset], marker='^', color='green', s=100)
        
        if not ind_sell_plotted:
            ax_ind.scatter(ts, data.loc[ts, ind_asset], marker='v', color='red', s=100, label=f"Sell {ind_asset}")
            ind_sell_plotted = True
        else:
            ax_ind.scatter(ts, data.loc[ts, ind_asset], marker='v', color='red', s=100)

    for ts in sell_signals:
        if not dep_sell_plotted:
            ax_dep.scatter(ts, data.loc[ts, dep_asset], marker='v', color='red', s=100, label=f"Sell {dep_asset}")
            dep_sell_plotted = True
        else:
            ax_dep.scatter(ts, data.loc[ts, dep_asset], marker='v', color='red', s=100)
        
        if not ind_buy_plotted:
            ax_ind.scatter(ts, data.loc[ts, ind_asset], marker='^', color='green', s=100, label=f"Buy {ind_asset}")
            ind_buy_plotted = True
        else:
            ax_ind.scatter(ts, data.loc[ts, ind_asset], marker='^', color='green', s=100)

    # Plot close signals (black 'x')
    for ts in close_signals:
        if not close_plotted:
            ax_dep.scatter(ts, data.loc[ts, dep_asset], marker='x', color='black', s=100, label="Close Position")
            ax_ind.scatter(ts, data.loc[ts, ind_asset], marker='x', color='black', s=100)
            close_plotted = True
        else:
            ax_dep.scatter(ts, data.loc[ts, dep_asset], marker='x', color='black', s=100)
            ax_ind.scatter(ts, data.loc[ts, ind_asset], marker='x', color='black', s=100)

    # Combine legends.
    lines1, labels1 = ax_dep.get_legend_handles_labels()
    lines2, labels2 = ax_ind.get_legend_handles_labels()
    ax_dep.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Lower panel: plot normalized spread.
    ax_spread = axes[1]
    norm_spread = signals_df['Normalized Spread']
    ax_spread.plot(norm_spread.index, norm_spread, color='blue', label='Normalized Spread')
    ax_spread.axhline(1.5, color='orange', linestyle='--', label='+1.5σ')
    ax_spread.axhline(-1.5, color='orange', linestyle='--', label='-1.5σ')
    ax_spread.axhline(0, color='black', linestyle='-', label='0')
    ax_spread.set_ylabel('Normalized Spread')
    ax_spread.legend(loc='upper left')

    plt.show()